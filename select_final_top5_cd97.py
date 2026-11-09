#!/usr/bin/env python3
"""
select_final_top5_cd97.py

FINAL CD97 hit selection after the completed screening MD.

Selection logic
===============

STEP 1 -- HARD CD97 CONTACT-RETENTION GATE
-----------------------------------------
Use the PRE-MD PLIP interaction table as the definition of the biologically
relevant CD97 residues for each ligand.

For every unique CD97 target residue that PLIP detected before MD:
    - measure the minimum heavy-atom distance to LIG in final.pdb
    - call the residue retained when distance <= --contact-cutoff
      (default 4.5 Angstrom)

A ligand passes only if:
    1. >= --retention-threshold of its PRE-MD PLIP target residues remain
       geometrically engaged in the FINAL MD structure (default 0.90 = 90%);
    2. the original CD97 pocket route remains valid:
         A: original lid route -> at least one original lid residue remains
            contacted in final.pdb
         B: original deep-bypass route -> at least two original SIDE residues
            and at least one original BOTTOM residue remain contacted
       If PRE-MD PLIP satisfied both A and B, retaining either originally
       valid route is sufficient, because the >=90% residue-retention gate is
       independently enforced.
    3. the MD trajectory has no severe protein-ligand clash (<1.2 A).

IMPORTANT:
This gate tests retention of the SAME RESIDUE CONTACTS, not exact interaction
geometry. Example: if PLIP called N775 an H-bond before MD, the script tests
whether N775 remains within the contact cutoff at the final MD structure.
It does NOT claim that the exact H-bond angle/donor geometry persists.

STEP 2 -- STABILITY RANKING AMONG PASSERS ONLY
----------------------------------------------
Only Step-1 passers are ranked.

Score:
    30% P90 ligand heavy-atom RMSD              lower is better
    20% median ligand heavy-atom RMSD           lower is better
    15% RMSD standard deviation                 lower is better
    10% absolute RMSD drift (A/ns)              lower is better
    10% fraction of production frames RMSD<=3A higher is better
    10% median normalized 3.5A protein contact higher is better
     5% global 3.5A contact retention           higher is better

The contact-retention fraction is NOT mixed into the stability score:
it is a hard biological gate first, as requested.

OUTPUTS
-------
final_CD97_selection_all_200.csv
final_CD97_contact_retention_long.csv
final_CD97_step1_passers.csv
TOP5_FINAL_CD97.csv
final_CD97_selection_failures.csv

Optional selection directory:
TOP5_FINAL_CD97/
    TOP5_PDB/
        rank01_..._final_complex.pdb
        ...
    rank01_.../
        final_protein_ligand_complex.pdb
        original_docked_complex.pdb
        ligand.sdf
        trajectory.dcd
        pose_metrics.csv
        ...
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import shutil
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd


TARGET_RESIDUES = {
    685: {"label": "W685", "restype": "TRP", "group": "lid"},
    760: {"label": "F760", "restype": "PHE", "group": "lid"},

    597: {"label": "F597", "restype": "PHE", "group": "side"},
    623: {"label": "F623", "restype": "PHE", "group": "side"},
    753: {"label": "W753", "restype": "TRP", "group": "side"},
    771: {"label": "F771", "restype": "PHE", "group": "side"},
    555: {"label": "L555", "restype": "LEU", "group": "side"},
    757: {"label": "L757", "restype": "LEU", "group": "side"},

    775: {"label": "N775", "restype": "ASN", "group": "bottom"},
    778: {"label": "Q778", "restype": "GLN", "group": "bottom"},
    558: {"label": "S558", "restype": "SER", "group": "bottom"},
    752: {"label": "T752", "restype": "THR", "group": "bottom"},
    622: {"label": "C622", "restype": "CYS", "group": "bottom"},
}

LID = {685, 760}
SIDE = {597, 623, 753, 771, 555, 757}
BOTTOM = {775, 778, 558, 752, 622}


def normalize_molecule_id(text: str) -> str:
    """
    Normalize all of:
      rank110__foo_gnina_complex
      rank110__foo_gnina_complex.pdb
      foo_gnina.sdf
      foo_gnina
    to a common key.
    """
    name = Path(str(text)).name.strip()

    for suffix in (".pdb", ".sdf", ".mol2"):
        if name.lower().endswith(suffix):
            name = name[:-len(suffix)]
            break

    name = re.sub(r"^rank\d+__", "", name, flags=re.IGNORECASE)
    name = re.sub(r"_complex$", "", name, flags=re.IGNORECASE)
    return name


def percentile_quality(series: pd.Series, higher_is_better: bool) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    n = int(s.notna().sum())

    if n <= 1:
        return pd.Series(50.0, index=series.index)

    pct = s.rank(method="average", pct=True)

    if not higher_is_better:
        pct = 1.0 - pct + 1.0 / n

    return 100.0 * pct


def linear_slope(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 3 or np.allclose(x, x[0]):
        return 0.0
    return float(np.polyfit(x, y, 1)[0])


def infer_equilibration_ps(status: dict, metrics: pd.DataFrame) -> float:
    try:
        total_steps = int(status["target_total_steps"])
        production_steps = int(status["target_production_steps"])
        equil_steps = max(0, total_steps - production_steps)

        usable = metrics[(metrics["step"] > 0) & (metrics["time_ps"] > 0)]
        if not usable.empty:
            row = usable.iloc[-1]
            dt_ps = float(row["time_ps"]) / float(row["step"])
            return float(equil_steps) * dt_ps
    except Exception:
        pass

    return 100.0


def load_pre_md_plip(path: Path) -> dict[str, dict[int, set[str]]]:
    """
    normalized molecule key -> target residue number -> PLIP interaction types
    """
    mapping: dict[str, dict[int, set[str]]] = defaultdict(lambda: defaultdict(set))

    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {
            "molecule_id",
            "residue_number",
            "residue_label",
            "interaction_type",
        }

        if not required.issubset(set(reader.fieldnames or [])):
            raise ValueError(
                f"{path} is missing required columns: "
                + ", ".join(sorted(required))
            )

        for row in reader:
            try:
                resnum = int(float(row["residue_number"]))
            except Exception:
                continue

            if resnum not in TARGET_RESIDUES:
                continue

            key = normalize_molecule_id(row["molecule_id"])
            interaction = str(row["interaction_type"]).strip()

            if key and interaction:
                mapping[key][resnum].add(interaction)

    return mapping


def pdb_element(line: str) -> str:
    element = line[76:78].strip().upper() if len(line) >= 78 else ""

    if element:
        return element

    atom_name = re.sub(r"[^A-Za-z]", "", line[12:16]).upper()

    if atom_name.startswith(("CL", "BR")):
        return atom_name[:2]

    return atom_name[:1]


def crystallographic_cell(line: str) -> np.ndarray | None:
    """
    Build a 3x3 triclinic cell matrix from CRYST1.
    Rows are a,b,c vectors and Cartesian = fractional @ H.
    """
    try:
        a = float(line[6:15])
        b = float(line[15:24])
        c = float(line[24:33])
        alpha = math.radians(float(line[33:40]))
        beta = math.radians(float(line[40:47]))
        gamma = math.radians(float(line[47:54]))
    except Exception:
        return None

    sg = math.sin(gamma)

    if abs(sg) < 1.0e-10:
        return None

    va = np.array([a, 0.0, 0.0], dtype=float)
    vb = np.array(
        [b * math.cos(gamma), b * math.sin(gamma), 0.0],
        dtype=float,
    )

    cx = c * math.cos(beta)
    cy = c * (
        math.cos(alpha)
        - math.cos(beta) * math.cos(gamma)
    ) / sg

    cz2 = c * c - cx * cx - cy * cy
    cz = math.sqrt(max(0.0, cz2))
    vc = np.array([cx, cy, cz], dtype=float)

    return np.vstack([va, vb, vc])


def parse_pdb(path: Path) -> tuple[list[dict], np.ndarray | None]:
    atoms = []
    cell = None

    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if line.startswith("CRYST1"):
                cell = crystallographic_cell(line)
                continue

            if not line.startswith(("ATOM  ", "HETATM")):
                continue

            try:
                resid = int(line[22:26].strip())
                xyz = np.array(
                    [
                        float(line[30:38]),
                        float(line[38:46]),
                        float(line[46:54]),
                    ],
                    dtype=float,
                )
            except Exception:
                continue

            atoms.append(
                {
                    "record": line[:6].strip(),
                    "name": line[12:16].strip(),
                    "resname": line[17:20].strip().upper(),
                    "chain": line[21:22].strip(),
                    "resid": resid,
                    "element": pdb_element(line),
                    "xyz": xyz,
                }
            )

    return atoms, cell


def minimum_distance(
    group_a: list[dict],
    group_b: list[dict],
    cell: np.ndarray | None,
) -> float:
    a = np.asarray([atom["xyz"] for atom in group_a], dtype=float)
    b = np.asarray([atom["xyz"] for atom in group_b], dtype=float)

    delta = a[:, None, :] - b[None, :, :]

    if cell is not None:
        inv_cell = np.linalg.inv(cell)
        fractional = delta @ inv_cell
        fractional -= np.round(fractional)
        delta = fractional @ cell

    return float(
        np.sqrt(np.sum(delta * delta, axis=2)).min()
    )


def target_distances(
    pdb_path: Path,
    ligand_resname: str,
) -> dict[int, float]:
    atoms, cell = parse_pdb(pdb_path)

    ligand = [
        atom
        for atom in atoms
        if atom["resname"] == ligand_resname.upper()
        and atom["element"] != "H"
    ]

    if not ligand:
        raise ValueError(
            f"No heavy-atom ligand residue {ligand_resname!r} in {pdb_path}"
        )

    distances = {}

    for resnum, definition in TARGET_RESIDUES.items():
        residue = [
            atom
            for atom in atoms
            if atom["resid"] == resnum
            and atom["resname"] == definition["restype"]
            and atom["element"] != "H"
        ]

        if not residue:
            distances[resnum] = float("nan")
            continue

        distances[resnum] = minimum_distance(
            residue,
            ligand,
            cell,
        )

    return distances


def protein_ligand_min_distance(
    pdb_path: Path,
    ligand_resname: str,
) -> float:
    atoms, cell = parse_pdb(pdb_path)

    ligand = [
        atom
        for atom in atoms
        if atom["resname"] == ligand_resname.upper()
        and atom["element"] != "H"
    ]

    protein = [
        atom
        for atom in atoms
        if atom["record"] == "ATOM" and atom["resname"] != ligand_resname.upper()
        and atom["element"] != "H"
    ]

    if not ligand or not protein:
        return float("nan")

    return minimum_distance(protein, ligand, cell)


def generic_stability(
    run_dir: Path,
) -> tuple[dict, dict] | None:
    status_path = run_dir / "status.json"
    metrics_path = run_dir / "pose_metrics.csv"
    final_pdb = run_dir / "final.pdb"

    # Native fallback for GROMACS CD97_dynamics.py runs
    gmx_dir = run_dir / "gromacs"
    gro_file = gmx_dir / "em.gro"
    xtc_file = gmx_dir / "md_1ns.xtc"
    if not xtc_file.exists():
        xtc_file = gmx_dir / "md_5ns.xtc"  # Legacy output
    if not xtc_file.is_file() and gmx_dir.is_dir():
        xtcs = list(gmx_dir.glob("*.xtc"))
        if xtcs:
            xtc_file = xtcs[0]

    if (not metrics_path.is_file() or not final_pdb.is_file()) and gro_file.is_file() and xtc_file.is_file():
        try:
            import MDAnalysis as mda
            from MDAnalysis.analysis import rms
            from MDAnalysis.lib.distances import distance_array

            u = mda.Universe(str(gro_file), str(xtc_file))
            protein = u.select_atoms("protein")
            ligand = u.select_atoms("resname LIG and not name H*")
            if len(ligand) == 0:
                ligand = u.select_atoms("not protein and not resname SOL and not resname NA and not resname CL and not name H*")

            u.trajectory[0]
            ref_pos = ligand.positions.copy()

            if not final_pdb.is_file():
                u.trajectory[-1]
                final_sel = u.select_atoms("protein or resname LIG or (not protein and not resname SOL and not resname NA and not resname CL)")
                final_sel.write(str(final_pdb))

            if not metrics_path.is_file():
                rows = []
                for i, ts in enumerate(u.trajectory):
                    rmsd = float(rms.rmsd(ligand.positions, ref_pos, superposition=True))
                    d = distance_array(ligand.positions, protein.positions, box=ts.dimensions)
                    min_d = float(np.min(d))
                    c35 = int(np.sum(np.min(d, axis=1) <= 3.5))
                    c45 = int(np.sum(np.min(d, axis=1) <= 4.5))
                    rows.append({
                        "step": i,
                        "time_ps": float(ts.time),
                        "ligand_heavy_rmsd_A": rmsd,
                        "minimum_protein_ligand_A": min_d,
                        "ligand_atoms_contact_3.5A": c35,
                        "ligand_atoms_contact_4.5A": c45,
                    })
                pd.DataFrame(rows).to_csv(metrics_path, index=False)
        except Exception:
            pass

    if not status_path.is_file() or not metrics_path.is_file():
        return None

    try:
        status = json.loads(status_path.read_text())
    except Exception:
        return None

    if status.get("success") is not True:
        return None

    if status.get("stage") != "complete":
        return None

    try:
        metrics = pd.read_csv(metrics_path)
    except Exception:
        return None

    required = [
        "step",
        "time_ps",
        "ligand_heavy_rmsd_A",
        "minimum_protein_ligand_A",
        "ligand_atoms_contact_3.5A",
        "ligand_atoms_contact_4.5A",
    ]

    if not set(required).issubset(metrics.columns):
        return None

    for column in required:
        metrics[column] = pd.to_numeric(
            metrics[column],
            errors="coerce",
        )

    metrics = metrics.dropna(subset=required)

    if metrics.empty:
        return None

    equil_ps = infer_equilibration_ps(status, metrics)

    production = metrics[
        metrics["time_ps"] > equil_ps + 1.0e-8
    ].copy()

    if len(production) < 5:
        production = metrics[
            metrics["time_ps"] >= equil_ps
        ].copy()

    if len(production) < 5:
        return None

    heavy_atoms = int(
        status.get("ligand_heavy_atoms", 0) or 0
    )

    if heavy_atoms <= 0:
        heavy_atoms = int(
            max(
                1,
                production[
                    "ligand_atoms_contact_4.5A"
                ].max(),
            )
        )

    rmsd = production[
        "ligand_heavy_rmsd_A"
    ].to_numpy(dtype=float)

    time_ns = (
        production["time_ps"].to_numpy(dtype=float)
        - float(production["time_ps"].iloc[0])
    ) / 1000.0

    contact35 = production[
        "ligand_atoms_contact_3.5A"
    ].to_numpy(dtype=float)

    contact45 = production[
        "ligand_atoms_contact_4.5A"
    ].to_numpy(dtype=float)

    minimum_distance_series = production[
        "minimum_protein_ligand_A"
    ].to_numpy(dtype=float)

    early_n = max(
        3,
        int(math.ceil(0.10 * len(production))),
    )

    baseline35 = float(
        np.median(contact35[:early_n])
    )

    contact35_threshold = max(
        1.0,
        0.70 * baseline35,
    )

    result = {
        "complex_id": run_dir.name,
        "run_dir": str(run_dir.resolve()),
        "pdb": status.get("pdb", ""),
        "sdf": status.get("sdf", ""),
        "ligand_heavy_atoms": heavy_atoms,
        "equilibration_ps": equil_ps,
        "production_frames": len(production),

        "rmsd_median_A": float(np.median(rmsd)),
        "rmsd_mean_A": float(np.mean(rmsd)),
        "rmsd_p90_A": float(np.percentile(rmsd, 90)),
        "rmsd_max_A": float(np.max(rmsd)),
        "rmsd_std_A": float(np.std(rmsd, ddof=0)),
        "rmsd_slope_A_per_ns": linear_slope(
            time_ns,
            rmsd,
        ),
        "fraction_rmsd_le_3A": float(
            np.mean(rmsd <= 3.0)
        ),

        "global_contact35_median_fraction": float(
            np.median(contact35) / heavy_atoms
        ),
        "global_contact45_median_fraction": float(
            np.median(contact45) / heavy_atoms
        ),
        "global_contact35_retention_fraction": float(
            np.mean(
                contact35 >= contact35_threshold
            )
        ),

        "trajectory_min_protein_ligand_A": float(
            np.min(minimum_distance_series)
        ),
    }

    return result, status


def original_route(
    pre_residues: set[int],
) -> tuple[bool, bool, str]:
    route_a = bool(pre_residues & LID)

    route_b = (
        len(pre_residues & SIDE) >= 2
        and len(pre_residues & BOTTOM) >= 1
    )

    if route_a and route_b:
        label = "A+B"
    elif route_a:
        label = "A_lid"
    elif route_b:
        label = "B_deep_bypass"
    else:
        label = "NONE"

    return route_a, route_b, label


def route_retained(
    pre_residues: set[int],
    retained_pre_residues: set[int],
) -> tuple[bool, str]:
    pre_a, pre_b, pre_label = original_route(
        pre_residues
    )

    final_a = bool(
        retained_pre_residues & LID
    )

    final_b = (
        len(retained_pre_residues & SIDE) >= 2
        and len(retained_pre_residues & BOTTOM) >= 1
    )

    if pre_a and pre_b:
        passed = final_a or final_b
    elif pre_a:
        passed = final_a
    elif pre_b:
        passed = final_b
    else:
        passed = False

    return passed, pre_label


def write_clean_complex(
    source_pdb: Path,
    output_pdb: Path,
    ligand_resname: str,
) -> tuple[int, int]:
    protein_atoms = 0
    ligand_atoms = 0
    lines = []

    with source_pdb.open(
        "r",
        encoding="utf-8",
        errors="replace",
    ) as handle:
        for raw in handle:
            line = raw.rstrip("\n")

            if not line.startswith(("ATOM  ", "HETATM")):
                continue

            resname = line[17:20].strip().upper()

            if resname == ligand_resname.upper() or resname in ("LIG", "UNL", "MOL"):
                het_line = "HETATM" + line[6:]
                lines.append(het_line)
                ligand_atoms += 1
            else:
                lines.append(line)
                protein_atoms += 1

    if protein_atoms == 0:
        raise ValueError(
            f"No protein atoms in {source_pdb}"
        )

    if ligand_atoms == 0:
        raise ValueError(
            f"No {ligand_resname} ligand atoms in {source_pdb}"
        )

    lines.extend(["TER", "END"])

    output_pdb.write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )

    return protein_atoms, ligand_atoms


def short_name(complex_id: str) -> str:
    text = re.sub(
        r"^rank\d+__",
        "",
        complex_id,
        flags=re.IGNORECASE,
    )

    return re.sub(
        r"_complex$",
        "",
        text,
        flags=re.IGNORECASE,
    )


def copy_top5(
    top: pd.DataFrame,
    selection_dir: Path,
    ligand_resname: str,
) -> None:
    selection_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    pdb_dir = selection_dir / "TOP5_PDB"
    pdb_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    manifest_rows = []

    for _, row in top.iterrows():
        rank = int(row["final_rank"])
        complex_id = str(row["complex_id"])
        run_dir = Path(row["run_dir"])

        detail_dir = (
            selection_dir
            / f"rank{rank:02d}_{complex_id}"
        )

        detail_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        original_pdb = Path(
            str(row.get("pdb", ""))
        )

        ligand_sdf = Path(
            str(row.get("sdf", ""))
        )

        if original_pdb.is_file():
            shutil.copy2(
                original_pdb,
                detail_dir / "original_docked_complex.pdb",
            )

        if ligand_sdf.is_file():
            shutil.copy2(
                ligand_sdf,
                detail_dir / "ligand.sdf",
            )

        for filename in [
            "final.pdb",
            "trajectory.dcd",
            "pose_metrics.csv",
            "state.csv",
            "status.json",
            "checkpoint.chk",
            "final_checkpoint.chk",
            "prepared_system.xml",
            "prepared_metadata.json",
            "initial_system.pdb",
        ]:
            source = run_dir / filename

            if source.is_file():
                shutil.copy2(
                    source,
                    detail_dir / filename,
                )

        source_final = run_dir / "final.pdb"

        detail_complex = (
            detail_dir
            / "final_protein_ligand_complex.pdb"
        )

        pdb_name = (
            f"rank{rank:02d}_"
            f"{short_name(complex_id)}"
            f"_final_complex.pdb"
        )

        final_pdb = pdb_dir / pdb_name

        protein_atoms, ligand_atoms = (
            write_clean_complex(
                source_final,
                detail_complex,
                ligand_resname,
            )
        )

        shutil.copy2(
            detail_complex,
            final_pdb,
        )

        manifest_rows.append(
            {
                "rank": rank,
                "complex_id": complex_id,
                "contact_retention_percent":
                    row["premd_contact_retention_percent"],
                "stability_score":
                    row["stability_score"],
                "final_pdb":
                    str(final_pdb.resolve()),
                "protein_atoms":
                    protein_atoms,
                "ligand_atoms":
                    ligand_atoms,
            }
        )

    pd.DataFrame(
        manifest_rows
    ).to_csv(
        selection_dir / "TOP5_PDB_manifest.csv",
        index=False,
    )


def main() -> int:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--output",
        required=True,
        help="CD97_openmm_1ns output directory",
    )

    parser.add_argument(
        "--plip-interactions",
        required=True,
        help="Pre-MD PLIP all_target_interactions.csv",
    )

    parser.add_argument(
        "--contact-cutoff",
        type=float,
        default=4.5,
        help="Final heavy-atom contact cutoff in Angstrom",
    )

    parser.add_argument(
        "--retention-threshold",
        type=float,
        default=0.90,
        help="Hard PRE-MD residue-contact retention threshold",
    )

    parser.add_argument(
        "--ligand-resname",
        default="LIG",
    )

    parser.add_argument(
        "--top",
        type=int,
        default=5,
    )

    parser.add_argument(
        "--selection-dir",
        default=None,
    )

    args = parser.parse_args()

    output = (
        Path(args.output)
        .expanduser()
        .resolve()
    )

    runs_dir = output / "runs"

    if not runs_dir.is_dir():
        raise SystemExit(
            f"Runs directory not found: {runs_dir}"
        )

    plip_path = (
        Path(args.plip_interactions)
        .expanduser()
        .resolve()
    )

    if not plip_path.is_file():
        raise SystemExit(
            f"PLIP interactions file not found: {plip_path}"
        )

    pre_md = load_pre_md_plip(
        plip_path
    )

    rows = []
    long_rows = []
    failures = []

    run_dirs = sorted(
        path
        for path in runs_dir.iterdir()
        if path.is_dir()
    )

    print(
        f"Completed run directories found: "
        f"{len(run_dirs)}"
    )

    print(
        f"Hard retention threshold      : "
        f"{args.retention_threshold*100:.1f}%"
    )

    print(
        f"Final contact cutoff          : "
        f"{args.contact_cutoff:.2f} A"
    )

    print()

    for index, run_dir in enumerate(
        run_dirs,
        start=1,
    ):
        print(
            f"[{index}/{len(run_dirs)}] "
            f"{run_dir.name}",
            flush=True,
        )

        try:
            generic = generic_stability(
                run_dir
            )

            if generic is None:
                failures.append(
                    {
                        "complex_id":
                            run_dir.name,
                        "error":
                            "run not complete/readable",
                    }
                )

                continue

            row, status = generic

            candidate_keys = {
                normalize_molecule_id(
                    run_dir.name
                ),
                normalize_molecule_id(
                    status.get("sdf", "")
                ),
                normalize_molecule_id(
                    status.get("pdb", "")
                ),
            }

            candidate_keys.discard("")

            pre_interactions = None
            matched_key = None

            for key in candidate_keys:
                if key in pre_md:
                    pre_interactions = (
                        pre_md[key]
                    )
                    matched_key = key
                    break

            if not pre_interactions:
                row.update(
                    {
                        "step1_pass": False,
                        "step1_failure_reason":
                            "no_preMD_PLIP_mapping",
                        "premd_molecule_key":
                            "",
                        "premd_target_residue_count":
                            0,
                        "premd_contact_retention_fraction":
                            0.0,
                        "premd_contact_retention_percent":
                            0.0,
                        "original_route":
                            "NONE",
                        "original_route_retained":
                            False,
                    }
                )

                rows.append(row)
                continue

            pre_residues = set(
                pre_interactions
            )

            final_pdb = (
                run_dir / "final.pdb"
            )

            if not final_pdb.is_file():
                raise FileNotFoundError(
                    f"Missing {final_pdb}"
                )

            final_distances = (
                target_distances(
                    final_pdb,
                    args.ligand_resname,
                )
            )

            retained = {
                residue
                for residue in pre_residues
                if (
                    np.isfinite(
                        final_distances.get(
                            residue,
                            np.nan,
                        )
                    )
                    and final_distances[
                        residue
                    ] <= args.contact_cutoff
                )
            }

            retention_fraction = (
                len(retained)
                / len(pre_residues)
            )

            route_ok, route_label = (
                route_retained(
                    pre_residues,
                    retained,
                )
            )

            final_min_distance = (
                protein_ligand_min_distance(
                    final_pdb,
                    args.ligand_resname,
                )
            )

            severe_clash = (
                np.isfinite(final_min_distance)
                and final_min_distance < 1.20
            )

            reasons = []

            if retention_fraction < (
                args.retention_threshold
            ):
                reasons.append(
                    f"contact_retention_"
                    f"{100*retention_fraction:.1f}%"
                    f"_below_"
                    f"{100*args.retention_threshold:.1f}%"
                )

            if not route_ok:
                reasons.append(
                    "original_CD97_route_not_retained"
                )

            if severe_clash:
                reasons.append(
                    "final_severe_clash_<1.2A"
                )

            step1_pass = (
                retention_fraction
                >= args.retention_threshold
                and route_ok
                and not severe_clash
            )

            row.update(
                {
                    "premd_molecule_key":
                        matched_key,
                    "premd_target_residue_count":
                        len(pre_residues),
                    "premd_target_residues":
                        ";".join(
                            TARGET_RESIDUES[r][
                                "label"
                            ]
                            for r in sorted(
                                pre_residues
                            )
                        ),
                    "final_retained_target_residue_count":
                        len(retained),
                    "final_retained_target_residues":
                        ";".join(
                            TARGET_RESIDUES[r][
                                "label"
                            ]
                            for r in sorted(
                                retained
                            )
                        ),
                    "premd_contact_retention_fraction":
                        retention_fraction,
                    "premd_contact_retention_percent":
                        100.0
                        * retention_fraction,
                    "original_route":
                        route_label,
                    "original_route_retained":
                        route_ok,
                    "final_min_protein_ligand_A":
                        final_min_distance,
                    "step1_pass":
                        step1_pass,
                    "step1_failure_reason":
                        ";".join(reasons),
                }
            )

            for residue in sorted(
                TARGET_RESIDUES
            ):
                definition = (
                    TARGET_RESIDUES[
                        residue
                    ]
                )

                interaction_types = (
                    sorted(
                        pre_interactions.get(
                            residue,
                            set(),
                        )
                    )
                )

                pre_present = (
                    residue
                    in pre_residues
                )

                distance = (
                    final_distances.get(
                        residue,
                        np.nan,
                    )
                )

                retained_flag = (
                    pre_present
                    and np.isfinite(
                        distance
                    )
                    and distance
                    <= args.contact_cutoff
                )

                long_rows.append(
                    {
                        "complex_id":
                            run_dir.name,
                        "molecule_key":
                            matched_key,
                        "residue_label":
                            definition[
                                "label"
                            ],
                        "residue_number":
                            residue,
                        "group":
                            definition[
                                "group"
                            ],
                        "premd_PLIP_present":
                            pre_present,
                        "premd_PLIP_interaction_types":
                            "|".join(
                                interaction_types
                            ),
                        "final_min_heavy_atom_distance_A":
                            distance,
                        "final_contact_cutoff_A":
                            args.contact_cutoff,
                        "retained_from_preMD":
                            retained_flag,
                    }
                )

            rows.append(row)

        except Exception as exc:
            failures.append(
                {
                    "complex_id":
                        run_dir.name,
                    "error":
                        str(exc),
                }
            )

    if not rows:
        raise SystemExit(
            "No completed systems were analyzed."
        )

    table = pd.DataFrame(rows)

    passers = table[
        table["step1_pass"] == True
    ].copy()

    print()
    print("STEP 1 — CD97 CONTACT RETENTION")
    print("=" * 70)
    print(
        f"Analyzed systems : {len(table)}"
    )
    print(
        f"Passed >=90% gate: {len(passers)}"
    )
    print(
        f"Failed gate      : "
        f"{len(table)-len(passers)}"
    )
    print(
        f"Analysis failures: "
        f"{len(failures)}"
    )

    # STEP 2: percentile scoring is intentionally computed
    # ONLY among Step-1 passers.
    if not passers.empty:
        passers[
            "q_rmsd_p90"
        ] = percentile_quality(
            passers["rmsd_p90_A"],
            False,
        )

        passers[
            "q_rmsd_median"
        ] = percentile_quality(
            passers["rmsd_median_A"],
            False,
        )

        passers[
            "q_rmsd_std"
        ] = percentile_quality(
            passers["rmsd_std_A"],
            False,
        )

        passers[
            "abs_rmsd_slope_A_per_ns"
        ] = passers[
            "rmsd_slope_A_per_ns"
        ].abs()

        passers[
            "q_rmsd_slope"
        ] = percentile_quality(
            passers[
                "abs_rmsd_slope_A_per_ns"
            ],
            False,
        )

        passers[
            "q_fraction_rmsd_under3"
        ] = percentile_quality(
            passers[
                "fraction_rmsd_le_3A"
            ],
            True,
        )

        passers[
            "q_global_contact35"
        ] = percentile_quality(
            passers[
                "global_contact35_median_fraction"
            ],
            True,
        )

        passers[
            "q_global_contact_retention"
        ] = percentile_quality(
            passers[
                "global_contact35_retention_fraction"
            ],
            True,
        )

        passers[
            "stability_score"
        ] = (
            0.30
            * passers[
                "q_rmsd_p90"
            ]
            + 0.20
            * passers[
                "q_rmsd_median"
            ]
            + 0.15
            * passers[
                "q_rmsd_std"
            ]
            + 0.10
            * passers[
                "q_rmsd_slope"
            ]
            + 0.10
            * passers[
                "q_fraction_rmsd_under3"
            ]
            + 0.10
            * passers[
                "q_global_contact35"
            ]
            + 0.05
            * passers[
                "q_global_contact_retention"
            ]
        )

        passers = passers.sort_values(
            by=[
                "stability_score",
                "premd_contact_retention_fraction",
                "rmsd_p90_A",
            ],
            ascending=[
                False,
                False,
                True,
            ],
        ).reset_index(
            drop=True
        )

        passers[
            "final_rank"
        ] = np.arange(
            1,
            len(passers) + 1,
        )

        rank_map = dict(
            zip(
                passers[
                    "complex_id"
                ],
                passers[
                    "final_rank"
                ],
            )
        )

        score_map = dict(
            zip(
                passers[
                    "complex_id"
                ],
                passers[
                    "stability_score"
                ],
            )
        )

        table[
            "final_rank"
        ] = table[
            "complex_id"
        ].map(rank_map)

        table[
            "stability_score"
        ] = table[
            "complex_id"
        ].map(score_map)

    else:
        table[
            "final_rank"
        ] = np.nan

        table[
            "stability_score"
        ] = np.nan

    all_csv = (
        output
        / "final_CD97_selection_all_200.csv"
    )

    long_csv = (
        output
        / "final_CD97_contact_retention_long.csv"
    )

    pass_csv = (
        output
        / "final_CD97_step1_passers.csv"
    )

    top_csv = (
        output
        / f"TOP{args.top}_FINAL_CD97.csv"
    )

    fail_csv = (
        output
        / "final_CD97_selection_failures.csv"
    )

    table.to_csv(
        all_csv,
        index=False,
    )

    pd.DataFrame(
        long_rows
    ).to_csv(
        long_csv,
        index=False,
    )

    passers.to_csv(
        pass_csv,
        index=False,
    )

    top = passers.head(
        args.top
    ).copy()

    top.to_csv(
        top_csv,
        index=False,
    )

    pd.DataFrame(
        failures
    ).to_csv(
        fail_csv,
        index=False,
    )

    print()
    print("STEP 2 — STABILITY RANKING")
    print("=" * 70)

    if top.empty:
        print(
            "No molecules passed the hard "
            "CD97 retention gate."
        )
    else:
        print(
            f"Top candidates returned: "
            f"{len(top)}"
        )

        if len(top) < args.top:
            print(
                f"WARNING: only {len(top)} "
                f"molecules passed; the script "
                f"will NOT force a top {args.top}."
            )

        display_columns = [
            "final_rank",
            "complex_id",
            "premd_contact_retention_percent",
            "original_route",
            "stability_score",
            "rmsd_median_A",
            "rmsd_p90_A",
            "rmsd_std_A",
            "fraction_rmsd_le_3A",
            "premd_target_residues",
            "final_retained_target_residues",
        ]

        with pd.option_context(
            "display.max_colwidth",
            55,
            "display.width",
            260,
        ):
            print(
                top[
                    display_columns
                ].to_string(
                    index=False,
                    float_format=lambda x:
                        f"{x:.3f}",
                )
            )

    print()
    print("OUTPUT FILES")
    print("=" * 70)
    print(
        f"All 200 audit     : {all_csv}"
    )
    print(
        f"Per-residue audit : {long_csv}"
    )
    print(
        f"Step-1 passers    : {pass_csv}"
    )
    print(
        f"Top candidates    : {top_csv}"
    )
    print(
        f"Failures          : {fail_csv}"
    )

    if args.selection_dir and not top.empty:
        selection_dir = (
            Path(args.selection_dir)
            .expanduser()
            .resolve()
        )

        copy_top5(
            top,
            selection_dir,
            args.ligand_resname,
        )

        shutil.copy2(
            top_csv,
            selection_dir
            / top_csv.name,
        )

        print()
        print(
            "FINAL PROTEIN-LIGAND PDBs:"
        )
        print(
            selection_dir / "TOP5_PDB"
        )

        for pdb in sorted(
            (
                selection_dir
                / "TOP5_PDB"
            ).glob("*.pdb")
        ):
            print(
                f"  {pdb.name}"
            )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
