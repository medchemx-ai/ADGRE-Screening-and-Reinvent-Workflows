#!/usr/bin/env python3
"""
select_top5_cd97_persistent.py

Rank completed CD97 OpenMM screening trajectories using BOTH:
1) generic pose stability (ligand RMSD + global pocket contacts), and
2) persistence of the CD97 target-residue interactions that were present
   BEFORE MD according to the prior PLIP analysis.

CD97 residues
-------------
Lid:
    W685, F760
Side:
    F597, F623, W753, F771, L555, L757
Bottom:
    N775, Q778, S558, T752, C622

Pre-MD interaction source
-------------------------
Preferred:
    PLIP_GLOBAL_ANALYSIS/all_target_interactions.csv

The script reads the exact PLIP residue labels and interaction types that
were present before dynamics.  During MD it measures residue-contact
persistence geometrically as the fraction of production frames in which
the minimum heavy-atom distance between that residue and LIG is <= cutoff
(default 4.0 A).

IMPORTANT:
This answers "does the same residue remain engaged?"  It does NOT claim
that the exact PLIP interaction geometry (e.g. the same H-bond or pi stack)
is present in every frame.  Exact interaction-type occupancy requires a
separate geometry-specific H-bond/pi analysis.

The MD trajectory is OpenMM DCD.  A small, dependency-free DCD reader is
included, so MDTraj/MDAnalysis/AmberTools are NOT required.

Final score
-----------
50% generic pose stability
25% weighted persistence of PRE-MD PLIP target residues
    (lid weight 4; side/bottom weight 2)
15% persistence of the ORIGINAL pocket route:
    A = any lid contact
    B = >=2 side contacts AND >=1 bottom contact
10% persistence of any valid CD97 pocket route during MD

Outputs
-------
CD97_persistence_ranking_all_200.csv
CD97_residue_persistence_long.csv
TOP5_CD97_stable_persistent.csv
TOP5_CD97_STABLE_PERSISTENT/  (optional copied files)\nTOP5_CD97_STABLE_PERSISTENT/TOP5_PDB/  (five final protein-ligand PDBs)
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import shutil
import struct
import sys
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


TARGET_RESIDUES = {
    685: {"label": "W685", "restype": "TRP", "group": "lid", "weight": 4.0},
    760: {"label": "F760", "restype": "PHE", "group": "lid", "weight": 4.0},

    597: {"label": "F597", "restype": "PHE", "group": "side", "weight": 2.0},
    623: {"label": "F623", "restype": "PHE", "group": "side", "weight": 2.0},
    753: {"label": "W753", "restype": "TRP", "group": "side", "weight": 2.0},
    771: {"label": "F771", "restype": "PHE", "group": "side", "weight": 2.0},
    555: {"label": "L555", "restype": "LEU", "group": "side", "weight": 2.0},
    757: {"label": "L757", "restype": "LEU", "group": "side", "weight": 2.0},

    775: {"label": "N775", "restype": "ASN", "group": "bottom", "weight": 2.0},
    778: {"label": "Q778", "restype": "GLN", "group": "bottom", "weight": 2.0},
    558: {"label": "S558", "restype": "SER", "group": "bottom", "weight": 2.0},
    752: {"label": "T752", "restype": "THR", "group": "bottom", "weight": 2.0},
    622: {"label": "C622", "restype": "CYS", "group": "bottom", "weight": 2.0},
}

LABEL_TO_RESNUM = {v["label"]: k for k, v in TARGET_RESIDUES.items()}
LID = {685, 760}
SIDE = {597, 623, 753, 771, 555, 757}
BOTTOM = {775, 778, 558, 752, 622}

STANDARD_RESIDUES = {
    "ALA", "ARG", "ASN", "ASP", "ASH", "CYS", "CYM", "CYX", "GLN", "GLU",
    "GLH", "GLY", "HIS", "HID", "HIE", "HIP", "ILE", "LEU", "LYS", "LYN",
    "MET", "MSE", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL",
}


def percentile_quality(series: pd.Series, higher_is_better: bool) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    n = int(s.notna().sum())
    if n <= 1:
        return pd.Series(50.0, index=series.index)
    pct = s.rank(method="average", pct=True)
    if not higher_is_better:
        pct = 1.0 - pct + 1.0 / n
    return 100.0 * pct


def infer_equilibration_ps(status: dict, metrics: pd.DataFrame) -> float:
    try:
        total_steps = int(status["target_total_steps"])
        production_steps = int(status["target_production_steps"])
        equil_steps = max(0, total_steps - production_steps)
        usable = metrics[(metrics["step"] > 0) & (metrics["time_ps"] > 0)]
        if not usable.empty:
            row = usable.iloc[-1]
            dt_ps = float(row["time_ps"]) / float(row["step"])
            return equil_steps * dt_ps
    except Exception:
        pass
    return 100.0


def linear_slope(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 3 or np.allclose(x, x[0]):
        return 0.0
    return float(np.polyfit(x, y, 1)[0])


def normalized_molecule_key(text: str) -> str:
    """
    Convert:
      rank001__MOL_gnina_complex
      rank001__MOL_gnina_complex.pdb
    to:
      MOL_gnina
    matching the molecule_id used by the PLIP selector.
    """
    name = Path(str(text)).name
    if name.lower().endswith(".pdb"):
        name = name[:-4]
    name = re.sub(r"^rank\d+__", "", name, flags=re.IGNORECASE)
    if name.lower().endswith("_complex"):
        name = name[:-8]
    return name


def locate_plip_interactions(output_root: Path, supplied: str | None) -> Path | None:
    if supplied:
        path = Path(supplied).expanduser().resolve()
        return path if path.is_file() else None

    candidates = [
        output_root.parent.parent / "all_target_interactions.csv",
        output_root.parent / "all_target_interactions.csv",
        output_root / "all_target_interactions.csv",
    ]
    for path in candidates:
        if path.is_file():
            return path.resolve()
    return None


def load_plip_interactions(path: Path | None) -> dict[str, dict[int, set[str]]]:
    """
    molecule_id -> residue_number -> set(interaction types)
    """
    mapping: dict[str, dict[int, set[str]]] = defaultdict(lambda: defaultdict(set))
    if path is None:
        return mapping

    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = set(reader.fieldnames or [])
        needed = {"molecule_id", "residue_number", "residue_label", "interaction_type"}
        if not needed.issubset(fields):
            raise ValueError(
                f"{path} does not look like the CD97 PLIP all_target_interactions.csv. "
                f"Required columns: {sorted(needed)}"
            )

        for row in reader:
            try:
                resnum = int(float(row["residue_number"]))
            except Exception:
                continue
            if resnum not in TARGET_RESIDUES:
                continue
            molecule_id = str(row["molecule_id"]).strip()
            itype = str(row["interaction_type"]).strip()
            if molecule_id and itype:
                mapping[molecule_id][resnum].add(itype)
    return mapping


def pdb_atom_element(line: str) -> str:
    element = line[76:78].strip().upper() if len(line) >= 78 else ""
    if element:
        return element
    atom_name = re.sub(r"[^A-Za-z]", "", line[12:16]).upper()
    if atom_name.startswith(("CL", "BR")):
        return atom_name[:2]
    return atom_name[:1]


def parse_pdb_topology(path: Path) -> dict:
    """
    Parse atom order exactly as written in the PDB.  DCD atom order from
    OpenMM DCDReporter matches initial_system.pdb atom order.
    """
    atoms = []
    box = None

    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if line.startswith("CRYST1"):
                try:
                    box = np.array([
                        float(line[6:15]),
                        float(line[15:24]),
                        float(line[24:33]),
                    ], dtype=float)
                except Exception:
                    box = None
                continue

            if not line.startswith(("ATOM  ", "HETATM")):
                continue

            try:
                resid = int(line[22:26].strip())
                xyz = np.array([
                    float(line[30:38]),
                    float(line[38:46]),
                    float(line[46:54]),
                ], dtype=float)
            except Exception:
                continue

            atoms.append({
                "index": len(atoms),
                "record": line[:6].strip(),
                "name": line[12:16].strip(),
                "resname": line[17:20].strip().upper(),
                "chain": line[21:22].strip(),
                "resid": resid,
                "element": pdb_atom_element(line),
                "xyz": xyz,
            })

    return {"atoms": atoms, "box": box}


def build_md_atom_selections(topology: dict, ligand_resname: str) -> tuple[np.ndarray, dict[int, np.ndarray]]:
    atoms = topology["atoms"]

    ligand = np.array([
        a["index"] for a in atoms
        if a["resname"] == ligand_resname.upper()
        and a["element"] != "H"
    ], dtype=int)

    if ligand.size == 0:
        raise ValueError(f"No heavy atoms with residue name {ligand_resname} in initial_system.pdb")

    residues: dict[int, np.ndarray] = {}

    for resnum, definition in TARGET_RESIDUES.items():
        idx = [
            a["index"] for a in atoms
            if a["resid"] == resnum
            and a["resname"] == definition["restype"]
            and a["element"] != "H"
        ]
        if not idx:
            raise ValueError(
                f"Target residue {definition['label']} ({definition['restype']} {resnum}) "
                "was not found in initial_system.pdb with the expected residue number/type."
            )
        residues[resnum] = np.array(idx, dtype=int)

    return ligand, residues


def read_fortran_record(handle, endian: str) -> bytes | None:
    marker = handle.read(4)
    if marker == b"":
        return None
    if len(marker) != 4:
        raise EOFError("Truncated DCD record marker")
    n = struct.unpack(endian + "i", marker)[0]
    if n < 0 or n > 2_000_000_000:
        raise ValueError(f"Invalid DCD record length: {n}")
    payload = handle.read(n)
    if len(payload) != n:
        raise EOFError("Truncated DCD record payload")
    closing = handle.read(4)
    if len(closing) != 4:
        raise EOFError("Truncated DCD closing marker")
    n2 = struct.unpack(endian + "i", closing)[0]
    if n2 != n:
        raise ValueError(f"DCD record marker mismatch: {n} != {n2}")
    return payload


def dcd_subset_frames(
    path: Path,
    selected_indices: np.ndarray,
) -> Iterable[tuple[int, np.ndarray, np.ndarray | None]]:
    """
    Yield (frame_index, selected_xyz_A, orthorhombic_box_lengths_A).

    Supports the CHARMM-style little-endian DCD written by OpenMM, and
    also detects big endian.  Coordinates are returned in Angstrom.
    """
    with path.open("rb") as handle:
        first = handle.read(4)
        if len(first) != 4:
            raise ValueError(f"Empty/truncated DCD: {path}")

        little = struct.unpack("<i", first)[0]
        big = struct.unpack(">i", first)[0]
        if little == 84:
            endian = "<"
        elif big == 84:
            endian = ">"
        else:
            raise ValueError(
                f"Unsupported DCD header marker in {path}: little={little}, big={big}"
            )

        # We already consumed the first record marker.
        header = handle.read(84)
        close = handle.read(4)
        if len(header) != 84 or len(close) != 4:
            raise ValueError("Truncated DCD header")
        if header[:4] not in (b"CORD", b"VELD"):
            raise ValueError(f"Unexpected DCD signature: {header[:4]!r}")
        if struct.unpack(endian + "i", close)[0] != 84:
            raise ValueError("Corrupt DCD header closing marker")

        # Title block.
        title = read_fortran_record(handle, endian)
        if title is None:
            raise ValueError("Missing DCD title block")

        # Atom count record.
        natom_payload = read_fortran_record(handle, endian)
        if natom_payload is None or len(natom_payload) < 4:
            raise ValueError("Missing DCD atom-count block")
        natoms = struct.unpack(endian + "i", natom_payload[:4])[0]

        if selected_indices.size and int(selected_indices.max()) >= natoms:
            raise ValueError(
                f"PDB/DCD atom mismatch: selected atom index {selected_indices.max()} "
                f"but DCD contains {natoms} atoms"
            )

        coordinate_record_size = 4 * natoms
        frame_index = 0

        while True:
            record = read_fortran_record(handle, endian)
            if record is None:
                break

            box = None
            if len(record) == 48:
                cell = np.frombuffer(record, dtype=endian + "f8", count=6)
                # CHARMM/OpenMM DCD ordering contains cell lengths at 0,2,5.
                candidate = np.array([cell[0], cell[2], cell[5]], dtype=float)
                if np.all(np.isfinite(candidate)) and np.all(candidate > 0):
                    box = candidate
                x_payload = read_fortran_record(handle, endian)
            else:
                x_payload = record

            if x_payload is None:
                break

            if len(x_payload) != coordinate_record_size:
                raise ValueError(
                    f"Unexpected DCD X record size in frame {frame_index}: "
                    f"{len(x_payload)}; expected {coordinate_record_size}"
                )

            y_payload = read_fortran_record(handle, endian)
            z_payload = read_fortran_record(handle, endian)
            if y_payload is None or z_payload is None:
                raise EOFError("Truncated DCD coordinate frame")
            if len(y_payload) != coordinate_record_size or len(z_payload) != coordinate_record_size:
                raise ValueError("Unexpected DCD Y/Z record size")

            x = np.frombuffer(x_payload, dtype=endian + "f4", count=natoms)[selected_indices]
            y = np.frombuffer(y_payload, dtype=endian + "f4", count=natoms)[selected_indices]
            z = np.frombuffer(z_payload, dtype=endian + "f4", count=natoms)[selected_indices]

            xyz = np.column_stack((x, y, z)).astype(float, copy=False)
            yield frame_index, xyz, box
            frame_index += 1


def minimum_distance_A(
    residue_xyz: np.ndarray,
    ligand_xyz: np.ndarray,
    box_A: np.ndarray | None,
) -> float:
    delta = residue_xyz[:, None, :] - ligand_xyz[None, :, :]

    if box_A is not None and np.all(box_A > 0):
        # Orthorhombic minimum-image convention.  OpenMM addSolvent() used
        # a cubic box for this screening workflow.
        delta -= box_A[None, None, :] * np.round(
            delta / box_A[None, None, :]
        )

    d2 = np.sum(delta * delta, axis=2)
    return float(np.sqrt(np.min(d2)))


def analyze_residue_persistence(
    run_dir: Path,
    status: dict,
    metrics: pd.DataFrame,
    plip_by_molecule: dict[str, dict[int, set[str]]],
    cutoff_A: float,
    ligand_resname: str,
) -> tuple[dict, list[dict]]:
    trajectory = run_dir / "trajectory.dcd"
    topology_pdb = run_dir / "initial_system.pdb"

    if not trajectory.is_file():
        raise FileNotFoundError(f"Missing trajectory: {trajectory}")
    if not topology_pdb.is_file():
        raise FileNotFoundError(f"Missing topology PDB: {topology_pdb}")

    topology = parse_pdb_topology(topology_pdb)
    ligand_indices, residue_indices = build_md_atom_selections(topology, ligand_resname)

    selected_global = np.unique(np.concatenate(
        [ligand_indices] + [residue_indices[r] for r in TARGET_RESIDUES]
    ))
    global_to_local = {int(g): i for i, g in enumerate(selected_global)}

    ligand_local = np.array([global_to_local[int(i)] for i in ligand_indices], dtype=int)
    residue_local = {
        r: np.array([global_to_local[int(i)] for i in idx], dtype=int)
        for r, idx in residue_indices.items()
    }

    equil_ps = infer_equilibration_ps(status, metrics)
    production_row_indices = set(
        np.flatnonzero(metrics["time_ps"].to_numpy(dtype=float) > equil_ps + 1e-8).tolist()
    )
    if len(production_row_indices) < 5:
        production_row_indices = set(
            np.flatnonzero(metrics["time_ps"].to_numpy(dtype=float) >= equil_ps).tolist()
        )

    molecule_key = normalized_molecule_key(run_dir.name)
    premd = plip_by_molecule.get(molecule_key, {})

    # If exact key did not match, try normalized keys from the PLIP CSV.
    if not premd and plip_by_molecule:
        for key, value in plip_by_molecule.items():
            if normalized_molecule_key(key) == molecule_key:
                premd = value
                break

    contact_counts = {r: 0 for r in TARGET_RESIDUES}
    distances = {r: [] for r in TARGET_RESIDUES}
    path_a_frames = 0
    path_b_frames = 0
    valid_path_frames = 0
    production_frames = 0
    dcd_frames = 0

    default_box = topology["box"]

    for frame_idx, subset_xyz, dcd_box in dcd_subset_frames(trajectory, selected_global):
        dcd_frames += 1
        if frame_idx not in production_row_indices:
            continue

        production_frames += 1
        box = dcd_box if dcd_box is not None else default_box
        lig_xyz = subset_xyz[ligand_local]

        frame_contacted = set()

        for resnum in TARGET_RESIDUES:
            res_xyz = subset_xyz[residue_local[resnum]]
            d = minimum_distance_A(res_xyz, lig_xyz, box)
            distances[resnum].append(d)
            if d <= cutoff_A:
                contact_counts[resnum] += 1
                frame_contacted.add(resnum)

        path_a = bool(frame_contacted & LID)
        path_b = (
            len(frame_contacted & SIDE) >= 2
            and len(frame_contacted & BOTTOM) >= 1
        )
        if path_a:
            path_a_frames += 1
        if path_b:
            path_b_frames += 1
        if path_a or path_b:
            valid_path_frames += 1

    if production_frames == 0:
        raise RuntimeError("No production DCD frames were available for persistence analysis")

    # The DCD and pose_metrics reporters use the same interval.
    frame_mismatch = dcd_frames != len(metrics)

    persistence = {
        r: contact_counts[r] / production_frames
        for r in TARGET_RESIDUES
    }

    initial_residues = set(premd)
    initial_lid = initial_residues & LID
    initial_side = initial_residues & SIDE
    initial_bottom = initial_residues & BOTTOM

    pre_path_a = bool(initial_lid)
    pre_path_b = len(initial_side) >= 2 and len(initial_bottom) >= 1

    path_a_fraction = path_a_frames / production_frames
    path_b_fraction = path_b_frames / production_frames
    valid_path_fraction = valid_path_frames / production_frames

    if pre_path_a and pre_path_b:
        # The original pose supported both routes.  Count a frame as retaining
        # the original pocket logic if either originally valid route survives.
        original_route_retention = valid_path_fraction
        original_route = "A+B"
    elif pre_path_a:
        original_route_retention = path_a_fraction
        original_route = "A_lid"
    elif pre_path_b:
        original_route_retention = path_b_fraction
        original_route = "B_deep_bypass"
    else:
        original_route_retention = 0.0
        original_route = "NONE"

    weighted_num = 0.0
    weighted_den = 0.0
    for r in initial_residues:
        weight = float(TARGET_RESIDUES[r]["weight"])
        weighted_num += weight * persistence[r]
        weighted_den += weight
    weighted_initial_retention = weighted_num / weighted_den if weighted_den else 0.0

    def group_initial_mean(group_set: set[int]) -> float | None:
        present = sorted(initial_residues & group_set)
        if not present:
            return None
        return float(np.mean([persistence[r] for r in present]))

    summary = {
        "premd_source": "PLIP" if initial_residues else "PLIP_no_match",
        "premd_molecule_key": molecule_key,
        "premd_target_residues": ";".join(TARGET_RESIDUES[r]["label"] for r in sorted(initial_residues)),
        "premd_target_residue_count": len(initial_residues),
        "premd_lid_count": len(initial_lid),
        "premd_side_count": len(initial_side),
        "premd_bottom_count": len(initial_bottom),
        "premd_path_A_lid": pre_path_a,
        "premd_path_B_bypass": pre_path_b,
        "premd_original_route": original_route,
        "md_contact_cutoff_A": cutoff_A,
        "md_production_contact_frames": production_frames,
        "dcd_frames_total": dcd_frames,
        "pose_metrics_rows": len(metrics),
        "dcd_metrics_frame_mismatch": frame_mismatch,
        "weighted_initial_residue_retention_fraction": weighted_initial_retention,
        "lid_initial_contact_retention_fraction": group_initial_mean(LID),
        "side_initial_contact_retention_fraction": group_initial_mean(SIDE),
        "bottom_initial_contact_retention_fraction": group_initial_mean(BOTTOM),
        "md_path_A_lid_fraction": path_a_fraction,
        "md_path_B_bypass_fraction": path_b_fraction,
        "md_any_valid_path_fraction": valid_path_fraction,
        "original_route_retention_fraction": original_route_retention,
    }

    long_rows = []

    for r, definition in TARGET_RESIDUES.items():
        vals = np.asarray(distances[r], dtype=float)
        types = sorted(premd.get(r, set()))
        summary[f"{definition['label']}_premd_types"] = "|".join(types)
        summary[f"{definition['label']}_premd_present"] = bool(types)
        summary[f"{definition['label']}_persistence_fraction"] = persistence[r]
        summary[f"{definition['label']}_median_min_distance_A"] = float(np.median(vals))
        summary[f"{definition['label']}_p90_min_distance_A"] = float(np.percentile(vals, 90))

        long_rows.append({
            "complex_id": run_dir.name,
            "molecule_key": molecule_key,
            "residue_label": definition["label"],
            "residue_number": r,
            "residue_type": definition["restype"],
            "residue_group": definition["group"],
            "residue_weight": definition["weight"],
            "premd_plip_present": bool(types),
            "premd_plip_interaction_types": "|".join(types),
            "md_contact_cutoff_A": cutoff_A,
            "md_contact_persistence_fraction": persistence[r],
            "md_contact_persistence_percent": 100.0 * persistence[r],
            "md_median_min_distance_A": float(np.median(vals)),
            "md_p90_min_distance_A": float(np.percentile(vals, 90)),
            "md_min_distance_A": float(np.min(vals)),
            "md_max_distance_A": float(np.max(vals)),
            "production_frames": production_frames,
        })

    return summary, long_rows


def analyze_generic_stability(run_dir: Path) -> tuple[dict, dict, pd.DataFrame] | None:
    status_file = run_dir / "status.json"
    metrics_file = run_dir / "pose_metrics.csv"

    if not status_file.exists() or not metrics_file.exists():
        return None

    try:
        status = json.loads(status_file.read_text())
    except Exception:
        return None

    if status.get("success") is not True or status.get("stage") != "complete":
        return None

    try:
        df = pd.read_csv(metrics_file)
    except Exception:
        return None

    required = {
        "step", "time_ps", "ligand_heavy_rmsd_A", "minimum_protein_ligand_A",
        "ligand_atoms_contact_3.5A", "ligand_atoms_contact_4.5A",
    }
    if not required.issubset(df.columns):
        return None

    for col in required:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=list(required))

    if df.empty:
        return None

    equil_ps = infer_equilibration_ps(status, df)
    prod = df[df["time_ps"] > equil_ps + 1e-8].copy()
    if len(prod) < 5:
        prod = df[df["time_ps"] >= equil_ps].copy()
    if len(prod) < 5:
        return None

    heavy = int(status.get("ligand_heavy_atoms", 0) or 0)
    if heavy <= 0:
        heavy = int(max(prod["ligand_atoms_contact_4.5A"].max(), 1))

    rmsd = prod["ligand_heavy_rmsd_A"].to_numpy(dtype=float)
    t_ns = (prod["time_ps"].to_numpy(dtype=float) - float(prod["time_ps"].iloc[0])) / 1000.0
    min_dist = prod["minimum_protein_ligand_A"].to_numpy(dtype=float)
    c35 = prod["ligand_atoms_contact_3.5A"].to_numpy(dtype=float)
    c45 = prod["ligand_atoms_contact_4.5A"].to_numpy(dtype=float)

    n_early = max(3, int(math.ceil(0.10 * len(prod))))
    baseline_c35 = float(np.median(c35[:n_early]))
    c35_threshold = max(1.0, 0.70 * baseline_c35)
    final_window = max(3, int(math.ceil(0.10 * len(prod))))

    row = {
        "complex_id": run_dir.name,
        "run_dir": str(run_dir.resolve()),
        "pdb": status.get("pdb", ""),
        "sdf": status.get("sdf", ""),
        "ligand_heavy_atoms": heavy,
        "equilibration_ps_inferred": equil_ps,
        "production_frames_pose_metrics": len(prod),

        "rmsd_median_A": float(np.median(rmsd)),
        "rmsd_mean_A": float(np.mean(rmsd)),
        "rmsd_std_A": float(np.std(rmsd, ddof=0)),
        "rmsd_p90_A": float(np.percentile(rmsd, 90)),
        "rmsd_max_A": float(np.max(rmsd)),
        "rmsd_final10pct_median_A": float(np.median(rmsd[-final_window:])),
        "rmsd_slope_A_per_ns": linear_slope(t_ns, rmsd),
        "fraction_rmsd_le_2.5A": float(np.mean(rmsd <= 2.5)),
        "fraction_rmsd_le_3.0A": float(np.mean(rmsd <= 3.0)),

        "min_distance_median_A": float(np.median(min_dist)),
        "min_distance_min_A": float(np.min(min_dist)),
        "contact35_median_fraction": float(np.median(c35) / heavy),
        "contact45_median_fraction": float(np.median(c45) / heavy),
        "contact35_retention_fraction": float(np.mean(c35 >= c35_threshold)),
    }

    flags = []
    if row["min_distance_min_A"] < 1.20:
        flags.append("possible_clash_<1.2A")
    if row["contact45_median_fraction"] < 0.20:
        flags.append("weak_global_contact_4.5A")
    if row["rmsd_p90_A"] > 6.0:
        flags.append("large_RMSD_p90")
    if row["rmsd_final10pct_median_A"] > 6.0:
        flags.append("large_final_RMSD")

    row["generic_review_flag"] = ";".join(flags)
    row["generic_eligible"] = len(flags) == 0
    return row, status, df



def write_clean_protein_ligand_pdb(
    source_pdb: Path,
    output_pdb: Path,
    ligand_resname: str = "LIG",
) -> tuple[int, int]:
    """
    Write a clean final protein-ligand complex PDB from the MD final PDB.

    Keeps:
      - all ATOM records (protein)
      - HETATM records belonging to ligand_resname

    Removes water, ions, and unrelated HETATM species.
    """
    if not source_pdb.is_file():
        raise FileNotFoundError(f"Final MD PDB not found: {source_pdb}")

    protein_atoms = 0
    ligand_atoms = 0
    out_lines = []

    with source_pdb.open("r", encoding="utf-8", errors="replace") as handle:
        for raw in handle:
            line = raw.rstrip("\n")

            if line.startswith("ATOM  "):
                out_lines.append(line)
                protein_atoms += 1
                continue

            if line.startswith("HETATM"):
                resname = line[17:20].strip().upper()
                if resname == ligand_resname.upper():
                    out_lines.append(line)
                    ligand_atoms += 1

    if protein_atoms == 0:
        raise ValueError(f"No protein ATOM records found in {source_pdb}")
    if ligand_atoms == 0:
        raise ValueError(
            f"No ligand HETATM records with residue name {ligand_resname!r} "
            f"found in {source_pdb}"
        )

    out_lines.extend(["TER", "END"])
    output_pdb.write_text("\n".join(out_lines) + "\n", encoding="utf-8")
    return protein_atoms, ligand_atoms


def short_complex_name(complex_id: str) -> str:
    name = re.sub(r"^rank\d+__", "", str(complex_id), flags=re.IGNORECASE)
    name = re.sub(r"_complex$", "", name, flags=re.IGNORECASE)
    return name

def copy_top_files(
    top: pd.DataFrame,
    selection_root: Path,
    ligand_resname: str = "LIG",
) -> None:
    """
    Copy selected outputs and generate five final protein-ligand-only PDBs.
    """
    selection_root.mkdir(parents=True, exist_ok=True)
    pdb_root = selection_root / "TOP5_PDB"
    pdb_root.mkdir(parents=True, exist_ok=True)

    manifest_rows = []

    for _, row in top.iterrows():
        rank = int(row["final_rank"])
        cid = str(row["complex_id"])
        src_run = Path(row["run_dir"])
        dest = selection_root / f"rank{rank:02d}_{cid}"
        dest.mkdir(parents=True, exist_ok=True)

        pdb = Path(str(row.get("pdb", "")))
        sdf = Path(str(row.get("sdf", "")))

        if pdb.is_file():
            shutil.copy2(pdb, dest / "original_docked_complex.pdb")
        if sdf.is_file():
            shutil.copy2(sdf, dest / "ligand.sdf")

        for name in [
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
            source = src_run / name
            if source.is_file():
                shutil.copy2(source, dest / name)

        final_md_pdb = src_run / "final.pdb"
        clean_local = dest / "final_protein_ligand_complex.pdb"

        short_name = short_complex_name(cid)
        clean_top5 = pdb_root / f"rank{rank:02d}_{short_name}_final_complex.pdb"

        protein_atoms = 0
        ligand_atoms = 0

        if final_md_pdb.is_file():
            protein_atoms, ligand_atoms = write_clean_protein_ligand_pdb(
                final_md_pdb,
                clean_local,
                ligand_resname=ligand_resname,
            )
            shutil.copy2(clean_local, clean_top5)

        manifest_rows.append({
            "rank": rank,
            "complex_id": cid,
            "final_score": row["final_CD97_stability_persistence_score"],
            "general_stability_score": row["general_stability_score"],
            "weighted_initial_residue_retention_fraction":
                row["weighted_initial_residue_retention_fraction"],
            "original_route_retention_fraction":
                row["original_route_retention_fraction"],
            "md_any_valid_path_fraction":
                row["md_any_valid_path_fraction"],
            "final_complex_pdb":
                str(clean_top5.resolve()) if clean_top5.exists() else "",
            "protein_atoms": protein_atoms,
            "ligand_atoms": ligand_atoms,
        })

    manifest = selection_root / "TOP5_PDB_manifest.csv"
    pd.DataFrame(manifest_rows).sort_values("rank").to_csv(manifest, index=False)

    print()
    print("FINAL TOP-5 PROTEIN-LIGAND PDB COMPLEXES")
    print("=" * 90)
    for item in sorted(manifest_rows, key=lambda x: x["rank"]):
        print(
            f"Rank {item['rank']}: {item['final_complex_pdb']} "
            f"(protein atoms={item['protein_atoms']}, "
            f"ligand atoms={item['ligand_atoms']})"
        )

    print()
    print(f"PDB-only folder: {pdb_root}")
    print(f"PDB manifest   : {manifest}")

def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Rank CD97 OpenMM trajectories by generic stability and persistence "
            "of the pre-MD PLIP interactions at W685/F760, side residues, and bottom residues."
        )
    )
    parser.add_argument("--output", required=True, help="CD97_openmm_1ns output folder")
    parser.add_argument("--top", type=int, default=5)
    parser.add_argument("--plip-interactions", default=None,
                        help="Optional all_target_interactions.csv. Auto-detected when omitted.")
    parser.add_argument("--contact-cutoff", type=float, default=4.0,
                        help="Heavy-atom residue/LIG contact cutoff in Angstrom. Default 4.0.")
    parser.add_argument("--ligand-resname", default="LIG")
    parser.add_argument("--selection-dir", default=None)
    parser.add_argument("--limit", type=int, default=None,
                        help="Analyze only the first N completed runs for testing.")
    args = parser.parse_args()

    root = Path(args.output).expanduser().resolve()
    runs = root / "runs"
    if not runs.is_dir():
        raise SystemExit(f"Runs directory not found: {runs}")

    plip_path = locate_plip_interactions(root, args.plip_interactions)
    if plip_path is None:
        raise SystemExit(
            "Could not find all_target_interactions.csv automatically.\n"
            "Supply it explicitly with:\n"
            "  --plip-interactions /path/to/PLIP_GLOBAL_ANALYSIS/all_target_interactions.csv"
        )

    plip_map = load_plip_interactions(plip_path)

    run_dirs = [p for p in sorted(runs.iterdir()) if p.is_dir()]
    if args.limit is not None:
        run_dirs = run_dirs[:args.limit]

    rows = []
    long_rows = []
    failures = []

    print(f"Pre-MD PLIP interactions: {plip_path}")
    print(f"MD contact cutoff       : {args.contact_cutoff:.2f} A")
    print()

    for i, run_dir in enumerate(run_dirs, start=1):
        print(f"[{i}/{len(run_dirs)}] {run_dir.name}", flush=True)
        try:
            generic = analyze_generic_stability(run_dir)
            if generic is None:
                failures.append({
                    "complex_id": run_dir.name,
                    "error": "not a completed/readable run",
                })
                continue

            row, status, metrics = generic
            persistence_summary, residue_rows = analyze_residue_persistence(
                run_dir=run_dir,
                status=status,
                metrics=metrics,
                plip_by_molecule=plip_map,
                cutoff_A=args.contact_cutoff,
                ligand_resname=args.ligand_resname,
            )
            row.update(persistence_summary)

            # A selected TOP_DIVERSE ligand should have had a valid PLIP route.
            cd97_flags = []
            if row["premd_target_residue_count"] == 0:
                cd97_flags.append("no_preMD_PLIP_mapping")
            if not (row["premd_path_A_lid"] or row["premd_path_B_bypass"]):
                cd97_flags.append("preMD_PLIP_route_not_reconstructed")
            if row["dcd_metrics_frame_mismatch"]:
                cd97_flags.append("DCD_pose_metrics_frame_count_mismatch")

            row["cd97_review_flag"] = ";".join(cd97_flags)
            row["eligible"] = bool(row["generic_eligible"]) and not cd97_flags

            rows.append(row)
            long_rows.extend(residue_rows)

        except Exception as exc:
            failures.append({
                "complex_id": run_dir.name,
                "error": str(exc),
            })

    if not rows:
        raise SystemExit("No runs were successfully analyzed.")

    table = pd.DataFrame(rows)

    # ----- Generic pose-stability score: 0-100 -----
    table["q_rmsd_p90"] = percentile_quality(table["rmsd_p90_A"], False)
    table["q_rmsd_median"] = percentile_quality(table["rmsd_median_A"], False)
    table["q_rmsd_std"] = percentile_quality(table["rmsd_std_A"], False)
    table["abs_rmsd_slope"] = table["rmsd_slope_A_per_ns"].abs()
    table["q_rmsd_slope"] = percentile_quality(table["abs_rmsd_slope"], False)
    table["q_rmsd_under3"] = percentile_quality(table["fraction_rmsd_le_3.0A"], True)
    table["q_global_contact35"] = percentile_quality(table["contact35_median_fraction"], True)
    table["q_global_contact45"] = percentile_quality(table["contact45_median_fraction"], True)
    table["q_global_contact_retention"] = percentile_quality(
        table["contact35_retention_fraction"], True
    )

    table["general_stability_score"] = (
        0.25 * table["q_rmsd_p90"]
        + 0.15 * table["q_rmsd_median"]
        + 0.10 * table["q_rmsd_std"]
        + 0.10 * table["q_rmsd_slope"]
        + 0.10 * table["q_rmsd_under3"]
        + 0.15 * table["q_global_contact35"]
        + 0.10 * table["q_global_contact45"]
        + 0.05 * table["q_global_contact_retention"]
    )

    # ----- CD97-specific persistence score: absolute occupancy, 0-100 -----
    table["cd97_weighted_initial_residue_persistence_score"] = (
        100.0 * table["weighted_initial_residue_retention_fraction"]
    )
    table["cd97_original_route_persistence_score"] = (
        100.0 * table["original_route_retention_fraction"]
    )
    table["cd97_any_valid_route_persistence_score"] = (
        100.0 * table["md_any_valid_path_fraction"]
    )

    # 50% generic + 50% target-specific.
    table["final_CD97_stability_persistence_score"] = (
        0.50 * table["general_stability_score"]
        + 0.25 * table["cd97_weighted_initial_residue_persistence_score"]
        + 0.15 * table["cd97_original_route_persistence_score"]
        + 0.10 * table["cd97_any_valid_route_persistence_score"]
    )

    eligible = table[table["eligible"]].copy()
    eligible = eligible.sort_values(
        by=[
            "final_CD97_stability_persistence_score",
            "weighted_initial_residue_retention_fraction",
            "original_route_retention_fraction",
            "rmsd_p90_A",
        ],
        ascending=[False, False, False, True],
    ).reset_index(drop=True)

    eligible["final_rank"] = np.arange(1, len(eligible) + 1)

    table = table.merge(
        eligible[["complex_id", "final_rank"]],
        on="complex_id",
        how="left",
    )
    table = table.sort_values(
        by=["eligible", "final_CD97_stability_persistence_score"],
        ascending=[False, False],
    )

    top = eligible.head(args.top).copy()

    ranking_csv = root / "CD97_persistence_ranking_all_200.csv"
    long_csv = root / "CD97_residue_persistence_long.csv"
    top_csv = root / f"TOP{args.top}_CD97_stable_persistent.csv"
    failure_csv = root / "CD97_persistence_analysis_failures.csv"

    table.to_csv(ranking_csv, index=False)
    pd.DataFrame(long_rows).to_csv(long_csv, index=False)
    top.to_csv(top_csv, index=False)
    pd.DataFrame(failures).to_csv(failure_csv, index=False)

    print()
    print("CD97 PERSISTENCE ANALYSIS COMPLETE")
    print("=" * 90)
    print(f"Runs successfully analyzed : {len(table)}")
    print(f"Eligible for ranking       : {len(eligible)}")
    print(f"Analysis/review failures   : {len(failures) + (len(table)-len(eligible))}")
    print(f"Full ranking               : {ranking_csv}")
    print(f"Per-residue persistence    : {long_csv}")
    print(f"Top {args.top}                     : {top_csv}")
    print()
    print(f"TOP {args.top} — STABILITY + PRE-MD CD97 INTERACTION PERSISTENCE")
    print("=" * 130)

    show_cols = [
        "final_rank",
        "complex_id",
        "final_CD97_stability_persistence_score",
        "general_stability_score",
        "weighted_initial_residue_retention_fraction",
        "original_route_retention_fraction",
        "md_any_valid_path_fraction",
        "rmsd_median_A",
        "rmsd_p90_A",
        "premd_target_residues",
        "premd_original_route",
    ]

    with pd.option_context("display.max_colwidth", 60, "display.width", 260):
        print(top[show_cols].to_string(
            index=False,
            float_format=lambda x: f"{x:.3f}",
        ))

    print()
    print("Per-residue persistence for the selected top complexes:")
    for _, row in top.iterrows():
        print(f"\nRank {int(row['final_rank'])}: {row['complex_id']}")
        for resnum, definition in TARGET_RESIDUES.items():
            label = definition["label"]
            pre = row[f"{label}_premd_types"]
            p = 100.0 * float(row[f"{label}_persistence_fraction"])
            marker = "PRE-MD PLIP" if str(pre).strip() else "not pre-MD"
            types = f" [{pre}]" if str(pre).strip() else ""
            print(f"  {label:5s} {marker:12s}{types:32s} -> MD contact persistence {p:6.1f}%")

    if args.selection_dir:
        selection_dir = Path(args.selection_dir).expanduser().resolve()
        copy_top_files(top, selection_dir, ligand_resname=args.ligand_resname)
        shutil.copy2(top_csv, selection_dir / top_csv.name)
        print(f"\nTop files copied to: {selection_dir}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
