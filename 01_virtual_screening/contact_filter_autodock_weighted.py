#!/usr/bin/env python3
"""
Weighted contact-based ranking for AutoDock-GPU poses.

Residue groups:
  - Lid: highest weight
  - Hydrophobic-side residues: equal lower weight
  - Polar-bottom residues: equal lower weight

Expected input files:
    <basename>-best.pdbqt
    <basename>.dlg
    <basename>.xml

Ranking priority:
    1. Weighted group coverage
    2. Number of groups contacted
    3. Total unique target residues contacted
    4. Number of ligand heavy atoms making contacts
    5. Number of receptor-ligand heavy-atom contact pairs
    6. Mean minimum distance to contacted residues
    7. AutoDock score, more negative is better

Output directory:
    <input_folder>/<input_folder_name>_top<TOP_N>_contact_analysis

Only Python's standard library is required.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import re
import shutil
import sys
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


FLOAT_RE = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[Ee][+-]?\d+)?"

XML_SCORE_RE = re.compile(
    rf"<\s*free_NRG_binding\s*>\s*({FLOAT_RE})\s*<\s*/\s*free_NRG_binding\s*>",
    re.IGNORECASE,
)

RESIDUE_RE = re.compile(
    r"^(?:(?P<chain>[^:]):)?(?:(?P<resname>[A-Za-z]{1,3}))?"
    r"(?P<resid>-?\d+)(?P<icode>[A-Za-z]?)$"
)

ONE_TO_THREE = {
    "A": "ALA", "R": "ARG", "N": "ASN", "D": "ASP", "C": "CYS",
    "E": "GLU", "Q": "GLN", "G": "GLY", "H": "HIS", "I": "ILE",
    "L": "LEU", "K": "LYS", "M": "MET", "F": "PHE", "P": "PRO",
    "S": "SER", "T": "THR", "W": "TRP", "Y": "TYR", "V": "VAL",
}

_WORKER_RESIDUES = None
_WORKER_GROUPS = None
_WORKER_WEIGHTS = None
_WORKER_CUTOFF2 = None


@dataclass(frozen=True)
class ResidueSpec:
    token: str
    chain: Optional[str]
    resname: Optional[str]
    resid: int
    icode: str


@dataclass
class RankedPose:
    basename: str
    pose_file: str
    weighted_contact_score: float
    groups_contacted: int
    total_unique_residues_contacted: int
    lid_contacts: int
    hydrophobic_contacts: int
    polar_contacts: int
    contacting_ligand_heavy_atoms: int
    ligand_heavy_atoms: int
    heavy_atom_pair_contacts: int
    mean_min_contact_distance: float
    autodock_score: float
    contacted_residues: str
    all_residue_min_distances: str


def parse_residue(token: str) -> ResidueSpec:
    token = token.strip()
    match = RESIDUE_RE.fullmatch(token)

    if not match:
        raise ValueError(
            f"Invalid residue '{token}'. Use F597, PHE597, A:F597, or A:597."
        )

    resname = match.group("resname")
    if resname:
        resname = resname.upper()
        if len(resname) == 1:
            if resname not in ONE_TO_THREE:
                raise ValueError(f"Unknown one-letter residue code in '{token}'.")
            resname = ONE_TO_THREE[resname]

    return ResidueSpec(
        token=token,
        chain=match.group("chain"),
        resname=resname,
        resid=int(match.group("resid")),
        icode=match.group("icode").upper(),
    )


def atom_type(line: str) -> str:
    parts = line.split()

    if parts:
        value = parts[-1].upper()
        if value in {"H", "HD", "HS"}:
            return "H"
        if value.startswith("CL"):
            return "CL"
        if value.startswith("BR"):
            return "BR"
        if value and value[0].isalpha():
            return value

    element = line[76:78].strip().upper() if len(line) >= 78 else ""
    if element:
        return element

    name = line[12:16].strip().upper() if len(line) >= 16 else ""
    name = re.sub(r"^\d+", "", name)
    return name[:1] if name else ""


def parse_xyz(line: str) -> Optional[tuple[float, float, float]]:
    try:
        return float(line[30:38]), float(line[38:46]), float(line[46:54])
    except (ValueError, IndexError):
        parts = line.split()
        for start in (6, 5):
            try:
                return (
                    float(parts[start]),
                    float(parts[start + 1]),
                    float(parts[start + 2]),
                )
            except (ValueError, IndexError):
                continue
    return None


def parse_receptor_identity(
    line: str,
) -> Optional[tuple[str, str, int, str]]:
    try:
        return (
            line[21:22].strip(),
            line[17:20].strip().upper(),
            int(line[22:26].strip()),
            line[26:27].strip().upper(),
        )
    except (ValueError, IndexError):
        parts = line.split()
        try:
            resname = parts[3].upper()

            if len(parts[4]) == 1 and not re.fullmatch(r"-?\d+", parts[4]):
                chain = parts[4]
                resid = int(parts[5])
            else:
                chain = ""
                resid = int(parts[4])

            return chain, resname, resid, ""
        except (ValueError, IndexError):
            return None


def matches(
    spec: ResidueSpec,
    chain: str,
    resname: str,
    resid: int,
    icode: str,
) -> bool:
    if spec.chain is not None and chain != spec.chain:
        return False
    if spec.resname is not None and resname != spec.resname:
        return False
    if spec.resid != resid:
        return False
    if spec.icode and spec.icode != icode:
        return False
    return True


def read_target_residues(
    receptor: Path,
    specs_by_group: dict[str, list[ResidueSpec]],
) -> tuple[
    dict[str, list[tuple[float, float, float]]],
    dict[str, list[str]],
    list[str],
]:
    all_specs = [spec for specs in specs_by_group.values() for spec in specs]

    atoms_by_identity: dict[
        tuple[str, str, int, str],
        list[tuple[float, float, float]],
    ] = {}

    with receptor.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if not (line.startswith("ATOM") or line.startswith("HETATM")):
                continue

            if atom_type(line) == "H":
                continue

            identity = parse_receptor_identity(line)
            xyz = parse_xyz(line)

            if identity is None or xyz is None:
                continue

            chain, resname, resid, icode = identity

            if any(
                matches(spec, chain, resname, resid, icode)
                for spec in all_specs
            ):
                atoms_by_identity.setdefault(identity, []).append(xyz)

    residue_atoms = {}
    group_labels = {group: [] for group in specs_by_group}
    missing = []

    for group, specs in specs_by_group.items():
        for spec in specs:
            found = []

            for identity, atoms in atoms_by_identity.items():
                chain, resname, resid, icode = identity

                if matches(spec, chain, resname, resid, icode):
                    found.append((identity, atoms))

            if not found:
                missing.append(f"{group}:{spec.token}")
                continue

            for identity, atoms in sorted(found):
                chain, resname, resid, icode = identity
                label = f"{chain or '-'}:{resname}{resid}{icode}"

                residue_atoms[label] = atoms

                if label not in group_labels[group]:
                    group_labels[group].append(label)

    return residue_atoms, group_labels, missing


def read_ligand_atoms(
    path: Path,
) -> list[tuple[float, float, float]]:
    atoms = []

    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if not (line.startswith("ATOM") or line.startswith("HETATM")):
                continue

            if atom_type(line) == "H":
                continue

            xyz = parse_xyz(line)
            if xyz is not None:
                atoms.append(xyz)

    return atoms


def read_autodock_scores(input_dir: Path) -> dict[str, float]:
    scores = {}
    selected_scores_csv = input_dir / "selected_scores.csv"

    if selected_scores_csv.is_file():
        with selected_scores_csv.open(
            "r",
            newline="",
            encoding="utf-8",
        ) as handle:
            for row in csv.DictReader(handle):
                basename = (row.get("basename") or "").strip()
                value = (row.get("autodock_score_kcal_mol") or "").strip()

                if not basename or not value:
                    continue

                try:
                    score = float(value)
                except ValueError:
                    continue

                if math.isfinite(score):
                    scores[basename] = score

    for xml_path in input_dir.glob("*.xml"):
        basename = xml_path.stem

        if basename in scores:
            continue

        try:
            text = xml_path.read_text(
                encoding="utf-8",
                errors="replace",
            )
        except OSError:
            continue

        values = []

        for match in XML_SCORE_RE.finditer(text):
            try:
                score = float(match.group(1))
            except ValueError:
                continue

            if math.isfinite(score):
                values.append(score)

        if values:
            scores[basename] = min(values)

    return scores


def init_worker(
    residue_atoms,
    group_labels,
    weights,
    cutoff2,
):
    global _WORKER_RESIDUES
    global _WORKER_GROUPS
    global _WORKER_WEIGHTS
    global _WORKER_CUTOFF2

    _WORKER_RESIDUES = residue_atoms
    _WORKER_GROUPS = group_labels
    _WORKER_WEIGHTS = weights
    _WORKER_CUTOFF2 = cutoff2


def score_pose(task):
    pose_path_text, autodock_score = task
    pose_path = Path(pose_path_text)
    basename = pose_path.name.removesuffix("-best.pdbqt")

    try:
        ligand_atoms = read_ligand_atoms(pose_path)
    except OSError as exc:
        return None, f"{pose_path.name}: {exc}"

    if not ligand_atoms:
        return None, f"{pose_path.name}: no readable heavy atoms"

    contacted_residues = set()
    contacting_ligand_atoms = set()
    heavy_atom_pair_contacts = 0
    minimum_distances = {}

    for residue_label, receptor_atoms in _WORKER_RESIDUES.items():
        minimum_distance_squared = math.inf
        residue_contacted = False

        for ligand_index, (lx, ly, lz) in enumerate(ligand_atoms):
            ligand_atom_contacts_residue = False

            for rx, ry, rz in receptor_atoms:
                dx = lx - rx
                dy = ly - ry
                dz = lz - rz
                distance_squared = dx * dx + dy * dy + dz * dz

                if distance_squared < minimum_distance_squared:
                    minimum_distance_squared = distance_squared

                if distance_squared <= _WORKER_CUTOFF2:
                    heavy_atom_pair_contacts += 1
                    residue_contacted = True
                    ligand_atom_contacts_residue = True

            if ligand_atom_contacts_residue:
                contacting_ligand_atoms.add(ligand_index)

        minimum_distances[residue_label] = (
            math.sqrt(minimum_distance_squared)
            if math.isfinite(minimum_distance_squared)
            else math.inf
        )

        if residue_contacted:
            contacted_residues.add(residue_label)

    group_counts = {}
    weighted_contact_score = 0.0
    groups_contacted = 0

    for group, labels in _WORKER_GROUPS.items():
        count = sum(label in contacted_residues for label in labels)
        group_counts[group] = count

        if count > 0:
            groups_contacted += 1

        if labels:
            coverage = count / len(labels)
            weighted_contact_score += _WORKER_WEIGHTS[group] * coverage

    contacted_distance_values = [
        minimum_distances[label]
        for label in contacted_residues
    ]

    mean_min_contact_distance = (
        sum(contacted_distance_values) / len(contacted_distance_values)
        if contacted_distance_values
        else math.inf
    )

    result = RankedPose(
        basename=basename,
        pose_file=pose_path.name,
        weighted_contact_score=weighted_contact_score,
        groups_contacted=groups_contacted,
        total_unique_residues_contacted=len(contacted_residues),
        lid_contacts=group_counts.get("lid", 0),
        hydrophobic_contacts=group_counts.get("hydrophobic", 0),
        polar_contacts=group_counts.get("polar", 0),
        contacting_ligand_heavy_atoms=len(contacting_ligand_atoms),
        ligand_heavy_atoms=len(ligand_atoms),
        heavy_atom_pair_contacts=heavy_atom_pair_contacts,
        mean_min_contact_distance=mean_min_contact_distance,
        autodock_score=autodock_score,
        contacted_residues=";".join(sorted(contacted_residues)),
        all_residue_min_distances=";".join(
            f"{label}={minimum_distances[label]:.3f}"
            for label in sorted(minimum_distances)
        ),
    )

    return result, None


def transfer_file(
    source: Path,
    destination: Path,
    mode: str,
) -> None:
    if mode == "copy":
        shutil.copy2(source, destination)
    elif mode == "hardlink":
        os.link(source, destination)
    elif mode == "symlink":
        destination.symlink_to(
            os.path.relpath(source, destination.parent)
        )
    else:
        raise ValueError(f"Unsupported mode: {mode}")


def finite_text(value: float, digits: int = 4) -> str:
    return f"{value:.{digits}f}" if math.isfinite(value) else ""


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Select AutoDock-GPU poses using weighted CD97 contact groups."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("receptor", type=Path)
    parser.add_argument("input_dir", type=Path)
    parser.add_argument("top_n", type=int)

    parser.add_argument("--lid", nargs="+", required=True)
    parser.add_argument("--hydrophobic", nargs="+", required=True)
    parser.add_argument("--polar", nargs="+", required=True)

    parser.add_argument("--lid-weight", type=float, default=3.0)
    parser.add_argument("--hydrophobic-weight", type=float, default=2.0)
    parser.add_argument("--polar-weight", type=float, default=2.0)

    parser.add_argument("--cutoff", type=float, default=4.0)
    parser.add_argument("--min-total-contacts", type=int, default=1)
    parser.add_argument("--require-lid", action="store_true")

    parser.add_argument(
        "--workers",
        type=int,
        default=max(1, min(8, os.cpu_count() or 1)),
    )

    parser.add_argument(
        "--mode",
        choices=("copy", "hardlink", "symlink"),
        default="copy",
    )

    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")

    return parser


def main() -> int:
    args = build_parser().parse_args()

    receptor = args.receptor.expanduser().resolve()
    input_dir = args.input_dir.expanduser().resolve()

    if not receptor.is_file():
        print(f"ERROR: receptor not found: {receptor}", file=sys.stderr)
        return 2

    if not input_dir.is_dir():
        print(f"ERROR: input folder not found: {input_dir}", file=sys.stderr)
        return 2

    if args.top_n <= 0 or args.cutoff <= 0 or args.workers <= 0:
        print(
            "ERROR: top_n, cutoff, and workers must be positive.",
            file=sys.stderr,
        )
        return 2

    try:
        specs_by_group = {
            "lid": [parse_residue(x) for x in args.lid],
            "hydrophobic": [parse_residue(x) for x in args.hydrophobic],
            "polar": [parse_residue(x) for x in args.polar],
        }
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    weights = {
        "lid": args.lid_weight,
        "hydrophobic": args.hydrophobic_weight,
        "polar": args.polar_weight,
    }

    residue_atoms, group_labels, missing = read_target_residues(
        receptor,
        specs_by_group,
    )

    if not residue_atoms:
        print(
            "ERROR: none of the requested residues were found.",
            file=sys.stderr,
        )
        return 3

    print(f"Receptor          : {receptor}")
    print(f"Input folder      : {input_dir}")
    print(f"Top N             : {args.top_n:,}")
    print(f"Contact cutoff    : {args.cutoff:.2f} Å")
    print(
        f"Weights           : lid={args.lid_weight}, "
        f"hydrophobic={args.hydrophobic_weight}, "
        f"polar={args.polar_weight}"
    )
    print(f"Lid residues      : {' '.join(group_labels['lid'])}")
    print(
        f"Hydrophobic       : "
        f"{' '.join(group_labels['hydrophobic'])}"
    )
    print(f"Polar residues    : {' '.join(group_labels['polar'])}")

    if missing:
        print("WARNING: residues not found:", file=sys.stderr)
        for item in missing:
            print(f"  {item}", file=sys.stderr)

    pose_files = sorted(input_dir.glob("*-best.pdbqt"))

    if not pose_files:
        print(
            "ERROR: no *-best.pdbqt files found.",
            file=sys.stderr,
        )
        return 3

    autodock_scores = read_autodock_scores(input_dir)

    tasks = [
        (
            str(path),
            autodock_scores.get(
                path.name.removesuffix("-best.pdbqt"),
                math.inf,
            ),
        )
        for path in pose_files
    ]

    print(f"Pose files        : {len(pose_files):,}")
    print(f"Workers           : {args.workers}")
    print("Calculating contacts...")

    results = []
    failures = []

    if args.workers == 1:
        init_worker(
            residue_atoms,
            group_labels,
            weights,
            args.cutoff ** 2,
        )

        iterator = map(score_pose, tasks)

        for index, (result, failure) in enumerate(iterator, 1):
            if result:
                results.append(result)
            if failure:
                failures.append(failure)

            if index % 500 == 0:
                print(
                    f"Processed {index:,}/{len(tasks):,}",
                    file=sys.stderr,
                )
    else:
        with ProcessPoolExecutor(
            max_workers=args.workers,
            initializer=init_worker,
            initargs=(
                residue_atoms,
                group_labels,
                weights,
                args.cutoff ** 2,
            ),
        ) as pool:
            for index, (result, failure) in enumerate(
                pool.map(
                    score_pose,
                    tasks,
                    chunksize=25,
                ),
                1,
            ):
                if result:
                    results.append(result)
                if failure:
                    failures.append(failure)

                if index % 500 == 0:
                    print(
                        f"Processed {index:,}/{len(tasks):,}",
                        file=sys.stderr,
                    )

    eligible = [
        result
        for result in results
        if result.total_unique_residues_contacted >= args.min_total_contacts
        and (not args.require_lid or result.lid_contacts > 0)
    ]

    eligible.sort(
        key=lambda result: (
            -result.weighted_contact_score,
            -result.groups_contacted,
            -result.total_unique_residues_contacted,
            -result.contacting_ligand_heavy_atoms,
            -result.heavy_atom_pair_contacts,
            result.mean_min_contact_distance,
            result.autodock_score,
            result.basename,
        )
    )

    selected = eligible[: args.top_n]

    print(f"Readable poses     : {len(results):,}")
    print(f"Eligible poses     : {len(eligible):,}")
    print(f"Selected poses     : {len(selected):,}")

    if not selected:
        print(
            "ERROR: no ligand passed the contact criteria.",
            file=sys.stderr,
        )
        return 4

    print("\nTop 10:")

    for rank, result in enumerate(selected[:10], 1):
        dock_score = finite_text(result.autodock_score, 3)

        print(
            f"{rank:>3} "
            f"weighted={result.weighted_contact_score:>5.2f} "
            f"lid={result.lid_contacts} "
            f"hydro={result.hydrophobic_contacts} "
            f"polar={result.polar_contacts} "
            f"total={result.total_unique_residues_contacted} "
            f"dock={dock_score:>7} "
            f"{result.basename}"
        )

    if args.dry_run:
        print("\nDry run: no output folder created.")
        return 0

    output_dir = (
        input_dir
        / f"{input_dir.name}_top{args.top_n}_contact_analysis"
    )

    if output_dir.exists():
        if not args.overwrite:
            print(
                f"ERROR: output folder exists:\n"
                f"  {output_dir}\n"
                f"Use --overwrite to replace it.",
                file=sys.stderr,
            )
            return 5

        shutil.rmtree(output_dir)

    output_dir.mkdir()

    transferred = 0
    missing_associated_files = []

    for result in selected:
        expected_files = [
            input_dir / f"{result.basename}-best.pdbqt",
            input_dir / f"{result.basename}.dlg",
            input_dir / f"{result.basename}.xml",
        ]

        for source in expected_files:
            if source.is_file():
                transfer_file(
                    source,
                    output_dir / source.name,
                    args.mode,
                )
                transferred += 1
            else:
                missing_associated_files.append(
                    (result.basename, source.name)
                )

    ranking_csv = output_dir / "contact_ranking.csv"

    with ranking_csv.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.writer(handle)

        writer.writerow([
            "rank",
            "basename",
            "weighted_contact_score",
            "groups_contacted",
            "total_unique_target_residues_contacted",
            "lid_contacts",
            "hydrophobic_contacts",
            "polar_contacts",
            "contacting_ligand_heavy_atoms",
            "ligand_heavy_atoms",
            "heavy_atom_pair_contacts",
            "mean_min_contact_distance_A",
            "autodock_score_kcal_mol",
            "contacted_residues",
            "all_target_residue_min_distances_A",
        ])

        for rank, result in enumerate(selected, 1):
            writer.writerow([
                rank,
                result.basename,
                f"{result.weighted_contact_score:.6f}",
                result.groups_contacted,
                result.total_unique_residues_contacted,
                result.lid_contacts,
                result.hydrophobic_contacts,
                result.polar_contacts,
                result.contacting_ligand_heavy_atoms,
                result.ligand_heavy_atoms,
                result.heavy_atom_pair_contacts,
                finite_text(result.mean_min_contact_distance, 4),
                finite_text(result.autodock_score, 6),
                result.contacted_residues,
                result.all_residue_min_distances,
            ])

    with (
        output_dir / "contact_scoring_parameters.txt"
    ).open("w", encoding="utf-8") as handle:
        handle.write(f"receptor={receptor}\n")
        handle.write(f"input_dir={input_dir}\n")
        handle.write(f"top_n={args.top_n}\n")
        handle.write(f"cutoff_A={args.cutoff}\n")
        handle.write(f"lid={' '.join(args.lid)}\n")
        handle.write(
            f"hydrophobic={' '.join(args.hydrophobic)}\n"
        )
        handle.write(f"polar={' '.join(args.polar)}\n")
        handle.write(f"lid_weight={args.lid_weight}\n")
        handle.write(
            f"hydrophobic_weight={args.hydrophobic_weight}\n"
        )
        handle.write(f"polar_weight={args.polar_weight}\n")
        handle.write(f"require_lid={args.require_lid}\n")
        handle.write(
            f"min_total_contacts={args.min_total_contacts}\n"
        )

    with (
        output_dir / "contact_scoring_failures.csv"
    ).open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(["failure"])
        writer.writerows([[failure] for failure in failures])

    with (
        output_dir / "missing_associated_files.csv"
    ).open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(["basename", "missing_file"])
        writer.writerows(missing_associated_files)

    print(f"\nOutput folder     : {output_dir}")
    print(f"Files transferred : {transferred:,}")
    print(f"Ranking CSV       : {ranking_csv}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
