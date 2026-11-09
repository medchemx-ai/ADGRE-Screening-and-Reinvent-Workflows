#!/usr/bin/env python3
"""
CD97 PLIP target-residue scoring and diverse top-hit selection.

This script analyzes prepared receptor-ligand complexes before any
subsequent GNINA rescoring. Its input table and pose files must exist:

TOP100_PLIP_INPUT/
├── complexes/
├── selected_sdf/
├── ligand_pdb/
└── selected_top_unique.csv

What it does
------------
1. Runs PLIP in Python-module mode on every receptor-ligand complex.
2. Extracts all PLIP non-covalent interactions involving specified CD97
   pocket residues.
3. Scores interactions using a transparent residue-group and
   interaction-type weighting scheme.
4. Applies a pocket-validity rule:
      - at least one lid residue is contacted, OR
      - a deep-pocket bypass is present:
        >=2 hydrophobic-side residues and >=1 polar-bottom residue.
5. Ranks candidates primarily by the CD97 PLIP interaction score.
6. Selects up to 20 diverse molecules using:
      - unique Bemis-Murcko scaffold, and
      - Morgan/ECFP4 Tanimoto similarity <= 0.60 to every already
        selected molecule.
7. Copies the ORIGINAL pose-preserved complex/SDF/PDB files. It does
   not redock, minimize, align, rotate, translate, or regenerate poses.

Important similarity convention
-------------------------------
For structural diversity, 0.60 is used as the MAXIMUM allowed pairwise
Tanimoto similarity. A requirement of similarity >0.60 would select
similar analogues, not diverse scaffolds.

Default CD97 target residues
----------------------------
Hydrophobic lid:
    W685, F760

Hydrophobic pocket sides:
    F597, F623, W753, F771, L555, L757

Polar pocket bottom:
    N775, Q778, S558, T752, C622

Default residue-group weights:
    lid    = 4.0
    side   = 2.0
    bottom = 2.0

Default PLIP interaction-type multipliers:
    hydrophobic  = 1.00
    hydrogen bond= 1.50
    pi stacking  = 1.50
    pi cation    = 1.50
    salt bridge  = 1.50
    halogen bond = 1.00
    water bridge = 0.75
    metal complex= 1.00

Each residue/interaction-type combination is counted once, so multiple
atom-atom contacts of the same type to the same residue do not
artificially dominate the score.

Example
-------
python3 cd97_plip_score_select_diverse.py \
    /path/to/TOP100_PLIP_INPUT \
    --top 20 \
    --max-similarity 0.60 \
    --dry-run

python3 cd97_plip_score_select_diverse.py \
    /path/to/TOP100_PLIP_INPUT \
    --top 20 \
    --max-similarity 0.60
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import os
import re
import shutil
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable


# ---------------------------------------------------------------------
# CD97 TARGET DEFINITION
# ---------------------------------------------------------------------

TARGET_RESIDUES: dict[int, dict[str, str]] = {
    685: {"label": "W685", "expected_restype": "TRP", "group": "lid"},
    760: {"label": "F760", "expected_restype": "PHE", "group": "lid"},

    597: {"label": "F597", "expected_restype": "PHE", "group": "side"},
    623: {"label": "F623", "expected_restype": "PHE", "group": "side"},
    753: {"label": "W753", "expected_restype": "TRP", "group": "side"},
    771: {"label": "F771", "expected_restype": "PHE", "group": "side"},
    555: {"label": "L555", "expected_restype": "LEU", "group": "side"},
    757: {"label": "L757", "expected_restype": "LEU", "group": "side"},

    775: {"label": "N775", "expected_restype": "ASN", "group": "bottom"},
    778: {"label": "Q778", "expected_restype": "GLN", "group": "bottom"},
    558: {"label": "S558", "expected_restype": "SER", "group": "bottom"},
    752: {"label": "T752", "expected_restype": "THR", "group": "bottom"},
    622: {"label": "C622", "expected_restype": "CYS", "group": "bottom"},
}

GROUP_WEIGHTS: dict[str, float] = {
    "lid": 4.0,
    "side": 2.0,
    "bottom": 2.0,
}

INTERACTION_WEIGHTS: dict[str, float] = {
    "hydrophobic": 1.00,
    "hbond": 1.50,
    "pistacking": 1.50,
    "pication": 1.50,
    "saltbridge": 1.50,
    "halogen": 1.00,
    "waterbridge": 0.75,
    "metal": 1.00,
}

# PLIP interaction-set attributes. Directional lists are combined.
PLIP_INTERACTION_ATTRIBUTES: dict[str, tuple[str, ...]] = {
    "hydrophobic": ("hydrophobic_contacts",),
    "hbond": ("hbonds_ldon", "hbonds_pdon"),
    "pistacking": ("pistacking",),
    "pication": ("pication_laro", "pication_paro"),
    "saltbridge": ("saltbridge_lneg", "saltbridge_pneg"),
    "halogen": ("halogen_bonds",),
    "waterbridge": ("water_bridges",),
    "metal": ("metal_complexes",),
}


@dataclass
class InteractionRecord:
    molecule_id: str
    selected_rank_input: int | None
    complex_pdb: str
    binding_site_id: str
    interaction_type: str
    residue_label: str
    residue_number: int
    residue_type: str
    residue_chain: str
    residue_group: str
    distance_angstrom: float | None
    residue_weight: float
    interaction_weight: float
    score_contribution: float
    detail: str


@dataclass
class MoleculeAnalysis:
    molecule_id: str
    selected_rank_input: int | None
    ligand_filename: str
    result_name: str
    complex_pdb: Path
    selected_sdf: Path
    ligand_pdb: Path | None
    gnina_cnn_score: float | None
    gnina_cnn_affinity: float | None
    gnina_minimized_affinity: float | None
    status: str
    error: str
    binding_site_id: str
    target_interaction_score: float
    lid_bonus: float
    region_bonus: float
    total_plip_score: float
    target_residue_hits: int
    lid_hits: int
    side_hits: int
    bottom_hits: int
    lid_engaged: bool
    deep_pocket_bypass: bool
    pocket_path_valid: bool
    interaction_type_count: int
    target_interaction_count: int
    hit_labels: str
    interaction_summary: str
    murcko_scaffold_smiles: str = ""
    canonical_smiles: str = ""
    diversity_status: str = ""
    maximum_similarity_to_selected: float | None = None
    most_similar_selected_rank: int | None = None


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run PLIP on CD97 top complexes, score target-residue "
            "interactions, and select diverse high-scoring molecules."
        )
    )

    parser.add_argument(
        "plip_input_folder",
        help=(
            "TOP100_PLIP_INPUT folder containing complexes/, "
            "selected_sdf/, and selected_top_unique.csv."
        ),
    )

    parser.add_argument(
        "--top",
        type=int,
        default=20,
        help="Number of diverse molecules to select. Default: 20.",
    )

    parser.add_argument(
        "--max-similarity",
        type=float,
        default=0.60,
        help=(
            "Maximum allowed pairwise Morgan/ECFP4 Tanimoto similarity "
            "among selected molecules. Default: 0.60."
        ),
    )

    parser.add_argument(
        "--min-target-residues",
        type=int,
        default=3,
        help=(
            "Minimum number of unique listed CD97 residues required for "
            "diverse final selection. Default: 3."
        ),
    )

    parser.add_argument(
        "--receptor-chain",
        default=None,
        help=(
            "Optional receptor chain restriction, e.g. A. By default, "
            "target residue numbers are accepted from any receptor chain."
        ),
    )

    parser.add_argument(
        "--ligand-resname",
        default="LIG",
        help="Ligand residue name used in the complex PDB. Default: LIG.",
    )

    parser.add_argument(
        "--ligand-chain",
        default="Z",
        help="Ligand chain used in the complex PDB. Default: Z.",
    )

    parser.add_argument(
        "--ligand-resnum",
        type=int,
        default=1,
        help="Ligand residue number used in the complex PDB. Default: 1.",
    )

    parser.add_argument(
        "--allow-same-scaffold",
        action="store_true",
        help=(
            "Do not require unique Bemis-Murcko scaffolds. The molecular "
            "similarity threshold is still applied."
        ),
    )

    parser.add_argument(
        "--allow-invalid-pocket-path",
        action="store_true",
        help=(
            "Allow final selection without lid engagement or the defined "
            "deep-pocket bypass. Normally such poses remain in the full "
            "ranking but are excluded from the final diverse set."
        ),
    )

    parser.add_argument(
        "--plip-nohydro",
        action="store_true",
        help=(
            "Tell PLIP not to add hydrogens. Use this only when the input "
            "complexes already have suitable explicit hydrogens."
        ),
    )

    parser.add_argument(
        "--output-name",
        default="PLIP_CD97_SCORE_OUT",
        help=(
            "Output subfolder created inside the input folder. "
            "Default: PLIP_CD97_SCORE_OUT."
        ),
    )

    parser.add_argument(
        "--force",
        action="store_true",
        help="Delete and rebuild an existing output folder.",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Validate files and package imports without running the "
            "100-complex analysis."
        ),
    )

    return parser.parse_args()


def safe_name(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._")
    return cleaned or "molecule"


def float_or_none(value: Any) -> float | None:
    if value is None:
        return None

    text = str(value).strip()

    if not text:
        return None

    try:
        number = float(text)
    except (TypeError, ValueError):
        return None

    if not math.isfinite(number):
        return None

    return number


def int_or_none(value: Any) -> int | None:
    number = float_or_none(value)

    if number is None:
        return None

    return int(number)


def import_dependencies() -> dict[str, Any]:
    try:
        from plip.basic import config as plip_config
        from plip.structure.preparation import PDBComplex
    except Exception as error:
        raise RuntimeError(
            "PLIP Python modules could not be imported. Install PLIP "
            "and matching Open Babel Python bindings in this environment.\n"
            f"Original error: {error}"
        ) from error

    try:
        from rdkit import Chem, DataStructs, rdBase
        from rdkit.Chem import rdFingerprintGenerator
        from rdkit.Chem.Scaffolds import MurckoScaffold
    except Exception as error:
        raise RuntimeError(
            "RDKit could not be imported. Install RDKit in this "
            f"environment.\nOriginal error: {error}"
        ) from error

    return {
        "plip_config": plip_config,
        "PDBComplex": PDBComplex,
        "Chem": Chem,
        "DataStructs": DataStructs,
        "rdBase": rdBase,
        "rdFingerprintGenerator": rdFingerprintGenerator,
        "MurckoScaffold": MurckoScaffold,
    }


def read_selection_table(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)

        if reader.fieldnames is None:
            raise RuntimeError(f"CSV contains no header: {path}")

        required = {
            "selected_rank",
            "ligand_filename",
            "result_name",
            "selected_sdf",
            "complex_pdb",
        }
        missing = required.difference(reader.fieldnames)

        if missing:
            raise RuntimeError(
                "selected_top_unique.csv is missing required columns: "
                + ", ".join(sorted(missing))
            )

        return [dict(row) for row in reader]


def resolve_existing_path(
    raw_value: str,
    fallback_directory: Path,
    rank: int | None,
    suffix: str,
) -> Path:
    if raw_value.strip():
        candidate = Path(raw_value).expanduser()

        if candidate.is_file():
            return candidate.resolve()

    if rank is not None and fallback_directory.is_dir():
        matches = sorted(
            fallback_directory.glob(f"rank{rank:03d}__*{suffix}")
        )

        if len(matches) == 1:
            return matches[0].resolve()

    return Path(raw_value).expanduser().resolve()


def build_input_records(
    input_folder: Path,
    rows: list[dict[str, str]],
) -> list[MoleculeAnalysis]:
    complexes_dir = input_folder / "complexes"
    selected_sdf_dir = input_folder / "selected_sdf"
    ligand_pdb_dir = input_folder / "ligand_pdb"

    records: list[MoleculeAnalysis] = []

    for row in rows:
        selected_rank = int_or_none(row.get("selected_rank"))
        molecule_id = safe_name(
            row.get("result_name", "")
            or row.get("ligand_filename", "")
            or f"rank_{selected_rank}"
        )

        complex_path = resolve_existing_path(
            row.get("complex_pdb", ""),
            complexes_dir,
            selected_rank,
            "_complex.pdb",
        )
        sdf_path = resolve_existing_path(
            row.get("selected_sdf", ""),
            selected_sdf_dir,
            selected_rank,
            ".sdf",
        )

        ligand_pdb_raw = row.get("ligand_pdb", "")
        ligand_pdb: Path | None = None

        if ligand_pdb_raw.strip():
            candidate = Path(ligand_pdb_raw).expanduser()

            if candidate.is_file():
                ligand_pdb = candidate.resolve()
        elif selected_rank is not None and ligand_pdb_dir.is_dir():
            matches = sorted(
                ligand_pdb_dir.glob(
                    f"rank{selected_rank:03d}__*_ligand.pdb"
                )
            )

            if len(matches) == 1:
                ligand_pdb = matches[0].resolve()

        records.append(
            MoleculeAnalysis(
                molecule_id=molecule_id,
                selected_rank_input=selected_rank,
                ligand_filename=row.get("ligand_filename", ""),
                result_name=row.get("result_name", ""),
                complex_pdb=complex_path,
                selected_sdf=sdf_path,
                ligand_pdb=ligand_pdb,
                gnina_cnn_score=float_or_none(
                    row.get("cnn_score")
                    or row.get("primary_score")
                ),
                gnina_cnn_affinity=float_or_none(
                    row.get("cnn_affinity")
                ),
                gnina_minimized_affinity=float_or_none(
                    row.get("minimized_affinity_kcal_mol")
                ),
                status="pending",
                error="",
                binding_site_id="",
                target_interaction_score=0.0,
                lid_bonus=0.0,
                region_bonus=0.0,
                total_plip_score=0.0,
                target_residue_hits=0,
                lid_hits=0,
                side_hits=0,
                bottom_hits=0,
                lid_engaged=False,
                deep_pocket_bypass=False,
                pocket_path_valid=False,
                interaction_type_count=0,
                target_interaction_count=0,
                hit_labels="",
                interaction_summary="",
            )
        )

    return records


def validate_input_records(records: list[MoleculeAnalysis]) -> list[str]:
    errors: list[str] = []

    if not records:
        errors.append("No selected molecules were present in the CSV.")

    for record in records:
        if not record.complex_pdb.is_file():
            errors.append(
                f"Missing complex PDB for {record.molecule_id}: "
                f"{record.complex_pdb}"
            )

        if not record.selected_sdf.is_file():
            errors.append(
                f"Missing selected SDF for {record.molecule_id}: "
                f"{record.selected_sdf}"
            )

    return errors


def contact_distance(contact: Any, interaction_type: str) -> float | None:
    if interaction_type == "waterbridge":
        distances = [
            float_or_none(getattr(contact, "distance_aw", None)),
            float_or_none(getattr(contact, "distance_dw", None)),
        ]
        valid = [value for value in distances if value is not None]
        return max(valid) if valid else None

    for attribute in (
        "distance",
        "distance_ad",
        "distance_ah",
    ):
        value = float_or_none(getattr(contact, attribute, None))

        if value is not None:
            return value

    return None


def contact_detail(contact: Any, interaction_type: str) -> str:
    detail: dict[str, Any] = {}

    attributes_by_type = {
        "hydrophobic": ("distance",),
        "hbond": (
            "distance_ad",
            "distance_ah",
            "angle",
            "protisdon",
            "sidechain",
        ),
        "pistacking": ("distance", "angle", "offset", "type"),
        "pication": (
            "distance",
            "offset",
            "protcharged",
            "type",
        ),
        "saltbridge": ("distance", "protispos"),
        "halogen": (
            "distance",
            "don_angle",
            "acc_angle",
            "sidechain",
        ),
        "waterbridge": (
            "distance_aw",
            "distance_dw",
            "d_angle",
            "w_angle",
            "protisdon",
        ),
        "metal": (
            "distance",
            "metal_type",
            "target_type",
            "coordination_num",
            "geometry",
            "location",
        ),
    }

    for attribute in attributes_by_type.get(interaction_type, ()):
        if hasattr(contact, attribute):
            value = getattr(contact, attribute)

            if isinstance(value, (str, int, float, bool)) or value is None:
                detail[attribute] = value
            else:
                detail[attribute] = str(value)

    return json.dumps(detail, sort_keys=True)


def choose_ligand_interaction_set(
    interaction_sets: dict[str, Any],
    ligand_resname: str,
    ligand_chain: str,
    ligand_resnum: int,
) -> tuple[str, Any]:
    expected = (
        ligand_resname.upper(),
        ligand_chain,
        str(ligand_resnum),
    )

    exact_key = ":".join(expected)

    if exact_key in interaction_sets:
        return exact_key, interaction_sets[exact_key]

    matches: list[tuple[str, Any]] = []

    for binding_site_id, interaction_set in interaction_sets.items():
        parts = binding_site_id.split(":")

        if len(parts) < 3:
            continue

        if (
            parts[0].upper() == expected[0]
            and parts[1] == expected[1]
            and parts[2] == expected[2]
        ):
            matches.append((binding_site_id, interaction_set))

    if len(matches) == 1:
        return matches[0]

    available = ", ".join(sorted(interaction_sets)) or "none"

    raise RuntimeError(
        "Could not uniquely identify ligand binding site "
        f"{exact_key}. Available PLIP binding sites: {available}"
    )


def run_plip_for_record(
    record: MoleculeAnalysis,
    dependencies: dict[str, Any],
    receptor_chain: str | None,
    ligand_resname: str,
    ligand_chain: str,
    ligand_resnum: int,
    plip_nohydro: bool,
) -> list[InteractionRecord]:
    plip_config = dependencies["plip_config"]
    PDBComplex = dependencies["PDBComplex"]

    # Do not write a PLIP-fixed structure beside the source complex.
    plip_config.NOFIXFILE = True
    plip_config.NOHYDRO = bool(plip_nohydro)

    complex_object = PDBComplex()
    complex_object.load_pdb(str(record.complex_pdb))
    complex_object.analyze()

    binding_site_id, interaction_set = choose_ligand_interaction_set(
        interaction_sets=complex_object.interaction_sets,
        ligand_resname=ligand_resname,
        ligand_chain=ligand_chain,
        ligand_resnum=ligand_resnum,
    )

    record.binding_site_id = binding_site_id
    raw_target_interactions: list[InteractionRecord] = []

    for interaction_type, attribute_names in (
        PLIP_INTERACTION_ATTRIBUTES.items()
    ):
        for attribute_name in attribute_names:
            contacts = getattr(interaction_set, attribute_name, [])

            for contact in contacts:
                residue_number = int_or_none(
                    getattr(contact, "resnr", None)
                )

                if residue_number is None:
                    continue

                if residue_number not in TARGET_RESIDUES:
                    continue

                residue_chain = str(
                    getattr(contact, "reschain", "")
                ).strip()

                if (
                    receptor_chain is not None
                    and residue_chain != receptor_chain
                ):
                    continue

                target_definition = TARGET_RESIDUES[residue_number]
                residue_type = str(
                    getattr(contact, "restype", "")
                ).strip().upper()

                residue_group = target_definition["group"]
                residue_weight = GROUP_WEIGHTS[residue_group]
                interaction_weight = INTERACTION_WEIGHTS[
                    interaction_type
                ]

                raw_target_interactions.append(
                    InteractionRecord(
                        molecule_id=record.molecule_id,
                        selected_rank_input=record.selected_rank_input,
                        complex_pdb=str(record.complex_pdb),
                        binding_site_id=binding_site_id,
                        interaction_type=interaction_type,
                        residue_label=target_definition["label"],
                        residue_number=residue_number,
                        residue_type=residue_type,
                        residue_chain=residue_chain,
                        residue_group=residue_group,
                        distance_angstrom=contact_distance(
                            contact,
                            interaction_type,
                        ),
                        residue_weight=residue_weight,
                        interaction_weight=interaction_weight,
                        score_contribution=(
                            residue_weight * interaction_weight
                        ),
                        detail=contact_detail(
                            contact,
                            interaction_type,
                        ),
                    )
                )

    # Retain only the shortest representative for each
    # residue/interaction-type/chain tuple. This prevents multiple atom
    # contacts from inflating the score.
    representative: dict[
        tuple[int, str, str],
        InteractionRecord,
    ] = {}

    for interaction in raw_target_interactions:
        key = (
            interaction.residue_number,
            interaction.residue_chain,
            interaction.interaction_type,
        )
        previous = representative.get(key)

        if previous is None:
            representative[key] = interaction
            continue

        old_distance = (
            previous.distance_angstrom
            if previous.distance_angstrom is not None
            else float("inf")
        )
        new_distance = (
            interaction.distance_angstrom
            if interaction.distance_angstrom is not None
            else float("inf")
        )

        if new_distance < old_distance:
            representative[key] = interaction

    interactions = sorted(
        representative.values(),
        key=lambda item: (
            item.residue_number,
            item.interaction_type,
            item.residue_chain,
        ),
    )

    hit_residues = {
        interaction.residue_number for interaction in interactions
    }
    lid_hit_residues = {
        residue
        for residue in hit_residues
        if TARGET_RESIDUES[residue]["group"] == "lid"
    }
    side_hit_residues = {
        residue
        for residue in hit_residues
        if TARGET_RESIDUES[residue]["group"] == "side"
    }
    bottom_hit_residues = {
        residue
        for residue in hit_residues
        if TARGET_RESIDUES[residue]["group"] == "bottom"
    }

    target_interaction_score = sum(
        interaction.score_contribution
        for interaction in interactions
    )

    if len(lid_hit_residues) == 2:
        lid_bonus = 6.0
    elif len(lid_hit_residues) == 1:
        lid_bonus = 2.0
    else:
        lid_bonus = 0.0

    groups_hit = {
        TARGET_RESIDUES[residue]["group"]
        for residue in hit_residues
    }
    region_bonus = 4.0 if len(groups_hit) == 3 else 0.0

    lid_engaged = len(lid_hit_residues) >= 1
    deep_pocket_bypass = (
        len(side_hit_residues) >= 2
        and len(bottom_hit_residues) >= 1
    )
    pocket_path_valid = lid_engaged or deep_pocket_bypass

    interaction_types = {
        interaction.interaction_type
        for interaction in interactions
    }
    type_counts: dict[str, int] = defaultdict(int)

    for interaction in interactions:
        type_counts[interaction.interaction_type] += 1

    record.status = "completed"
    record.target_interaction_score = target_interaction_score
    record.lid_bonus = lid_bonus
    record.region_bonus = region_bonus
    record.total_plip_score = (
        target_interaction_score + lid_bonus + region_bonus
    )
    record.target_residue_hits = len(hit_residues)
    record.lid_hits = len(lid_hit_residues)
    record.side_hits = len(side_hit_residues)
    record.bottom_hits = len(bottom_hit_residues)
    record.lid_engaged = lid_engaged
    record.deep_pocket_bypass = deep_pocket_bypass
    record.pocket_path_valid = pocket_path_valid
    record.interaction_type_count = len(interaction_types)
    record.target_interaction_count = len(interactions)
    record.hit_labels = ";".join(
        TARGET_RESIDUES[residue]["label"]
        for residue in sorted(hit_residues)
    )
    record.interaction_summary = ";".join(
        f"{itype}:{type_counts[itype]}"
        for itype in sorted(type_counts)
    )

    return interactions


def load_rdkit_molecule(
    sdf_path: Path,
    Chem: Any,
) -> Any:
    supplier = Chem.SDMolSupplier(
        str(sdf_path),
        removeHs=False,
        sanitize=True,
    )

    for molecule in supplier:
        if molecule is not None:
            return molecule

    # Fallback for occasional sanitization issues.
    supplier = Chem.SDMolSupplier(
        str(sdf_path),
        removeHs=False,
        sanitize=False,
    )

    for molecule in supplier:
        if molecule is None:
            continue

        try:
            Chem.SanitizeMol(molecule)
        except Exception:
            pass

        return molecule

    raise RuntimeError(f"RDKit could not read molecule from {sdf_path}")


def add_chemical_descriptors(
    records: Iterable[MoleculeAnalysis],
    dependencies: dict[str, Any],
) -> dict[str, Any]:
    Chem = dependencies["Chem"]
    rdFingerprintGenerator = dependencies["rdFingerprintGenerator"]
    MurckoScaffold = dependencies["MurckoScaffold"]

    generator = rdFingerprintGenerator.GetMorganGenerator(
        radius=2,
        fpSize=2048,
        includeChirality=True,
        useBondTypes=True,
    )

    fingerprints: dict[str, Any] = {}

    for record in records:
        if record.status != "completed":
            continue

        try:
            molecule = load_rdkit_molecule(record.selected_sdf, Chem)
            record.canonical_smiles = Chem.MolToSmiles(
                molecule,
                canonical=True,
                isomericSmiles=True,
            )

            scaffold = MurckoScaffold.GetScaffoldForMol(molecule)

            if scaffold is not None and scaffold.GetNumAtoms() > 0:
                record.murcko_scaffold_smiles = Chem.MolToSmiles(
                    scaffold,
                    canonical=True,
                    isomericSmiles=True,
                )
            else:
                # Avoid treating every acyclic molecule as the same empty
                # Murcko scaffold.
                record.murcko_scaffold_smiles = (
                    "ACYCLIC:" + record.canonical_smiles
                )

            fingerprints[record.molecule_id] = (
                generator.GetFingerprint(molecule)
            )
        except Exception as error:
            record.status = "failed_rdkit"
            record.error = str(error)

    return fingerprints


def ranking_key(record: MoleculeAnalysis) -> tuple[Any, ...]:
    return (
        -record.total_plip_score,
        -record.target_residue_hits,
        -record.lid_hits,
        -record.interaction_type_count,
        (
            record.selected_rank_input
            if record.selected_rank_input is not None
            else 10**9
        ),
        record.molecule_id,
    )


def select_diverse_records(
    ranked_records: list[MoleculeAnalysis],
    fingerprints: dict[str, Any],
    dependencies: dict[str, Any],
    top_n: int,
    max_similarity: float,
    minimum_target_residues: int,
    allow_same_scaffold: bool,
    allow_invalid_pocket_path: bool,
) -> tuple[
    list[MoleculeAnalysis],
    list[dict[str, Any]],
]:
    DataStructs = dependencies["DataStructs"]

    selected: list[MoleculeAnalysis] = []
    selected_fingerprints: list[Any] = []
    selected_scaffolds: set[str] = set()
    rejected: list[dict[str, Any]] = []

    for record in ranked_records:
        if len(selected) >= top_n:
            break

        if record.status != "completed":
            rejected.append(
                {
                    "molecule_id": record.molecule_id,
                    "reason": record.status,
                    "detail": record.error,
                }
            )
            continue

        if record.target_residue_hits < minimum_target_residues:
            record.diversity_status = "rejected_minimum_target_residues"
            rejected.append(
                {
                    "molecule_id": record.molecule_id,
                    "reason": record.diversity_status,
                    "detail": (
                        f"{record.target_residue_hits} target residues; "
                        f"{minimum_target_residues} required"
                    ),
                }
            )
            continue

        if (
            not allow_invalid_pocket_path
            and not record.pocket_path_valid
        ):
            record.diversity_status = "rejected_invalid_pocket_path"
            rejected.append(
                {
                    "molecule_id": record.molecule_id,
                    "reason": record.diversity_status,
                    "detail": (
                        "No lid interaction and no deep-pocket bypass "
                        "(requires >=2 side and >=1 bottom residues)"
                    ),
                }
            )
            continue

        if (
            not allow_same_scaffold
            and record.murcko_scaffold_smiles in selected_scaffolds
        ):
            record.diversity_status = "rejected_duplicate_murcko_scaffold"
            rejected.append(
                {
                    "molecule_id": record.molecule_id,
                    "reason": record.diversity_status,
                    "detail": record.murcko_scaffold_smiles,
                }
            )
            continue

        fingerprint = fingerprints.get(record.molecule_id)

        if fingerprint is None:
            record.diversity_status = "rejected_missing_fingerprint"
            rejected.append(
                {
                    "molecule_id": record.molecule_id,
                    "reason": record.diversity_status,
                    "detail": "",
                }
            )
            continue

        maximum_similarity = 0.0
        most_similar_rank: int | None = None

        if selected_fingerprints:
            similarities = list(
                DataStructs.BulkTanimotoSimilarity(
                    fingerprint,
                    selected_fingerprints,
                )
            )
            maximum_similarity = max(similarities)
            most_similar_rank = similarities.index(
                maximum_similarity
            ) + 1

        record.maximum_similarity_to_selected = maximum_similarity
        record.most_similar_selected_rank = most_similar_rank

        if maximum_similarity > max_similarity:
            record.diversity_status = "rejected_similarity"
            rejected.append(
                {
                    "molecule_id": record.molecule_id,
                    "reason": record.diversity_status,
                    "detail": (
                        f"maximum_similarity={maximum_similarity:.6f}; "
                        f"limit={max_similarity:.6f}; "
                        f"most_similar_selected_rank="
                        f"{most_similar_rank}"
                    ),
                }
            )
            continue

        record.diversity_status = "selected"
        selected.append(record)
        selected_fingerprints.append(fingerprint)
        selected_scaffolds.add(record.murcko_scaffold_smiles)

    return selected, rejected


def record_to_dict(
    record: MoleculeAnalysis,
    final_rank: int | None = None,
) -> dict[str, Any]:
    return {
        "final_diverse_rank": (
            final_rank if final_rank is not None else ""
        ),
        "molecule_id": record.molecule_id,
        "selected_rank_input": record.selected_rank_input,
        "ligand_filename": record.ligand_filename,
        "result_name": record.result_name,
        "status": record.status,
        "error": record.error,
        "binding_site_id": record.binding_site_id,
        "target_interaction_score": round(
            record.target_interaction_score,
            6,
        ),
        "lid_bonus": round(record.lid_bonus, 6),
        "region_bonus": round(record.region_bonus, 6),
        "total_plip_score": round(record.total_plip_score, 6),
        "target_residue_hits": record.target_residue_hits,
        "lid_hits": record.lid_hits,
        "side_hits": record.side_hits,
        "bottom_hits": record.bottom_hits,
        "lid_engaged": record.lid_engaged,
        "deep_pocket_bypass": record.deep_pocket_bypass,
        "pocket_path_valid": record.pocket_path_valid,
        "interaction_type_count": record.interaction_type_count,
        "target_interaction_count": record.target_interaction_count,
        "hit_labels": record.hit_labels,
        "interaction_summary": record.interaction_summary,
        "gnina_cnn_score": record.gnina_cnn_score,
        "gnina_cnn_affinity": record.gnina_cnn_affinity,
        "gnina_minimized_affinity": (
            record.gnina_minimized_affinity
        ),
        "murcko_scaffold_smiles": record.murcko_scaffold_smiles,
        "canonical_smiles": record.canonical_smiles,
        "diversity_status": record.diversity_status,
        "maximum_similarity_to_selected": (
            round(record.maximum_similarity_to_selected, 6)
            if record.maximum_similarity_to_selected is not None
            else ""
        ),
        "most_similar_selected_rank": (
            record.most_similar_selected_rank
            if record.most_similar_selected_rank is not None
            else ""
        ),
        "complex_pdb": str(record.complex_pdb),
        "selected_sdf": str(record.selected_sdf),
        "ligand_pdb": (
            str(record.ligand_pdb)
            if record.ligand_pdb is not None
            else ""
        ),
    }


def interaction_to_dict(
    interaction: InteractionRecord,
) -> dict[str, Any]:
    return {
        "molecule_id": interaction.molecule_id,
        "selected_rank_input": interaction.selected_rank_input,
        "complex_pdb": interaction.complex_pdb,
        "binding_site_id": interaction.binding_site_id,
        "interaction_type": interaction.interaction_type,
        "residue_label": interaction.residue_label,
        "residue_number": interaction.residue_number,
        "residue_type": interaction.residue_type,
        "residue_chain": interaction.residue_chain,
        "residue_group": interaction.residue_group,
        "distance_angstrom": (
            round(interaction.distance_angstrom, 6)
            if interaction.distance_angstrom is not None
            else ""
        ),
        "residue_weight": interaction.residue_weight,
        "interaction_weight": interaction.interaction_weight,
        "score_contribution": interaction.score_contribution,
        "detail": interaction.detail,
    }


def write_dict_csv(
    path: Path,
    rows: list[dict[str, Any]],
    fieldnames: list[str] | None = None,
) -> None:
    if fieldnames is None:
        fieldnames = list(rows[0].keys()) if rows else []

    with path.open("w", newline="") as handle:
        if not fieldnames:
            handle.write("")
            return

        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
            extrasaction="ignore",
        )
        writer.writeheader()
        writer.writerows(rows)


def write_residue_matrix(
    path: Path,
    ranked_records: list[MoleculeAnalysis],
    interactions_by_molecule: dict[
        str,
        list[InteractionRecord],
    ],
) -> None:
    residue_columns = [
        TARGET_RESIDUES[number]["label"]
        for number in TARGET_RESIDUES
    ]

    rows: list[dict[str, Any]] = []

    for rank, record in enumerate(ranked_records, start=1):
        row: dict[str, Any] = {
            "plip_rank": rank,
            "molecule_id": record.molecule_id,
            "total_plip_score": record.total_plip_score,
        }

        interaction_labels: dict[str, set[str]] = defaultdict(set)

        for interaction in interactions_by_molecule.get(
            record.molecule_id,
            [],
        ):
            interaction_labels[
                interaction.residue_label
            ].add(interaction.interaction_type)

        for residue_label in residue_columns:
            row[residue_label] = "|".join(
                sorted(interaction_labels.get(residue_label, set()))
            )

        rows.append(row)

    write_dict_csv(
        path=path,
        rows=rows,
        fieldnames=[
            "plip_rank",
            "molecule_id",
            "total_plip_score",
            *residue_columns,
        ],
    )


def copy_selected_files(
    selected: list[MoleculeAnalysis],
    output_root: Path,
) -> None:
    selected_root = output_root / "TOP20_DIVERSE"
    complexes_dir = selected_root / "complexes"
    sdf_dir = selected_root / "selected_sdf"
    ligand_pdb_dir = selected_root / "ligand_pdb"

    complexes_dir.mkdir(parents=True, exist_ok=False)
    sdf_dir.mkdir(parents=True, exist_ok=False)
    ligand_pdb_dir.mkdir(parents=True, exist_ok=False)

    for final_rank, record in enumerate(selected, start=1):
        prefix = f"rank{final_rank:03d}__{record.molecule_id}"

        shutil.copy2(
            record.complex_pdb,
            complexes_dir / f"{prefix}_complex.pdb",
        )
        shutil.copy2(
            record.selected_sdf,
            sdf_dir / f"{prefix}.sdf",
        )

        if record.ligand_pdb is not None and record.ligand_pdb.is_file():
            shutil.copy2(
                record.ligand_pdb,
                ligand_pdb_dir / f"{prefix}_ligand.pdb",
            )


def write_similarity_matrix(
    path: Path,
    selected: list[MoleculeAnalysis],
    fingerprints: dict[str, Any],
    dependencies: dict[str, Any],
) -> None:
    DataStructs = dependencies["DataStructs"]
    labels = [
        f"rank{rank:03d}__{record.molecule_id}"
        for rank, record in enumerate(selected, start=1)
    ]

    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["molecule", *labels])

        for row_index, row_record in enumerate(selected):
            row_fp = fingerprints[row_record.molecule_id]
            values: list[str] = []

            for column_record in selected:
                similarity = DataStructs.TanimotoSimilarity(
                    row_fp,
                    fingerprints[column_record.molecule_id],
                )
                values.append(f"{similarity:.6f}")

            writer.writerow([labels[row_index], *values])


def main() -> None:
    args = parse_arguments()
    input_folder = Path(
        args.plip_input_folder
    ).expanduser().resolve()

    errors: list[str] = []

    if not input_folder.is_dir():
        errors.append(f"Input folder not found: {input_folder}")

    selection_csv = input_folder / "selected_top_unique.csv"

    if not selection_csv.is_file():
        errors.append(
            f"Selection CSV not found: {selection_csv}"
        )

    if args.top < 1:
        errors.append("--top must be at least 1.")

    if not (0.0 <= args.max_similarity <= 1.0):
        errors.append(
            "--max-similarity must be between 0 and 1."
        )

    if args.min_target_residues < 1:
        errors.append(
            "--min-target-residues must be at least 1."
        )

    receptor_chain = (
        args.receptor_chain.strip()
        if args.receptor_chain is not None
        else None
    )

    if receptor_chain == "":
        receptor_chain = None

    if receptor_chain is not None and len(receptor_chain) != 1:
        errors.append(
            "--receptor-chain must contain exactly one character."
        )

    ligand_chain = args.ligand_chain.strip()
    ligand_resname = args.ligand_resname.strip().upper()

    if len(ligand_chain) != 1:
        errors.append(
            "--ligand-chain must contain exactly one character."
        )

    if not re.fullmatch(r"[A-Z0-9]{1,3}", ligand_resname):
        errors.append(
            "--ligand-resname must contain 1-3 letters or numbers."
        )

    if errors:
        print("\nINPUT ERROR", file=sys.stderr)
        print("-----------", file=sys.stderr)

        for error in errors:
            print(f"\n{error}", file=sys.stderr)

        sys.exit(1)

    try:
        dependencies = import_dependencies()
        rows = read_selection_table(selection_csv)
        records = build_input_records(input_folder, rows)
    except Exception as error:
        print(f"\nSETUP ERROR\n-----------\n{error}", file=sys.stderr)
        sys.exit(1)

    record_errors = validate_input_records(records)

    if record_errors:
        print("\nINPUT FILE ERRORS", file=sys.stderr)
        print("-----------------", file=sys.stderr)

        for error in record_errors:
            print(error, file=sys.stderr)

        sys.exit(1)

    output_root = input_folder / safe_name(args.output_name)

    plip_config = dependencies["plip_config"]
    rdBase = dependencies["rdBase"]

    print("\nCD97 PLIP target scoring and diverse selection")
    print("----------------------------------------------")
    print(f"Input folder:          {input_folder}")
    print(f"Complexes detected:    {len(records)}")
    print(f"PLIP version:          {plip_config.__version__}")
    print(f"RDKit version:         {rdBase.rdkitVersion}")
    print(f"Target residues:       {len(TARGET_RESIDUES)}")
    print(
        f"Receptor chain:        "
        f"{receptor_chain if receptor_chain else 'any'}"
    )
    print(
        f"Ligand identifier:     "
        f"{ligand_resname}:{ligand_chain}:{args.ligand_resnum}"
    )
    print(f"Requested final hits:  {args.top}")
    print(
        f"Maximum similarity:    {args.max_similarity:.2f} "
        "(selected pairs must be <= this value)"
    )
    print(
        f"Unique Murcko scaffold:"
        f" {'no' if args.allow_same_scaffold else 'yes'}"
    )
    print(
        f"Minimum target hits:   {args.min_target_residues}"
    )
    print(
        f"Pocket-path rule:      "
        f"{'disabled' if args.allow_invalid_pocket_path else 'enabled'}"
    )
    print(
        f"PLIP hydrogen mode:    "
        f"{'use input hydrogens only' if args.plip_nohydro else 'standard PLIP protonation'}"
    )
    print(f"Output folder:         {output_root}")

    if args.dry_run:
        print("\nDry run completed. PLIP analysis was not started.")
        return

    if output_root.exists():
        if args.force:
            shutil.rmtree(output_root)
        else:
            raise SystemExit(
                "\nOutput folder already exists:\n"
                f"  {output_root}\n\n"
                "Use --force to delete and rebuild it."
            )

    output_root.mkdir(parents=True, exist_ok=False)

    # Reduce PLIP console noise while retaining errors.
    logging.getLogger("plip").setLevel(logging.WARNING)

    all_interactions: list[InteractionRecord] = []
    interactions_by_molecule: dict[
        str,
        list[InteractionRecord],
    ] = {}
    start_time = time.monotonic()

    for index, record in enumerate(records, start=1):
        print(
            f"[{index}/{len(records)}] PLIP "
            f"{record.complex_pdb.name}"
        )

        try:
            interactions = run_plip_for_record(
                record=record,
                dependencies=dependencies,
                receptor_chain=receptor_chain,
                ligand_resname=ligand_resname,
                ligand_chain=ligand_chain,
                ligand_resnum=args.ligand_resnum,
                plip_nohydro=args.plip_nohydro,
            )
            all_interactions.extend(interactions)
            interactions_by_molecule[
                record.molecule_id
            ] = interactions
        except Exception as error:
            record.status = "failed_plip"
            record.error = str(error)
            interactions_by_molecule[record.molecule_id] = []

    fingerprints = add_chemical_descriptors(
        records=records,
        dependencies=dependencies,
    )

    ranked_records = sorted(records, key=ranking_key)

    selected, rejected = select_diverse_records(
        ranked_records=ranked_records,
        fingerprints=fingerprints,
        dependencies=dependencies,
        top_n=args.top,
        max_similarity=args.max_similarity,
        minimum_target_residues=args.min_target_residues,
        allow_same_scaffold=args.allow_same_scaffold,
        allow_invalid_pocket_path=args.allow_invalid_pocket_path,
    )

    # Mark completed but not selected records for clarity.
    selected_ids = {record.molecule_id for record in selected}

    for record in ranked_records:
        if (
            record.status == "completed"
            and record.molecule_id not in selected_ids
            and not record.diversity_status
        ):
            record.diversity_status = "not_reached_before_top_n_filled"

    all_ranked_rows = [
        {
            "plip_rank": rank,
            **record_to_dict(record),
        }
        for rank, record in enumerate(ranked_records, start=1)
    ]

    selected_rows = [
        record_to_dict(record, final_rank=rank)
        for rank, record in enumerate(selected, start=1)
    ]

    interaction_rows = [
        interaction_to_dict(interaction)
        for interaction in all_interactions
    ]

    write_dict_csv(
        output_root / "all_100_plip_ranked.csv",
        all_ranked_rows,
    )
    write_dict_csv(
        output_root / "all_target_interactions.csv",
        interaction_rows,
        fieldnames=[
            "molecule_id",
            "selected_rank_input",
            "complex_pdb",
            "binding_site_id",
            "interaction_type",
            "residue_label",
            "residue_number",
            "residue_type",
            "residue_chain",
            "residue_group",
            "distance_angstrom",
            "residue_weight",
            "interaction_weight",
            "score_contribution",
            "detail",
        ],
    )
    write_dict_csv(
        output_root / "top20_diverse_selected.csv",
        selected_rows,
    )
    write_dict_csv(
        output_root / "diversity_rejections.csv",
        rejected,
        fieldnames=["molecule_id", "reason", "detail"],
    )
    write_residue_matrix(
        output_root / "target_residue_interaction_matrix.csv",
        ranked_records,
        interactions_by_molecule,
    )

    copy_selected_files(selected, output_root)

    write_similarity_matrix(
        output_root
        / "TOP20_DIVERSE"
        / "pairwise_morgan_tanimoto.csv",
        selected,
        fingerprints,
        dependencies,
    )

    runtime_seconds = time.monotonic() - start_time
    failure_count = sum(
        1 for record in records if record.status != "completed"
    )

    parameters = {
        "created": datetime.now().isoformat(timespec="seconds"),
        "input_folder": str(input_folder),
        "selection_csv": str(selection_csv),
        "output_folder": str(output_root),
        "plip_version": plip_config.__version__,
        "rdkit_version": rdBase.rdkitVersion,
        "complexes_analyzed": len(records),
        "analysis_failures": failure_count,
        "requested_final_hits": args.top,
        "selected_final_hits": len(selected),
        "maximum_pairwise_similarity": args.max_similarity,
        "fingerprint": (
            "Morgan radius=2, fpSize=2048, chirality=True, "
            "bondTypes=True"
        ),
        "unique_murcko_scaffold_required": (
            not args.allow_same_scaffold
        ),
        "minimum_target_residues": args.min_target_residues,
        "pocket_path_rule_required": (
            not args.allow_invalid_pocket_path
        ),
        "pocket_path_rule": (
            "at least one lid residue OR "
            "at least two side residues and one bottom residue"
        ),
        "receptor_chain": receptor_chain,
        "ligand_identifier": (
            f"{ligand_resname}:{ligand_chain}:"
            f"{args.ligand_resnum}"
        ),
        "plip_nohydro": args.plip_nohydro,
        "target_residues": TARGET_RESIDUES,
        "group_weights": GROUP_WEIGHTS,
        "interaction_weights": INTERACTION_WEIGHTS,
        "bonuses": {
            "both_lid_residues": 6.0,
            "one_lid_residue": 2.0,
            "all_three_regions": 4.0,
        },
        "score_counting_rule": (
            "one representative per residue, receptor chain, "
            "and interaction type"
        ),
        "ranking": (
            "total PLIP score; target residue count; lid count; "
            "interaction-type count; input rank; molecule identifier"
        ),
        "pose_handling": (
            "Analysis only. Original pose-preserved PDB/SDF files are "
            "copied; no redocking, minimization, alignment, rotation, "
            "translation, or conformer generation."
        ),
        "runtime_seconds": round(runtime_seconds, 3),
    }

    (output_root / "run_parameters.json").write_text(
        json.dumps(parameters, indent=2) + "\n"
    )

    readme = f"""CD97 PLIP TARGET-RESIDUE SELECTION

Analyzed complexes: {len(records)}
PLIP failures: {failure_count}
Requested diverse molecules: {args.top}
Selected diverse molecules: {len(selected)}

Primary ranking:
1. weighted CD97 target-residue PLIP score
2. number of target residues contacted
3. number of lid residues contacted
4. number of interaction types
5. input rank
6. molecule identifier

Pocket validity:
- accepted when W685 or F760 is contacted, OR
- accepted as a deep-pocket bypass when at least two side residues and
  at least one bottom residue are contacted.

Diversity:
- Morgan/ECFP4-like fingerprint: radius 2, 2048 bits, chirality included
- maximum allowed pairwise Tanimoto similarity: {args.max_similarity:.2f}
- unique Bemis-Murcko scaffold required: {not args.allow_same_scaffold}

Important:
For diversity, similarity must be <= {args.max_similarity:.2f}.
Similarity > {args.max_similarity:.2f} means the molecules are relatively
similar and is therefore rejected by the default selection rule.

Pose preservation:
PLIP analyzes copies of the supplied complexes before later GNINA rescoring.
No pose is redocked, minimized, aligned, rotated, translated, or regenerated.
The files under TOP20_DIVERSE are direct copies of the original
pose-preserved files.

Key files:
- all_100_plip_ranked.csv
- all_target_interactions.csv
- target_residue_interaction_matrix.csv
- top20_diverse_selected.csv
- diversity_rejections.csv
- TOP20_DIVERSE/complexes/
- TOP20_DIVERSE/selected_sdf/
- TOP20_DIVERSE/pairwise_morgan_tanimoto.csv
"""
    (output_root / "README.txt").write_text(readme)

    print("\nPLIP analysis and diverse selection completed")
    print("---------------------------------------------")
    print(f"Complexes analyzed:  {len(records)}")
    print(f"Analysis failures:   {failure_count}")
    print(f"Diverse selected:    {len(selected)} / {args.top}")
    print(
        f"Full PLIP ranking:   "
        f"{output_root / 'all_100_plip_ranked.csv'}"
    )
    print(
        f"Interaction details: "
        f"{output_root / 'all_target_interactions.csv'}"
    )
    print(
        f"Final selection:     "
        f"{output_root / 'top20_diverse_selected.csv'}"
    )
    print(
        f"Final complexes:     "
        f"{output_root / 'TOP20_DIVERSE' / 'complexes'}"
    )

    if len(selected) < args.top:
        print(
            f"\nWARNING: only {len(selected)} molecules satisfied all "
            f"interaction, pocket-path, scaffold, and similarity rules. "
            "The script did not silently relax the diversity threshold.",
            file=sys.stderr,
        )
        sys.exit(2)

    if failure_count:
        sys.exit(3)


if __name__ == "__main__":
    main()
