#!/usr/bin/env python3
"""Prepare SMILES from CSV files for AutoDock Vina/AutoDock-GPU.

Workflow per molecule:
1. Read and validate SMILES.
2. Keep the largest molecular fragment (removes common counterions/salts).
3. Adjust the molecule to one protonation state at the requested pH using Open Babel.
4. Generate multiple 3D conformers with RDKit ETKDGv3.
5. Minimize each conformer with MMFF94s; fall back to UFF when needed.
6. Keep the lowest-energy conformer.
7. Write an explicit-H SDF and an AutoDock PDBQT using Meeko.

The script streams CSV rows and therefore does not load an entire file into memory.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

try:
    from rdkit import Chem
    from rdkit.Chem import AllChem
except ImportError as exc:
    raise SystemExit(
        "RDKit is not available. Activate the intended conda environment and run:\n"
        "  conda install -c conda-forge rdkit"
    ) from exc

try:
    from openbabel import pybel
except ImportError as exc:
    raise SystemExit(
        "Open Babel Python bindings are not available. Run:\n"
        "  conda install -c conda-forge openbabel"
    ) from exc

try:
    from meeko import MoleculePreparation, PDBQTWriterLegacy
except ImportError as exc:
    raise SystemExit(
        "Meeko is not available. Run:\n"
        "  conda install -c conda-forge meeko"
    ) from exc


SMILES_CANDIDATES = (
    "smiles",
    "canonical_smiles",
    "isomeric_smiles",
    "canonicalsmiles",
    "isomericsmiles",
    "molecule_smiles",
    "structure",
)

ID_CANDIDATES = (
    "id",
    "name",
    "compound_id",
    "compoundid",
    "molecule_id",
    "moleculeid",
    "zinc_id",
    "zincid",
    "title",
)

LOG_FIELDS = (
    "source_file",
    "row_number",
    "input_id",
    "input_smiles",
    "prepared_smiles",
    "formal_charge",
    "forcefield",
    "energy_kcal_mol",
    "output_base",
    "status",
    "message",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare CSV SMILES as minimized 3D SDF and AutoDock PDBQT files."
    )
    parser.add_argument(
        "--input",
        required=True,
        help="A CSV file or a directory containing CSV files.",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output directory for SDF, PDBQT, and preparation_log.csv.",
    )
    parser.add_argument(
        "--pattern",
        default="*.csv",
        help="CSV filename pattern when --input is a directory (default: *.csv).",
    )
    parser.add_argument(
        "--smiles-column",
        default=None,
        help="SMILES column name. If omitted, the script attempts to detect it.",
    )
    parser.add_argument(
        "--id-column",
        default=None,
        help="Compound-ID column name. If omitted, the script attempts to detect it.",
    )
    parser.add_argument(
        "--ph",
        type=float,
        default=7.4,
        help="pH used by Open Babel CorrectForPH (default: 7.4).",
    )
    parser.add_argument(
        "--num-confs",
        type=int,
        default=5,
        help=(
            "Number of 3D conformers generated per molecule before selecting the "
            "lowest-energy conformer (default: 5)."
        ),
    )
    parser.add_argument(
        "--max-iters",
        type=int,
        default=500,
        help="Maximum minimization iterations per conformer (default: 500).",
    )
    parser.add_argument(
        "--max-molecules",
        type=int,
        default=None,
        help="Stop after this many input rows across all CSV files; useful for testing.",
    )
    parser.add_argument(
        "--keep-salts",
        action="store_true",
        help="Keep all fragments instead of retaining only the largest fragment.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output files instead of skipping them.",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=100,
        help="Print progress every N processed rows (default: 100).",
    )
    args = parser.parse_args()

    if args.num_confs < 1:
        parser.error("--num-confs must be at least 1")
    if args.max_iters < 1:
        parser.error("--max-iters must be at least 1")
    if args.max_molecules is not None and args.max_molecules < 1:
        parser.error("--max-molecules must be at least 1")
    return args


def normalized_header(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.strip().lower()).strip("_")


def resolve_column(
    fieldnames: Sequence[str],
    requested: Optional[str],
    candidates: Sequence[str],
    kind: str,
) -> Optional[str]:
    mapping = {normalized_header(name): name for name in fieldnames if name is not None}

    if requested:
        key = normalized_header(requested)
        if key not in mapping:
            available = ", ".join(fieldnames)
            raise ValueError(
                f"Requested {kind} column '{requested}' was not found. "
                f"Available columns: {available}"
            )
        return mapping[key]

    for candidate in candidates:
        if candidate in mapping:
            return mapping[candidate]
    return None


def sniff_dialect(csv_path: Path) -> csv.Dialect:
    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        sample = handle.read(8192)
    try:
        return csv.Sniffer().sniff(sample, delimiters=",\t;")
    except csv.Error:
        return csv.excel


def discover_csv_files(input_path: Path, pattern: str) -> List[Path]:
    if input_path.is_file():
        return [input_path]
    if input_path.is_dir():
        files = sorted(path for path in input_path.glob(pattern) if path.is_file())
        if not files:
            raise FileNotFoundError(
                f"No files matching '{pattern}' were found in {input_path}"
            )
        return files
    raise FileNotFoundError(f"Input path does not exist: {input_path}")


def choose_largest_fragment(mol: Chem.Mol) -> Chem.Mol:
    fragments = Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=True)
    if not fragments:
        raise ValueError("No molecular fragment could be generated")
    return max(
        fragments,
        key=lambda frag: (frag.GetNumHeavyAtoms(), frag.GetNumAtoms()),
    )


def prepare_smiles_at_ph(
    input_smiles: str,
    ph: float,
    keep_salts: bool,
) -> Tuple[str, Chem.Mol]:
    rd_mol = Chem.MolFromSmiles(input_smiles)
    if rd_mol is None:
        raise ValueError("RDKit could not parse the input SMILES")

    if not keep_salts:
        rd_mol = choose_largest_fragment(rd_mol)

    clean_smiles = Chem.MolToSmiles(rd_mol, isomericSmiles=True)

    ob_mol = pybel.readstring("smi", clean_smiles)
    if ob_mol is None or ob_mol.OBMol.NumAtoms() == 0:
        raise ValueError("Open Babel could not parse the standardized SMILES")

    ob_mol.OBMol.CorrectForPH(float(ph))
    ph_smiles_text = ob_mol.write("can").strip()
    if not ph_smiles_text:
        raise ValueError("Open Babel did not return a pH-adjusted SMILES")

    ph_smiles = ph_smiles_text.split()[0]
    ph_mol = Chem.MolFromSmiles(ph_smiles)
    if ph_mol is None:
        raise ValueError("RDKit could not parse the pH-adjusted SMILES")

    canonical_ph_smiles = Chem.MolToSmiles(ph_mol, isomericSmiles=True)
    return canonical_ph_smiles, ph_mol


def deterministic_seed(text: str) -> int:
    digest = hashlib.sha256(text.encode("utf-8", errors="replace")).digest()
    # RDKit expects a signed 32-bit-compatible positive integer.
    return int.from_bytes(digest[:4], "big") % 2_147_483_647


def embed_conformers(mol: Chem.Mol, num_confs: int, seed: int) -> List[int]:
    params = AllChem.ETKDGv3()
    params.randomSeed = int(seed)
    params.enforceChirality = True
    params.pruneRmsThresh = 0.5
    params.useSmallRingTorsions = True
    params.useMacrocycleTorsions = True
    params.numThreads = 1

    conf_ids = list(AllChem.EmbedMultipleConfs(mol, numConfs=num_confs, params=params))
    if conf_ids:
        return conf_ids

    # Fallback for difficult structures.
    params = AllChem.ETKDGv3()
    params.randomSeed = int(seed)
    params.enforceChirality = True
    params.useRandomCoords = True
    params.numThreads = 1
    if hasattr(params, "maxIterations"):
        params.maxIterations = 2000

    conf_ids = list(AllChem.EmbedMultipleConfs(mol, numConfs=num_confs, params=params))
    if not conf_ids:
        raise ValueError("RDKit failed to generate a 3D conformer")
    return conf_ids


def minimize_and_select(
    mol: Chem.Mol,
    conf_ids: Sequence[int],
    max_iters: int,
) -> Tuple[Chem.Mol, str, float]:
    energies: List[Tuple[float, int]] = []
    forcefield_name = ""

    if AllChem.MMFFHasAllMoleculeParams(mol):
        properties = AllChem.MMFFGetMoleculeProperties(mol, mmffVariant="MMFF94s")
        if properties is None:
            raise ValueError("MMFF94s properties could not be assigned")
        forcefield_name = "MMFF94s"
        for conf_id in conf_ids:
            ff = AllChem.MMFFGetMoleculeForceField(
                mol, properties, confId=int(conf_id)
            )
            if ff is None:
                continue
            ff.Initialize()
            ff.Minimize(maxIts=max_iters)
            energies.append((float(ff.CalcEnergy()), int(conf_id)))
    else:
        forcefield_name = "UFF"
        for conf_id in conf_ids:
            try:
                ff = AllChem.UFFGetMoleculeForceField(mol, confId=int(conf_id))
            except Exception:
                ff = None
            if ff is None:
                continue
            ff.Initialize()
            ff.Minimize(maxIts=max_iters)
            energies.append((float(ff.CalcEnergy()), int(conf_id)))

    if not energies:
        raise ValueError("Neither MMFF94s nor UFF could minimize this molecule")

    best_energy, best_conf_id = min(energies, key=lambda item: item[0])
    selected = Chem.Mol(mol)
    selected_conf = Chem.Conformer(mol.GetConformer(best_conf_id))
    selected.RemoveAllConformers()
    selected.AddConformer(selected_conf, assignId=True)

    return selected, forcefield_name, best_energy


def make_3d_molecule(
    ph_mol: Chem.Mol,
    num_confs: int,
    max_iters: int,
    seed_text: str,
) -> Tuple[Chem.Mol, str, float]:
    mol_h = Chem.AddHs(ph_mol, addCoords=False)
    conf_ids = embed_conformers(
        mol_h,
        num_confs=num_confs,
        seed=deterministic_seed(seed_text),
    )
    return minimize_and_select(mol_h, conf_ids, max_iters)


def meeko_pdbqt_string(mol: Chem.Mol) -> str:
    preparator = MoleculePreparation()

    # Meeko 0.6/0.7 exposes prepare(); older releases also made the object callable.
    if hasattr(preparator, "prepare"):
        setups = preparator.prepare(mol)
    else:
        setups = preparator(mol)

    if not setups:
        raise ValueError("Meeko did not generate a ligand setup")

    result = PDBQTWriterLegacy.write_string(setups[0])

    # Some Meeko releases return only the string; others return
    # (string, success_boolean, error_message).
    if isinstance(result, tuple):
        pdbqt_text = result[0]
        is_ok = bool(result[1]) if len(result) > 1 else True
        error_message = str(result[2]) if len(result) > 2 else ""
        if not is_ok:
            raise ValueError(f"Meeko PDBQT writing failed: {error_message}")
    else:
        pdbqt_text = result

    if not isinstance(pdbqt_text, str) or not pdbqt_text.strip():
        raise ValueError("Meeko returned an empty PDBQT string")
    return pdbqt_text


def safe_token(value: str, max_length: int = 60) -> str:
    token = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip())
    token = token.strip("._-") or "compound"
    return token[:max_length]


def write_sdf(mol: Chem.Mol, output_path: Path) -> None:
    writer = Chem.SDWriter(str(output_path))
    if writer is None:
        raise OSError(f"Could not open SDF output: {output_path}")
    try:
        writer.write(mol)
    finally:
        writer.close()


def detect_smiles_column_from_first_row(
    fieldnames: Sequence[str], first_row: Dict[str, str]
) -> Optional[str]:
    for field in fieldnames:
        value = (first_row.get(field) or "").strip()
        if value and Chem.MolFromSmiles(value) is not None:
            return field
    return None


def process_csv(
    csv_path: Path,
    output_root: Path,
    log_writer: csv.DictWriter,
    args: argparse.Namespace,
    processed_so_far: int,
) -> int:
    dialect = sniff_dialect(csv_path)
    batch_name = safe_token(csv_path.stem, max_length=100)
    sdf_dir = output_root / "sdf" / batch_name
    pdbqt_dir = output_root / "pdbqt" / batch_name
    sdf_dir.mkdir(parents=True, exist_ok=True)
    pdbqt_dir.mkdir(parents=True, exist_ok=True)

    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle, dialect=dialect)
        if not reader.fieldnames:
            raise ValueError(f"CSV has no header row: {csv_path}")

        fieldnames = [field.strip() if field else field for field in reader.fieldnames]
        reader.fieldnames = fieldnames

        first_row = next(reader, None)
        if first_row is None:
            print(f"Skipping empty CSV: {csv_path}", file=sys.stderr)
            return processed_so_far

        smiles_column = resolve_column(
            fieldnames,
            args.smiles_column,
            SMILES_CANDIDATES,
            "SMILES",
        )
        if smiles_column is None:
            smiles_column = detect_smiles_column_from_first_row(fieldnames, first_row)
        if smiles_column is None:
            raise ValueError(
                f"Could not detect a SMILES column in {csv_path}. "
                "Use --smiles-column COLUMN_NAME."
            )

        id_column = resolve_column(
            fieldnames,
            args.id_column,
            ID_CANDIDATES,
            "ID",
        )

        from itertools import chain

        # Process the already-read first row and then continue streaming the file.
        for row_number, row in enumerate(chain([first_row], reader), start=2):
            if args.max_molecules is not None and processed_so_far >= args.max_molecules:
                return processed_so_far

            processed_so_far += 1
            raw_smiles = (row.get(smiles_column) or "").strip()
            raw_id = (row.get(id_column) or "").strip() if id_column else ""
            if not raw_id:
                raw_id = f"row_{row_number}"

            output_base = (
                f"{batch_name}__r{row_number:08d}__{safe_token(raw_id)}"
            )
            sdf_path = sdf_dir / f"{output_base}.sdf"
            pdbqt_path = pdbqt_dir / f"{output_base}.pdbqt"

            log_row = {
                "source_file": csv_path.name,
                "row_number": row_number,
                "input_id": raw_id,
                "input_smiles": raw_smiles,
                "prepared_smiles": "",
                "formal_charge": "",
                "forcefield": "",
                "energy_kcal_mol": "",
                "output_base": output_base,
                "status": "",
                "message": "",
            }

            try:
                if not raw_smiles:
                    raise ValueError("SMILES field is empty")

                if (
                    not args.overwrite
                    and sdf_path.exists()
                    and pdbqt_path.exists()
                ):
                    log_row["status"] = "skipped_existing"
                    log_row["message"] = "Both output files already exist"
                    log_writer.writerow(log_row)
                    continue

                prepared_smiles, ph_mol = prepare_smiles_at_ph(
                    raw_smiles,
                    ph=args.ph,
                    keep_salts=args.keep_salts,
                )
                mol_3d, forcefield, energy = make_3d_molecule(
                    ph_mol,
                    num_confs=args.num_confs,
                    max_iters=args.max_iters,
                    seed_text=f"{csv_path.name}|{row_number}|{raw_id}|{prepared_smiles}",
                )

                mol_3d.SetProp("_Name", output_base)
                mol_3d.SetProp("Input_ID", raw_id)
                mol_3d.SetProp("Input_SMILES", raw_smiles)
                mol_3d.SetProp("Prepared_SMILES", prepared_smiles)
                mol_3d.SetProp("Preparation_pH", f"{args.ph:g}")
                mol_3d.SetProp("Minimization_forcefield", forcefield)
                mol_3d.SetProp("Selected_conformer_energy_kcal_mol", f"{energy:.6f}")

                pdbqt_text = meeko_pdbqt_string(mol_3d)
                write_sdf(mol_3d, sdf_path)
                pdbqt_path.write_text(pdbqt_text, encoding="utf-8")

                log_row["prepared_smiles"] = prepared_smiles
                log_row["formal_charge"] = Chem.GetFormalCharge(ph_mol)
                log_row["forcefield"] = forcefield
                log_row["energy_kcal_mol"] = f"{energy:.6f}"
                log_row["status"] = "success"
            except Exception as exc:
                # Remove incomplete output from a failed molecule.
                for path in (sdf_path, pdbqt_path):
                    try:
                        if path.exists():
                            path.unlink()
                    except OSError:
                        pass
                log_row["status"] = "failed"
                log_row["message"] = f"{type(exc).__name__}: {exc}"

            log_writer.writerow(log_row)

            if args.progress_every and processed_so_far % args.progress_every == 0:
                print(
                    f"Processed {processed_so_far} rows; latest file: {csv_path.name}",
                    flush=True,
                )

    return processed_so_far


def main() -> int:
    args = parse_args()
    input_path = Path(args.input).expanduser().resolve()
    output_root = Path(args.output).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    try:
        csv_files = discover_csv_files(input_path, args.pattern)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    log_path = output_root / "preparation_log.csv"
    processed = 0

    print(f"Input:  {input_path}")
    print(f"Output: {output_root}")
    print(f"CSV files: {len(csv_files)}")
    print(f"pH: {args.ph:g}; conformers per molecule: {args.num_confs}")

    with log_path.open("w", encoding="utf-8", newline="") as log_handle:
        log_writer = csv.DictWriter(log_handle, fieldnames=LOG_FIELDS)
        log_writer.writeheader()

        for csv_path in csv_files:
            if args.max_molecules is not None and processed >= args.max_molecules:
                break
            print(f"Reading {csv_path}", flush=True)
            try:
                processed = process_csv(
                    csv_path,
                    output_root,
                    log_writer,
                    args,
                    processed,
                )
            except Exception as exc:
                print(f"ERROR in {csv_path}: {type(exc).__name__}: {exc}", file=sys.stderr)

    print(f"Finished. Input rows examined: {processed}")
    print(f"Log: {log_path}")
    print(f"PDBQT root: {output_root / 'pdbqt'}")
    print(f"SDF root:   {output_root / 'sdf'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
