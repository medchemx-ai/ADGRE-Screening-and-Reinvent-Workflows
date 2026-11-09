#!/usr/bin/env python3
"""
Select the top unique GNINA-scored molecules and prepare PLIP-ready PDB complexes.

Example
-------
python3 select_top_gnina_unique_for_plip.py \
    /path/to/GNINA_OUT \
    --receptor /path/to/receptor.pdbqt \
    --top 100 \
    --dry-run

python3 select_top_gnina_unique_for_plip.py \
    /path/to/GNINA_OUT \
    --receptor /path/to/receptor.pdbqt \
    --top 100

Default output
--------------
GNINA_OUT/TOP100_PLIP_INPUT/
    complexes/                 PLIP-ready receptor-ligand PDB files
    ligand_pdb/                ligand-only PDB files
    selected_sdf/              copied GNINA minimized SDF files
    selected_top_unique.csv
    duplicates_skipped.csv
    failed_candidates.csv
    receptor_for_plip.pdb
    run_parameters.json
    README_PLIP.txt

Selection
---------
- Sort by CNNscore from highest to lowest.
- Tie-break by CNNaffinity (high to low), then minimized affinity
  (more negative first).
- Remove structural duplicates using the full InChIKey by default.
- Continue down the ranked table until the requested number of unique,
  successfully prepared molecules is reached.

Requirements
------------
- Python 3 standard library
- Open Babel command-line executable: obabel
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

SUCCESS_STATUSES = {"completed", "skipped_existing"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Select top unique GNINA hits and create PLIP-ready complexes."
    )
    p.add_argument(
        "gnina_output_folder",
        help="Folder containing gnina_scores.csv and minimized_sdf/.",
    )
    p.add_argument("--receptor", required=True, help="Receptor PDB or PDBQT file.")
    p.add_argument("--top", type=int, default=100, help="Unique molecules to select.")
    p.add_argument(
        "--score-file", default="gnina_scores.csv", help="GNINA score CSV filename."
    )
    p.add_argument(
        "--score-column",
        default="cnn_score",
        help="Primary score column; larger values rank first.",
    )
    p.add_argument(
        "--output-name",
        default="TOP100_PLIP_INPUT",
        help="Output subfolder created inside the GNINA output folder.",
    )
    p.add_argument(
        "--dedup-key",
        choices=("full_inchikey", "connectivity", "canonical_smiles", "filename"),
        default="full_inchikey",
        help="Definition used to remove duplicate molecules.",
    )
    p.add_argument("--obabel", default=None, help="Full path to obabel if not on PATH.")
    p.add_argument("--ligand-resname", default="LIG")
    p.add_argument("--ligand-chain", default="Z")
    p.add_argument("--ligand-resnum", type=int, default=1)
    p.add_argument("--force", action="store_true", help="Rebuild existing output folder.")
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def safe_name(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", text).strip("._") or "molecule"


def find_obabel(explicit: str | None) -> Path:
    candidates: list[Path] = []
    if explicit:
        candidates.append(Path(explicit).expanduser())
    hit = shutil.which("obabel")
    if hit:
        candidates.append(Path(hit))
    if os.environ.get("CONDA_PREFIX"):
        candidates.append(Path(os.environ["CONDA_PREFIX"]) / "bin" / "obabel")
    for item in candidates:
        if item.is_file() and os.access(item, os.X_OK):
            return item.resolve()
    raise FileNotFoundError(
        "Open Babel 'obabel' was not found. Install Open Babel or pass "
        "--obabel /full/path/to/obabel"
    )


def to_float(value: Any) -> float | None:
    try:
        text = str(value).strip()
        return float(text) if text else None
    except (TypeError, ValueError):
        return None


def read_rows(path: Path, score_column: str) -> list[dict[str, Any]]:
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        fields = set(reader.fieldnames or [])
        required = {"ligand_filename", "result_name", "status", "output_sdf", score_column}
        missing = required - fields
        if missing:
            raise RuntimeError("Missing CSV columns: " + ", ".join(sorted(missing)))
        rows = []
        for number, raw in enumerate(reader, start=2):
            row = dict(raw)
            row["_csv_row"] = number
            row["_primary"] = to_float(row.get(score_column))
            row["_cnn_affinity"] = to_float(row.get("cnn_affinity"))
            row["_min_affinity"] = to_float(row.get("minimized_affinity_kcal_mol"))
            row["_original_rank"] = to_float(row.get("cnn_score_rank"))
            rows.append(row)
    return rows


def rank_key(row: dict[str, Any]) -> tuple[Any, ...]:
    return (
        -row["_primary"],
        -(row["_cnn_affinity"] if row["_cnn_affinity"] is not None else -1e99),
        row["_min_affinity"] if row["_min_affinity"] is not None else 1e99,
        row["_original_rank"] if row["_original_rank"] is not None else 1e99,
        row.get("ligand_filename", ""),
    )


def run(command: list[str], label: str) -> subprocess.CompletedProcess[str]:
    proc = subprocess.run(command, capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        raise RuntimeError(
            f"{label} failed (return code {proc.returncode}).\n"
            f"Command: {' '.join(shlex.quote(x) for x in command)}\n"
            f"STDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
        )
    return proc


def sdf_path_for(row: dict[str, Any], root: Path) -> Path:
    raw = str(row.get("output_sdf", "")).strip()
    if raw and Path(raw).expanduser().is_file():
        return Path(raw).expanduser().resolve()
    name = safe_name(str(row.get("result_name", "")))
    return (root / "minimized_sdf" / f"{name}_minimized.sdf").resolve()


def inchikey(obabel: Path, sdf: Path) -> str:
    proc = run(
        [str(obabel), "-isdf", str(sdf), "-oinchikey", "-f", "1", "-l", "1"],
        f"InChIKey generation for {sdf.name}",
    )
    pattern = re.compile(r"^[A-Z]{14}-[A-Z]{10}-[A-Z]$")
    for line in proc.stdout.splitlines():
        value = line.strip().split()[0] if line.strip() else ""
        if pattern.fullmatch(value):
            return value
    raise RuntimeError(f"No valid InChIKey produced for {sdf}")


def canonical_smiles(obabel: Path, sdf: Path) -> str:
    proc = run(
        [str(obabel), "-isdf", str(sdf), "-ocan", "-f", "1", "-l", "1"],
        f"canonical SMILES generation for {sdf.name}",
    )
    for line in proc.stdout.splitlines():
        if line.strip():
            return line.strip().split()[0]
    raise RuntimeError(f"No canonical SMILES produced for {sdf}")


def filename_key(row: dict[str, Any]) -> str:
    stem = Path(str(row.get("ligand_filename", ""))).stem
    for suffix in ("_stage2-best", "-stage2-best", "_gnina_minimized", "_gnina", "-best", "_best"):
        if stem.lower().endswith(suffix.lower()):
            stem = stem[: -len(suffix)]
            break
    return safe_name(stem).lower()


def identity_key(row: dict[str, Any], sdf: Path, obabel: Path, mode: str) -> str:
    if mode == "filename":
        return filename_key(row)
    if mode == "canonical_smiles":
        return canonical_smiles(obabel, sdf)
    key = inchikey(obabel, sdf)
    return key.split("-", 1)[0] if mode == "connectivity" else key


def convert_to_pdb(obabel: Path, source: Path, output: Path) -> None:
    fmt = source.suffix.lower().lstrip(".")
    if not fmt:
        raise RuntimeError(f"Cannot infer format from {source}")
    output.parent.mkdir(parents=True, exist_ok=True)
    run(
        [
            str(obabel), f"-i{fmt}", str(source), "-opdb", "-O", str(output),
            "-f", "1", "-l", "1",
        ],
        f"PDB conversion for {source.name}",
    )
    if not output.is_file() or output.stat().st_size == 0:
        raise RuntimeError(f"No PDB produced: {output}")


def prepare_receptor(receptor: Path, obabel: Path, output: Path) -> None:
    temporary = output.with_suffix(".tmp.pdb")
    if receptor.suffix.lower() == ".pdb":
        shutil.copy2(receptor, temporary)
    else:
        convert_to_pdb(obabel, receptor, temporary)
    cleaned = []
    for line in temporary.read_text(errors="replace").splitlines():
        if line[:6].strip().upper() in {"END", "ENDMDL", "CONECT", "MASTER", "MODEL"}:
            continue
        cleaned.append(line.rstrip())
    if not any(x.startswith(("ATOM  ", "HETATM")) for x in cleaned):
        raise RuntimeError("Prepared receptor contains no ATOM/HETATM records")
    output.write_text("\n".join(cleaned) + "\n")
    temporary.unlink(missing_ok=True)


def serial(line: str) -> int | None:
    try:
        return int(line[6:11].strip())
    except (ValueError, IndexError):
        return None


def ligand_atom_line(
    line: str, new_serial: int, resname: str, chain: str, resnum: int
) -> str:
    chars = list(line.rstrip("\n").ljust(80))
    chars[0:6] = list("HETATM")
    chars[6:11] = list(f"{new_serial:5d}")
    chars[16] = " "
    chars[17:20] = list(f"{resname:>3}"[-3:])
    chars[20] = " "
    chars[21] = chain
    chars[22:26] = list(f"{resnum:4d}")
    chars[26] = " "
    return "".join(chars).rstrip()


def make_complex(
    receptor_pdb: Path,
    ligand_pdb: Path,
    output: Path,
    resname: str,
    chain: str,
    resnum: int,
) -> None:
    rec_lines = receptor_pdb.read_text(errors="replace").splitlines()
    lig_lines = ligand_pdb.read_text(errors="replace").splitlines()
    rec_serials = [serial(x) for x in rec_lines if x.startswith(("ATOM  ", "HETATM"))]
    rec_max = max((x for x in rec_serials if x is not None), default=0)
    atoms = [x for x in lig_lines if x.startswith(("ATOM  ", "HETATM"))]
    if not atoms:
        raise RuntimeError(f"No ligand atoms found in {ligand_pdb}")

    mapping: dict[int, int] = {}
    new_atoms = []
    next_serial = rec_max + 2
    for line in atoms:
        old = serial(line)
        if old is None:
            raise RuntimeError(f"Invalid ligand atom serial in {ligand_pdb}")
        mapping[old] = next_serial
        new_atoms.append(ligand_atom_line(line, next_serial, resname, chain, resnum))
        next_serial += 1

    conect = []
    for line in lig_lines:
        if not line.startswith("CONECT"):
            continue
        old_numbers = [int(x) for x in re.findall(r"\d+", line[6:])]
        new_numbers = [mapping[x] for x in old_numbers if x in mapping]
        if len(new_numbers) >= 2:
            conect.append("CONECT" + "".join(f"{x:5d}" for x in new_numbers))

    result = [
        x for x in rec_lines
        if x[:6].strip().upper() not in {"END", "ENDMDL", "CONECT", "MASTER"}
    ]
    if not result or not result[-1].startswith("TER"):
        result.append(f"TER   {rec_max + 1:5d}")
    result.extend(new_atoms)
    result.extend(conect)
    result.append("END")
    output.write_text("\n".join(result) + "\n")


def write_csv(path: Path, columns: list[str], rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    root = Path(args.gnina_output_folder).expanduser().resolve()
    receptor = Path(args.receptor).expanduser().resolve()
    score_csv = root / args.score_file

    errors = []
    if not root.is_dir():
        errors.append(f"GNINA output folder not found: {root}")
    if not receptor.is_file():
        errors.append(f"Receptor not found: {receptor}")
    if not score_csv.is_file():
        errors.append(f"Score CSV not found: {score_csv}")
    if args.top < 1:
        errors.append("--top must be at least 1")
    resname = args.ligand_resname.strip().upper()
    chain = args.ligand_chain.strip()
    if not re.fullmatch(r"[A-Z0-9]{1,3}", resname):
        errors.append("--ligand-resname must contain 1-3 letters/numbers")
    if len(chain) != 1:
        errors.append("--ligand-chain must be exactly one character")
    if errors:
        raise SystemExit("INPUT ERROR\n" + "\n".join(errors))

    try:
        obabel = find_obabel(args.obabel)
        rows = read_rows(score_csv, args.score_column)
    except (FileNotFoundError, RuntimeError) as exc:
        raise SystemExit(str(exc)) from exc

    eligible = [
        row for row in rows
        if str(row.get("status", "")).strip() in SUCCESS_STATUSES
        and row["_primary"] is not None
    ]
    eligible.sort(key=rank_key)

    output_root = root / safe_name(args.output_name)
    complexes_dir = output_root / "complexes"
    ligand_dir = output_root / "ligand_pdb"
    sdf_dir = output_root / "selected_sdf"

    print("\nTop unique GNINA selection for PLIP")
    print("-----------------------------------")
    print(f"GNINA output:       {root}")
    print(f"Score CSV:          {score_csv}")
    print(f"Receptor:           {receptor}")
    print(f"Open Babel:         {obabel}")
    print(f"Eligible scored:    {len(eligible)}")
    print(f"Requested unique:   {args.top}")
    print(f"Ranking column:     {args.score_column} (high to low)")
    print(f"Duplicate key:      {args.dedup_key}")
    print(f"Output folder:      {output_root}")
    print(f"PLIP input folder:  {complexes_dir}")

    if args.dry_run:
        print("\nDry run completed. No files were selected or converted.")
        return

    if output_root.exists():
        if args.force:
            shutil.rmtree(output_root)
        else:
            raise SystemExit(f"Output already exists: {output_root}\nUse --force to rebuild it.")

    complexes_dir.mkdir(parents=True)
    ligand_dir.mkdir()
    sdf_dir.mkdir()
    receptor_pdb = output_root / "receptor_for_plip.pdb"
    prepare_receptor(receptor, obabel, receptor_pdb)

    selected: list[dict[str, Any]] = []
    duplicates: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    seen: dict[str, dict[str, Any]] = {}

    for position, row in enumerate(eligible, start=1):
        if len(selected) >= args.top:
            break
        ligand_name = str(row.get("ligand_filename", ""))
        result_name = safe_name(str(row.get("result_name", "")))
        sdf = sdf_path_for(row, root)
        if not sdf.is_file() or sdf.stat().st_size == 0:
            failures.append({
                "ranked_position": position,
                "ligand_filename": ligand_name,
                "result_name": result_name,
                "reason": "minimized SDF missing or empty",
                "sdf_path": str(sdf),
            })
            continue
        try:
            key = identity_key(row, sdf, obabel, args.dedup_key)
        except Exception as exc:
            failures.append({
                "ranked_position": position,
                "ligand_filename": ligand_name,
                "result_name": result_name,
                "reason": f"identity-key failure: {exc}",
                "sdf_path": str(sdf),
            })
            continue

        if key in seen:
            kept = seen[key]
            duplicates.append({
                "ranked_position": position,
                "ligand_filename": ligand_name,
                "result_name": result_name,
                "cnn_score": row.get("cnn_score", ""),
                "identity_key": key,
                "kept_selected_rank": kept["selected_rank"],
                "kept_ligand_filename": kept["ligand_filename"],
                "kept_cnn_score": kept["cnn_score"],
            })
            continue

        selected_rank = len(selected) + 1
        base = safe_name(f"rank{selected_rank:03d}__{result_name}")
        copied_sdf = sdf_dir / f"{base}.sdf"
        ligand_pdb = ligand_dir / f"{base}_ligand.pdb"
        complex_pdb = complexes_dir / f"{base}_complex.pdb"
        try:
            shutil.copy2(sdf, copied_sdf)
            convert_to_pdb(obabel, copied_sdf, ligand_pdb)
            make_complex(receptor_pdb, ligand_pdb, complex_pdb, resname, chain, args.ligand_resnum)
        except Exception as exc:
            copied_sdf.unlink(missing_ok=True)
            ligand_pdb.unlink(missing_ok=True)
            complex_pdb.unlink(missing_ok=True)
            failures.append({
                "ranked_position": position,
                "ligand_filename": ligand_name,
                "result_name": result_name,
                "reason": f"complex preparation failure: {exc}",
                "sdf_path": str(sdf),
            })
            continue

        record = {
            "selected_rank": selected_rank,
            "ranked_position_before_deduplication": position,
            "original_gnina_rank": row.get("cnn_score_rank", ""),
            "ligand_filename": ligand_name,
            "result_name": result_name,
            "cnn_score": row.get("cnn_score", ""),
            "cnn_affinity": row.get("cnn_affinity", ""),
            "minimized_affinity_kcal_mol": row.get("minimized_affinity_kcal_mol", ""),
            "cnn_vs": row.get("cnn_vs", ""),
            "cnn_affinity_variance": row.get("cnn_affinity_variance", ""),
            "dedup_method": args.dedup_key,
            "identity_key": key,
            "source_sdf": str(sdf),
            "selected_sdf": str(copied_sdf.resolve()),
            "ligand_pdb": str(ligand_pdb.resolve()),
            "complex_pdb": str(complex_pdb.resolve()),
            "plip_ligand_identifier": f"{resname}:{chain}:{args.ligand_resnum}",
        }
        selected.append(record)
        seen[key] = record
        print(f"[{selected_rank}/{args.top}] CNNscore={row.get('cnn_score', '')} {ligand_name}")

    write_csv(
        output_root / "selected_top_unique.csv",
        [
            "selected_rank", "ranked_position_before_deduplication", "original_gnina_rank",
            "ligand_filename", "result_name", "cnn_score", "cnn_affinity",
            "minimized_affinity_kcal_mol", "cnn_vs", "cnn_affinity_variance",
            "dedup_method", "identity_key", "source_sdf", "selected_sdf",
            "ligand_pdb", "complex_pdb", "plip_ligand_identifier",
        ],
        selected,
    )
    write_csv(
        output_root / "duplicates_skipped.csv",
        [
            "ranked_position", "ligand_filename", "result_name", "cnn_score",
            "identity_key", "kept_selected_rank", "kept_ligand_filename", "kept_cnn_score",
        ],
        duplicates,
    )
    write_csv(
        output_root / "failed_candidates.csv",
        ["ranked_position", "ligand_filename", "result_name", "reason", "sdf_path"],
        failures,
    )

    parameters = {
        "created": datetime.now().isoformat(timespec="seconds"),
        "gnina_output_folder": str(root),
        "score_csv": str(score_csv),
        "receptor_input": str(receptor),
        "receptor_for_plip": str(receptor_pdb),
        "obabel": str(obabel),
        "requested_unique": args.top,
        "selected_unique": len(selected),
        "eligible_scored_rows": len(eligible),
        "duplicates_skipped": len(duplicates),
        "failed_candidates": len(failures),
        "ranking": f"{args.score_column} descending; cnn_affinity descending; minimized affinity ascending",
        "dedup_key": args.dedup_key,
        "ligand_identifier": f"{resname}:{chain}:{args.ligand_resnum}",
        "plip_input_folder": str(complexes_dir),
    }
    (output_root / "run_parameters.json").write_text(json.dumps(parameters, indent=2) + "\n")

    readme = f"""PLIP-ready complexes are in:
{complexes_dir}

Selected unique molecules: {len(selected)}
Requested: {args.top}
Ranking: {args.score_column}, highest first
Duplicate definition: {args.dedup_key}
Ligand identifier: {resname}:{chain}:{args.ligand_resnum}

Single-complex example:
plip -f complexes/rank001__NAME_complex.pdb -o plip_result -x

Batch example (run from this folder):
mkdir -p plip_results
for complex in complexes/*_complex.pdb; do
    name=$(basename "$complex" _complex.pdb)
    mkdir -p "plip_results/$name"
    plip -f "$complex" -o "plip_results/$name" -x
done
"""
    (output_root / "README_PLIP.txt").write_text(readme)

    print("\nSelection completed")
    print("-------------------")
    print(f"Unique selected:     {len(selected)}")
    print(f"Duplicates skipped: {len(duplicates)}")
    print(f"Failed candidates:   {len(failures)}")
    print(f"Selection CSV:       {output_root / 'selected_top_unique.csv'}")
    print(f"PLIP complexes:      {complexes_dir}")

    if len(selected) < args.top:
        print(
            f"WARNING: only {len(selected)} unique molecules were successfully prepared; "
            f"{args.top} were requested.",
            file=sys.stderr,
        )
        sys.exit(2)


if __name__ == "__main__":
    main()
