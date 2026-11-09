#!/usr/bin/env python3
"""
Batch GNINA minimization and CNN rescoring for CD97 Stage-2 best poses.

Input
-----
A folder containing ligand files matching:
    *_stage2-best.pdbqt

Output
------
A subfolder created directly inside the input folder:
    GNINA_OUT/

GNINA_OUT contains:
    minimized_sdf/          one minimized/scored SDF per ligand
    logs/                   one GNINA log per ligand
    gnina_scores.csv        complete ranked score table
    failed_ligands.csv      failures only
    ligand_manifest.csv     input-to-output mapping
    run_parameters.json     reproducibility information
    run.log                 batch progress log

Default calculation
-------------------
For each already-positioned AutoDock-GPU pose, GNINA performs:
    --minimize
    --minimize_iters 1000
    --accurate_line
    --cnn_scoring rescore

This means the pose is locally energy-minimized with GNINA's empirical
scoring function and then evaluated with the default GNINA CNN ensemble.
No docking box is needed because this is minimization of an existing pose,
not a new global docking search.

The script is restart-safe. A ligand is skipped only when its existing SDF
contains usable minimizedAffinity, CNNscore, and CNNaffinity values.

Example
-------
python3 run_cd97_gnina_minimize_cnn.py \
    /path/to/Stage2_input_nrun50 \
    --dry-run

python3 run_cd97_gnina_minimize_cnn.py \
    /path/to/Stage2_input_nrun50
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
import time
from datetime import datetime
from pathlib import Path
from typing import Any


DEFAULT_RECEPTOR = Path(
    "/home/linux/Ayan_Computational/receptor/CD_receptor.pdbqt"
)

DEFAULT_PATTERN = "*_stage2-best.pdbqt"
OUTPUT_FOLDER_NAME = "GNINA_OUT"
DEFAULT_MINIMIZE_ITERS = 1000
DEFAULT_DEVICE = 0
DEFAULT_CPU = 1

REQUIRED_SCORE_FIELDS = (
    "minimizedAffinity",
    "CNNscore",
    "CNNaffinity",
)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Minimize CD97 Stage-2 best poses with GNINA and calculate "
            "CNNscore/CNNaffinity."
        )
    )

    parser.add_argument(
        "input_folder",
        help="Folder containing Stage-2 ligand PDBQT files.",
    )

    parser.add_argument(
        "--receptor",
        default=str(DEFAULT_RECEPTOR),
        help=f"Receptor file. Default: {DEFAULT_RECEPTOR}",
    )

    parser.add_argument(
        "--pattern",
        default=DEFAULT_PATTERN,
        help=(
            "Ligand filename pattern searched directly inside the input "
            f"folder. Default: {DEFAULT_PATTERN}"
        ),
    )

    parser.add_argument(
        "--gnina",
        default=None,
        help=(
            "GNINA executable path. If omitted, the script searches PATH "
            "and the active Conda environment."
        ),
    )

    parser.add_argument(
        "--device",
        type=int,
        default=DEFAULT_DEVICE,
        help=(
            "GNINA GPU device number. GNINA numbering begins at 0. "
            f"Default: {DEFAULT_DEVICE}"
        ),
    )

    parser.add_argument(
        "--cpu",
        type=int,
        default=DEFAULT_CPU,
        help=f"CPU threads given to GNINA. Default: {DEFAULT_CPU}",
    )

    parser.add_argument(
        "--minimize-iters",
        type=int,
        default=DEFAULT_MINIMIZE_ITERS,
        help=(
            "Maximum steepest-descent minimization iterations. "
            f"Default: {DEFAULT_MINIMIZE_ITERS}"
        ),
    )

    parser.add_argument(
        "--cnn-scoring",
        choices=("rescore", "refinement"),
        default="rescore",
        help=(
            "rescore: empirical minimization followed by CNN scoring "
            "(recommended here). refinement: CNN-guided refinement and "
            "scoring, substantially slower. Default: rescore"
        ),
    )

    parser.add_argument(
        "--force",
        action="store_true",
        help="Recalculate ligands even when valid output already exists.",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Validate all paths and show the command template without "
            "starting GNINA."
        ),
    )

    return parser.parse_args()


def find_gnina(explicit_path: str | None) -> Path:
    candidates: list[Path] = []

    if explicit_path:
        candidates.append(Path(explicit_path).expanduser())

    path_match = shutil.which("gnina")
    if path_match:
        candidates.append(Path(path_match))

    conda_prefix = os.environ.get("CONDA_PREFIX")
    if conda_prefix:
        candidates.append(Path(conda_prefix) / "bin" / "gnina")

    candidates.append(
        Path(
            "/media/linux/2c17f3da-5588-420c-a3a0-dab5c2b839ff/"
            "Home/anaconda3/envs/drugdesign_env/bin/gnina"
        )
    )

    checked: list[str] = []

    for candidate in candidates:
        resolved = candidate.expanduser()
        checked.append(str(resolved))

        if resolved.is_file() and os.access(resolved, os.X_OK):
            return resolved.resolve()

    checked_text = "\n".join(f"  {item}" for item in checked)

    raise FileNotFoundError(
        "\nGNINA executable was not found or is not executable.\n\n"
        "Locations checked:\n"
        f"{checked_text}\n\n"
        "Install GNINA in the active environment or provide:\n"
        "  --gnina /full/path/to/gnina"
    )


def safe_name(name: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", name).strip("._")
    return cleaned or "ligand"


def result_name(ligand: Path) -> str:
    stem = ligand.stem

    for suffix in ("_stage2-best", "-stage2-best", "_best", "-best"):
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
            break

    return safe_name(f"{stem}_gnina")


def validate_inputs(
    input_folder: Path,
    receptor: Path,
    ligand_pattern: str,
    device: int,
    cpu: int,
    minimize_iters: int,
) -> list[Path]:
    errors: list[str] = []

    if not input_folder.is_dir():
        errors.append(f"Input folder not found:\n  {input_folder}")

    if not receptor.is_file():
        errors.append(f"Receptor file not found:\n  {receptor}")

    if device < 0:
        errors.append("GNINA GPU device number cannot be negative.")

    if cpu < 1:
        errors.append("CPU thread count must be at least 1.")

    if minimize_iters < 1:
        errors.append("Minimization iterations must be at least 1.")

    ligands: list[Path] = []

    if input_folder.is_dir():
        ligands = sorted(input_folder.glob(ligand_pattern))

        if not ligands:
            errors.append(
                "No ligand files matched:\n"
                f"  {input_folder / ligand_pattern}"
            )

    duplicate_names: dict[str, Path] = {}
    duplicate_messages: list[str] = []

    for ligand in ligands:
        name = result_name(ligand)

        if name in duplicate_names:
            duplicate_messages.append(
                f"  {duplicate_names[name].name} and {ligand.name} -> {name}"
            )
        else:
            duplicate_names[name] = ligand

    if duplicate_messages:
        errors.append(
            "Duplicate output names would be produced:\n"
            + "\n".join(duplicate_messages)
        )

    if errors:
        print("\nINPUT ERROR", file=sys.stderr)
        print("-----------", file=sys.stderr)

        for error in errors:
            print(f"\n{error}", file=sys.stderr)

        sys.exit(1)

    return ligands


def parse_sdf_properties(sdf_path: Path) -> dict[str, str]:
    """
    Read properties from the first SDF record.

    GNINA normally writes one minimized pose for one input pose. The parser
    stops at the first '$$$$' record separator.
    """
    properties: dict[str, str] = {}
    lines = sdf_path.read_text(errors="replace").splitlines()
    index = 0

    while index < len(lines):
        line = lines[index]

        if line.strip() == "$$$$":
            break

        match = re.match(r"^>\s*<([^>]+)>", line.strip())

        if not match:
            index += 1
            continue

        key = match.group(1).strip()
        index += 1
        value_lines: list[str] = []

        while index < len(lines):
            current = lines[index]

            if current.strip() == "" or current.strip() == "$$$$":
                break

            value_lines.append(current.strip())
            index += 1

        properties[key] = "\n".join(value_lines).strip()
        index += 1

    return properties


def parse_log_scores(log_path: Path) -> dict[str, str]:
    """Fallback parser if an SDF property is absent."""
    if not log_path.is_file():
        return {}

    text = log_path.read_text(errors="replace")
    patterns = {
        "minimizedAffinity": (
            r"(?:Affinity|minimizedAffinity)\s*[:=]?\s*"
            r"([+-]?(?:\d+(?:\.\d*)?|\.\d+))"
        ),
        "CNNscore": (
            r"CNNscore\s*[:=]?\s*"
            r"([+-]?(?:\d+(?:\.\d*)?|\.\d+))"
        ),
        "CNNaffinity": (
            r"CNNaffinity\s*[:=]?\s*"
            r"([+-]?(?:\d+(?:\.\d*)?|\.\d+))"
        ),
        "CNN_VS": (
            r"CNN_VS\s*[:=]?\s*"
            r"([+-]?(?:\d+(?:\.\d*)?|\.\d+))"
        ),
        "CNNaffinity_variance": (
            r"(?:CNNaffinity_variance|CNNvariance)\s*[:=]?\s*"
            r"([+-]?(?:\d+(?:\.\d*)?|\.\d+))"
        ),
    }

    values: dict[str, str] = {}

    for key, pattern in patterns.items():
        matches = re.findall(pattern, text, flags=re.IGNORECASE)

        if matches:
            values[key] = matches[-1]

    return values


def float_or_none(value: Any) -> float | None:
    if value is None:
        return None

    try:
        return float(str(value).strip())
    except (TypeError, ValueError):
        return None


def extract_scores(
    sdf_path: Path,
    ligand_log_path: Path,
) -> dict[str, float | None]:
    properties: dict[str, str] = {}

    if sdf_path.is_file() and sdf_path.stat().st_size > 0:
        properties.update(parse_sdf_properties(sdf_path))

    fallback = parse_log_scores(ligand_log_path)

    for key, value in fallback.items():
        properties.setdefault(key, value)

    return {
        "minimized_affinity_kcal_mol": float_or_none(
            properties.get("minimizedAffinity")
        ),
        "cnn_score": float_or_none(properties.get("CNNscore")),
        "cnn_affinity": float_or_none(properties.get("CNNaffinity")),
        "cnn_vs": float_or_none(properties.get("CNN_VS")),
        "cnn_affinity_variance": float_or_none(
            properties.get("CNNaffinity_variance")
            or properties.get("CNNvariance")
        ),
    }


def scores_are_complete(scores: dict[str, float | None]) -> bool:
    return (
        scores["minimized_affinity_kcal_mol"] is not None
        and scores["cnn_score"] is not None
        and scores["cnn_affinity"] is not None
    )


def gnina_version(gnina: Path) -> str:
    try:
        process = subprocess.run(
            [str(gnina), "--version"],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=30,
            check=False,
        )
        version_text = process.stdout.strip()
        return version_text or f"return_code={process.returncode}"
    except Exception as error:
        return f"version_check_failed: {error}"


def build_command(
    gnina: Path,
    receptor: Path,
    ligand: Path,
    output_sdf: Path,
    ligand_log: Path,
    device: int,
    cpu: int,
    minimize_iters: int,
    cnn_scoring: str,
) -> list[str]:
    return [
        str(gnina),
        "--receptor",
        str(receptor),
        "--ligand",
        str(ligand),
        "--minimize",
        "--minimize_iters",
        str(minimize_iters),
        "--accurate_line",
        "--cnn_scoring",
        cnn_scoring,
        "--pose_sort_order",
        "CNNscore",
        "--device",
        str(device),
        "--cpu",
        str(cpu),
        "--out",
        str(output_sdf),
        "--log",
        str(ligand_log),
    ]


def write_manifest(
    manifest_path: Path,
    ligands: list[Path],
    minimized_dir: Path,
    logs_dir: Path,
) -> None:
    with manifest_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "ligand_filename",
                "ligand_full_path",
                "result_name",
                "output_sdf",
                "gnina_log",
            ]
        )

        for ligand in ligands:
            name = result_name(ligand)
            writer.writerow(
                [
                    ligand.name,
                    str(ligand.resolve()),
                    name,
                    str((minimized_dir / f"{name}_minimized.sdf").resolve()),
                    str((logs_dir / f"{name}.log").resolve()),
                ]
            )


def write_score_table(
    score_path: Path,
    records: list[dict[str, Any]],
) -> None:
    successful = [
        record
        for record in records
        if record["status"] in {"completed", "skipped_existing"}
    ]

    successful.sort(
        key=lambda record: (
            -(
                record["cnn_score"]
                if record["cnn_score"] is not None
                else float("-inf")
            ),
            -(
                record["cnn_affinity"]
                if record["cnn_affinity"] is not None
                else float("-inf")
            ),
            (
                record["minimized_affinity_kcal_mol"]
                if record["minimized_affinity_kcal_mol"] is not None
                else float("inf")
            ),
            record["ligand_filename"],
        )
    )

    failed = [
        record
        for record in records
        if record["status"] not in {"completed", "skipped_existing"}
    ]

    ranked_records: list[dict[str, Any]] = []

    for rank, record in enumerate(successful, start=1):
        updated = dict(record)
        updated["cnn_score_rank"] = rank
        ranked_records.append(updated)

    for record in failed:
        updated = dict(record)
        updated["cnn_score_rank"] = ""
        ranked_records.append(updated)

    columns = [
        "cnn_score_rank",
        "ligand_filename",
        "result_name",
        "status",
        "minimized_affinity_kcal_mol",
        "cnn_score",
        "cnn_affinity",
        "cnn_vs",
        "cnn_affinity_variance",
        "runtime_seconds",
        "input_pdbqt",
        "output_sdf",
        "gnina_log",
        "error",
    ]

    with score_path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=columns,
            extrasaction="ignore",
        )
        writer.writeheader()
        writer.writerows(ranked_records)


def write_failure_table(
    failure_path: Path,
    records: list[dict[str, Any]],
) -> None:
    failures = [
        record
        for record in records
        if record["status"] not in {"completed", "skipped_existing"}
    ]

    with failure_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "ligand_filename",
                "result_name",
                "status",
                "error",
                "input_pdbqt",
                "gnina_log",
            ]
        )

        for record in failures:
            writer.writerow(
                [
                    record["ligand_filename"],
                    record["result_name"],
                    record["status"],
                    record["error"],
                    record["input_pdbqt"],
                    record["gnina_log"],
                ]
            )


def main() -> None:
    args = parse_arguments()

    input_folder = Path(args.input_folder).expanduser().resolve()
    receptor = Path(args.receptor).expanduser().resolve()

    try:
        gnina = find_gnina(args.gnina)
    except FileNotFoundError as error:
        print(error, file=sys.stderr)
        sys.exit(1)

    ligands = validate_inputs(
        input_folder=input_folder,
        receptor=receptor,
        ligand_pattern=args.pattern,
        device=args.device,
        cpu=args.cpu,
        minimize_iters=args.minimize_iters,
    )

    output_root = input_folder / OUTPUT_FOLDER_NAME
    minimized_dir = output_root / "minimized_sdf"
    logs_dir = output_root / "logs"

    output_root.mkdir(parents=True, exist_ok=True)
    minimized_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = output_root / "ligand_manifest.csv"
    score_path = output_root / "gnina_scores.csv"
    failure_path = output_root / "failed_ligands.csv"
    parameter_path = output_root / "run_parameters.json"
    batch_log_path = output_root / "run.log"

    write_manifest(
        manifest_path=manifest_path,
        ligands=ligands,
        minimized_dir=minimized_dir,
        logs_dir=logs_dir,
    )

    version_text = gnina_version(gnina)

    example_name = result_name(ligands[0])
    example_command = build_command(
        gnina=gnina,
        receptor=receptor,
        ligand=ligands[0],
        output_sdf=minimized_dir / f"{example_name}_minimized.sdf",
        ligand_log=logs_dir / f"{example_name}.log",
        device=args.device,
        cpu=args.cpu,
        minimize_iters=args.minimize_iters,
        cnn_scoring=args.cnn_scoring,
    )

    parameters = {
        "prepared": datetime.now().isoformat(timespec="seconds"),
        "conda_environment": os.environ.get(
            "CONDA_DEFAULT_ENV",
            "unknown",
        ),
        "conda_prefix": os.environ.get("CONDA_PREFIX", "unknown"),
        "gnina_executable": str(gnina),
        "gnina_version": version_text,
        "receptor": str(receptor),
        "input_folder": str(input_folder),
        "ligand_pattern": args.pattern,
        "total_ligands": len(ligands),
        "output_folder": str(output_root),
        "minimize": True,
        "minimize_iters": args.minimize_iters,
        "accurate_line": True,
        "cnn_scoring": args.cnn_scoring,
        "cnn_model": "GNINA default ensemble",
        "pose_sort_order": "CNNscore",
        "gpu_device": args.device,
        "cpu_threads": args.cpu,
        "restart_safe": True,
        "force": args.force,
        "example_command": example_command,
    }

    parameter_path.write_text(json.dumps(parameters, indent=2) + "\n")

    previously_complete = 0

    for ligand in ligands:
        name = result_name(ligand)
        output_sdf = minimized_dir / f"{name}_minimized.sdf"
        ligand_log = logs_dir / f"{name}.log"
        scores = extract_scores(output_sdf, ligand_log)

        if scores_are_complete(scores):
            previously_complete += 1

    print("\nCD97 GNINA minimization and CNN scoring")
    print("---------------------------------------")
    print(f"GNINA executable:   {gnina}")
    print(f"GNINA version:      {version_text.splitlines()[0]}")
    print(f"Receptor:           {receptor}")
    print(f"Input folder:       {input_folder}")
    print(f"Ligand pattern:     {args.pattern}")
    print(f"Total ligands:      {len(ligands)}")
    print(f"Valid existing:     {previously_complete}")
    print(f"Pending:            {len(ligands) - previously_complete}")
    print(f"Minimize iterations:{args.minimize_iters}")
    print(f"CNN scoring:        {args.cnn_scoring}")
    print(f"GNINA GPU device:   {args.device}")
    print(f"Output folder:      {output_root}")
    print("\nExample command:")
    print(" ".join(shlex.quote(part) for part in example_command))

    if args.dry_run:
        print("\nDry run completed. GNINA was not started.")
        return

    records: list[dict[str, Any]] = []
    completed_count = 0
    skipped_count = 0
    failed_count = 0

    with batch_log_path.open("a", buffering=1) as batch_log:
        batch_log.write("\n" + "=" * 80 + "\n")
        batch_log.write(
            f"Batch started: "
            f"{datetime.now().isoformat(timespec='seconds')}\n"
        )
        batch_log.write(f"GNINA: {gnina}\n")
        batch_log.write(f"Input: {input_folder}\n")
        batch_log.write(f"Ligands: {len(ligands)}\n")
        batch_log.write("=" * 80 + "\n")
        batch_log.flush()

        for index, ligand in enumerate(ligands, start=1):
            name = result_name(ligand)
            output_sdf = minimized_dir / f"{name}_minimized.sdf"
            ligand_log = logs_dir / f"{name}.log"

            base_record: dict[str, Any] = {
                "ligand_filename": ligand.name,
                "result_name": name,
                "status": "",
                "minimized_affinity_kcal_mol": None,
                "cnn_score": None,
                "cnn_affinity": None,
                "cnn_vs": None,
                "cnn_affinity_variance": None,
                "runtime_seconds": 0.0,
                "input_pdbqt": str(ligand.resolve()),
                "output_sdf": str(output_sdf.resolve()),
                "gnina_log": str(ligand_log.resolve()),
                "error": "",
            }

            existing_scores = extract_scores(output_sdf, ligand_log)

            if not args.force and scores_are_complete(existing_scores):
                base_record.update(existing_scores)
                base_record["status"] = "skipped_existing"
                records.append(base_record)
                skipped_count += 1

                print(
                    f"[{index}/{len(ligands)}] SKIP "
                    f"{ligand.name}"
                )
                continue

            command = build_command(
                gnina=gnina,
                receptor=receptor,
                ligand=ligand,
                output_sdf=output_sdf,
                ligand_log=ligand_log,
                device=args.device,
                cpu=args.cpu,
                minimize_iters=args.minimize_iters,
                cnn_scoring=args.cnn_scoring,
            )

            print(
                f"[{index}/{len(ligands)}] RUN  "
                f"{ligand.name}"
            )

            batch_log.write(
                f"\n[{index}/{len(ligands)}] "
                f"{datetime.now().isoformat(timespec='seconds')} "
                f"{ligand.name}\n"
            )
            batch_log.write(
                "Command: "
                + " ".join(shlex.quote(part) for part in command)
                + "\n"
            )
            batch_log.flush()

            start_time = time.monotonic()

            try:
                process = subprocess.run(
                    command,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    check=False,
                )
                runtime_seconds = time.monotonic() - start_time
                base_record["runtime_seconds"] = round(
                    runtime_seconds,
                    3,
                )

                batch_log.write(process.stdout)
                batch_log.write(
                    f"\nReturn code: {process.returncode}\n"
                )
                batch_log.flush()

                if process.returncode != 0:
                    base_record["status"] = "failed_gnina"
                    base_record["error"] = (
                        f"GNINA return code {process.returncode}"
                    )
                    records.append(base_record)
                    failed_count += 1
                    continue

                scores = extract_scores(output_sdf, ligand_log)
                base_record.update(scores)

                if not scores_are_complete(scores):
                    base_record["status"] = "failed_missing_scores"
                    base_record["error"] = (
                        "GNINA finished, but minimizedAffinity, "
                        "CNNscore, or CNNaffinity could not be parsed."
                    )
                    records.append(base_record)
                    failed_count += 1
                    continue

                base_record["status"] = "completed"
                records.append(base_record)
                completed_count += 1

            except Exception as error:
                runtime_seconds = time.monotonic() - start_time
                base_record["runtime_seconds"] = round(
                    runtime_seconds,
                    3,
                )
                base_record["status"] = "failed_exception"
                base_record["error"] = str(error)
                records.append(base_record)
                failed_count += 1

                batch_log.write(
                    f"Python exception: {error}\n"
                )
                batch_log.flush()

            # Keep the tables current after every ligand so an interrupted
            # run still has a usable partial summary.
            write_score_table(score_path, records)
            write_failure_table(failure_path, records)

    write_score_table(score_path, records)
    write_failure_table(failure_path, records)

    parameters["finished"] = datetime.now().isoformat(
        timespec="seconds"
    )
    parameters["completed_this_run"] = completed_count
    parameters["skipped_existing"] = skipped_count
    parameters["failed"] = failed_count
    parameter_path.write_text(json.dumps(parameters, indent=2) + "\n")

    print("\nGNINA batch finished")
    print("--------------------")
    print(f"Completed this run: {completed_count}")
    print(f"Skipped existing:  {skipped_count}")
    print(f"Failed:            {failed_count}")
    print(f"Ranked scores:     {score_path}")
    print(f"Minimized poses:   {minimized_dir}")
    print(f"Failure table:     {failure_path}")
    print(f"Batch log:         {batch_log_path}")

    if failed_count:
        sys.exit(2)


if __name__ == "__main__":
    main()
