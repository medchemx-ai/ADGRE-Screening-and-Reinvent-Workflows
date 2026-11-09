#!/usr/bin/env python3
"""
CD97 Stage-2 virtual screening with AutoDock-GPU.

Purpose
-------
- Input: a Stage-1-selected folder containing ligand files ending in
  "-best.pdbqt".
- Receptor/grid maps: fixed CD97 AutoGrid files in
  /home/linux/Ayan_Computational/receptor
- Stage-2 docking: 50 independent LGA runs per ligand.
- Output: created INSIDE the input folder as:
      <input_folder_name>_nrun50
- Pose export: after docking, the script saves every available final pose
  when there are 20 or fewer; when there are more than 20, it saves only
  the 20 lowest-energy poses.

Important
---------
Do NOT use "--npdb 20" for this purpose. In AutoDock-GPU, --npdb 20
requests top population poses from EACH LGA run and can therefore create
far more than 20 files per ligand. This script instead ranks the final
run poses present in each DLG. It retains all poses when the count is
1-20 and retains only the best 20 when the count is greater than 20.

Example
-------
conda activate drugdesign_env

python3 run_cd97_stage2_autodock_gpu.py \
  /home/linux/Ayan_Computational/dockresult/.../Stage2_input \
  --dry-run

python3 run_cd97_stage2_autodock_gpu.py \
  /home/linux/Ayan_Computational/dockresult/.../Stage2_input

The script is restart-safe: ligands with existing non-empty DLG and XML
files are skipped, and missing top-20 pose folders are regenerated.
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
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


# ============================================================
# FIXED CD97 / AUTODOCK-GPU SETTINGS
# ============================================================

AUTODOCK_GPU = Path(
    "/media/linux/2c17f3da-5588-420c-a3a0-dab5c2b839ff/"
    "Home/anaconda3/envs/drugdesign_env/bin/autodock_gpu_128wi"
)

RECEPTOR_DIR = Path("/home/linux/Ayan_Computational/receptor")
PREFERRED_GRID_FLD = RECEPTOR_DIR / "CD_receptor.maps.fld"

N_RUNS = 50
TOP_POSES = 20
DEFAULT_DEVICE = 1

# More generous Stage-2 search ceiling than the AutoDock-GPU default.
# Heuristics still chooses a ligand-dependent evaluation count, while
# AutoStop terminates converged searches.
HEURMAX = 25_000_000
STOPSTD = 0.10

ENERGY_RE = re.compile(
    r"Estimated Free Energy of Binding\s*=\s*"
    r"([+-]?(?:\d+(?:\.\d*)?|\.\d+))",
    re.IGNORECASE,
)
RUN_RE = re.compile(r"\bRun\s*=\s*(\d+)", re.IGNORECASE)


@dataclass
class DockedPose:
    model_number: int
    run_number: int | None
    energy: float
    pdbqt_lines: list[str]


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run CD97 Stage-2 AutoDock-GPU docking with nrun=50 and "
            "export the 20 lowest-energy final poses per ligand."
        )
    )
    parser.add_argument(
        "ligand_folder",
        help=(
            "Stage-2 input folder containing Stage-1 ligand files "
            "ending in '-best.pdbqt'."
        ),
    )
    parser.add_argument(
        "--output-name",
        default=None,
        help=(
            "Optional name of the output subfolder created inside the "
            "input folder. Default: <input_folder_name>_nrun50."
        ),
    )
    parser.add_argument(
        "--device",
        type=int,
        default=DEFAULT_DEVICE,
        help=(
            "AutoDock-GPU device number; counting begins at 1. "
            f"Default: {DEFAULT_DEVICE}."
        ),
    )
    parser.add_argument(
        "--contact-analysis",
        action="store_true",
        help="Also enable AutoDock-GPU distance-based contact analysis.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Validate paths and create the output folder, file list, "
            "manifest, and run-parameter file without starting docking."
        ),
    )
    parser.add_argument(
        "--extract-only",
        action="store_true",
        help=(
            "Do not start docking. Extract/rebuild top-20 poses from "
            "existing DLG files in the Stage-2 output folder."
        ),
    )
    return parser.parse_args()


def locate_grid_fld() -> Path:
    if PREFERRED_GRID_FLD.is_file():
        return PREFERRED_GRID_FLD.resolve()

    candidates = sorted(RECEPTOR_DIR.glob("*.maps.fld"))

    if len(candidates) == 1:
        return candidates[0].resolve()

    if not candidates:
        raise FileNotFoundError(
            "\nNo AutoGrid .maps.fld file was found.\n\n"
            f"Expected inside:\n  {RECEPTOR_DIR}\n\n"
            "The receptor PDBQT alone is not sufficient for AutoDock-GPU."
        )

    names = "\n".join(f"  {path}" for path in candidates)
    raise RuntimeError(
        "\nMore than one .maps.fld file was found. Keep the required file as:\n"
        f"  {PREFERRED_GRID_FLD}\n\n"
        f"Files found:\n{names}"
    )


def safe_output_name(name: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", name).strip("._")
    return cleaned or "ligand"


def result_name_for_ligand(ligand: Path) -> str:
    stem = ligand.stem

    for suffix in ("-best", "_best"):
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
            break

    return safe_output_name(f"{stem}_stage2")


def find_stage2_ligands(ligand_folder: Path) -> list[Path]:
    # The current Stage-2 input contains Stage-1 selected best poses.
    ligands = sorted(ligand_folder.glob("*-best.pdbqt"))

    if not ligands:
        raise FileNotFoundError(
            "\nNo Stage-1 best-pose ligand files were found.\n\n"
            f"Expected files matching:\n  {ligand_folder}/*-best.pdbqt"
        )

    return ligands


def validate_inputs(
    ligand_folder: Path,
    device: int,
) -> list[Path]:
    errors: list[str] = []

    if not AUTODOCK_GPU.is_file():
        errors.append(
            "AutoDock-GPU executable was not found:\n"
            f"  {AUTODOCK_GPU}"
        )
    elif not os.access(AUTODOCK_GPU, os.X_OK):
        errors.append(
            "AutoDock-GPU exists but is not executable:\n"
            f"  {AUTODOCK_GPU}\n"
            f"Run:\n  chmod +x {shlex.quote(str(AUTODOCK_GPU))}"
        )

    if not ligand_folder.is_dir():
        errors.append(
            "Stage-2 ligand input folder was not found:\n"
            f"  {ligand_folder}"
        )

    if device < 1:
        errors.append("GPU device number must be at least 1.")

    ligands: list[Path] = []

    if ligand_folder.is_dir():
        try:
            ligands = find_stage2_ligands(ligand_folder)
        except FileNotFoundError as error:
            errors.append(str(error))

    output_names: dict[str, Path] = {}
    duplicate_messages: list[str] = []

    for ligand in ligands:
        result_name = result_name_for_ligand(ligand)

        if result_name in output_names:
            duplicate_messages.append(
                f"  {output_names[result_name].name} and "
                f"{ligand.name} -> {result_name}"
            )
        else:
            output_names[result_name] = ligand

    if duplicate_messages:
        errors.append(
            "Some ligand names would produce duplicate Stage-2 result names:\n"
            + "\n".join(duplicate_messages)
        )

    if errors:
        print("\nINPUT ERROR", file=sys.stderr)
        print("-----------", file=sys.stderr)

        for error in errors:
            print(f"\n{error}", file=sys.stderr)

        sys.exit(1)

    return ligands


def is_docking_complete(output_directory: Path, result_name: str) -> bool:
    dlg = output_directory / f"{result_name}.dlg"
    xml = output_directory / f"{result_name}.xml"

    return (
        dlg.is_file()
        and dlg.stat().st_size > 0
        and xml.is_file()
        and xml.stat().st_size > 0
    )


def pose_directory_is_complete(
    top_pose_root: Path,
    result_name: str,
) -> bool:
    pose_directory = top_pose_root / result_name

    if not pose_directory.is_dir():
        return False

    pose_files = sorted(pose_directory.glob("*_pose*.pdbqt"))
    return len(pose_files) == TOP_POSES


def write_manifest(
    manifest_path: Path,
    ligands: list[Path],
    output_directory: Path,
    top_pose_root: Path,
) -> None:
    with manifest_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "ligand_filename",
                "ligand_full_path",
                "result_name",
                "docking_status_before_run",
                "top20_status_before_run",
            ]
        )

        for ligand in ligands:
            result_name = result_name_for_ligand(ligand)
            docking_status = (
                "completed"
                if is_docking_complete(output_directory, result_name)
                else "pending"
            )
            pose_status = (
                "completed"
                if pose_directory_is_complete(top_pose_root, result_name)
                else "pending"
            )

            writer.writerow(
                [
                    ligand.name,
                    str(ligand.resolve()),
                    result_name,
                    docking_status,
                    pose_status,
                ]
            )


def finalize_pose_block(
    model_number: int,
    block_lines: list[str],
) -> DockedPose | None:
    energy: float | None = None
    run_number: int | None = None

    for line in block_lines:
        if energy is None:
            energy_match = ENERGY_RE.search(line)
            if energy_match:
                energy = float(energy_match.group(1))

        if run_number is None:
            run_match = RUN_RE.search(line)
            if run_match:
                run_number = int(run_match.group(1))

    if energy is None:
        return None

    return DockedPose(
        model_number=model_number,
        run_number=run_number,
        energy=energy,
        pdbqt_lines=block_lines.copy(),
    )


def parse_docked_poses(dlg_path: Path) -> list[DockedPose]:
    poses: list[DockedPose] = []
    current_lines: list[str] = []
    in_model = False
    model_number = 0

    with dlg_path.open("r", errors="replace") as handle:
        for raw_line in handle:
            if not raw_line.startswith("DOCKED:"):
                continue

            line = raw_line[len("DOCKED:"):].lstrip().rstrip("\n")

            if line.startswith("MODEL"):
                if in_model and current_lines:
                    pose = finalize_pose_block(model_number, current_lines)
                    if pose is not None:
                        poses.append(pose)

                in_model = True
                current_lines = [line]
                model_match = re.search(r"\bMODEL\s+(\d+)", line)
                model_number = (
                    int(model_match.group(1))
                    if model_match
                    else len(poses) + 1
                )
                continue

            if not in_model:
                continue

            current_lines.append(line)

            if line.startswith("ENDMDL"):
                pose = finalize_pose_block(model_number, current_lines)
                if pose is not None:
                    poses.append(pose)

                current_lines = []
                in_model = False

    if in_model and current_lines:
        pose = finalize_pose_block(model_number, current_lines)
        if pose is not None:
            poses.append(pose)

    # Keep only the best-energy occurrence for each explicit run number.
    # This protects against unusual DLG files that repeat a run block.
    best_by_run: dict[int, DockedPose] = {}
    poses_without_run: list[DockedPose] = []

    for pose in poses:
        if pose.run_number is None:
            poses_without_run.append(pose)
            continue

        previous = best_by_run.get(pose.run_number)
        if previous is None or pose.energy < previous.energy:
            best_by_run[pose.run_number] = pose

    unique_poses = list(best_by_run.values()) + poses_without_run
    unique_poses.sort(
        key=lambda pose: (
            pose.energy,
            pose.run_number if pose.run_number is not None else 10**9,
            pose.model_number,
        )
    )

    return unique_poses


def normalized_pose_lines(
    pose: DockedPose,
    rank: int,
) -> list[str]:
    body: list[str] = []

    for line in pose.pdbqt_lines:
        if line.startswith("MODEL") or line.startswith("ENDMDL"):
            continue
        body.append(line)

    run_text = (
        str(pose.run_number)
        if pose.run_number is not None
        else "unknown"
    )

    output = [
        "MODEL        1",
        f"REMARK STAGE2_RANK {rank}",
        f"REMARK AUTODOCK_RUN {run_text}",
        f"REMARK BINDING_ENERGY_KCAL_MOL {pose.energy:.4f}",
    ]
    output.extend(body)
    output.append("ENDMDL")
    return output


def extract_top_poses_from_dlg(
    dlg_path: Path,
    result_name: str,
    top_pose_root: Path,
) -> list[dict[str, object]]:
    poses = parse_docked_poses(dlg_path)

    available_pose_count = len(poses)

    if available_pose_count == 0:
        raise RuntimeError(
            f"{dlg_path.name}: no energy-labelled final poses "
            "could be extracted."
        )

    # Poses are already sorted from lowest (best) to highest energy.
    # Save every available pose when there are 20 or fewer.
    # When there are more than 20, save only the best 20.
    selected = poses[:TOP_POSES]
    exported_pose_count = len(selected)
    extraction_status = (
        "complete"
        if exported_pose_count == TOP_POSES
        else "incomplete"
    )
    final_directory = top_pose_root / result_name
    temporary_directory = top_pose_root / f".{result_name}.tmp"

    if temporary_directory.exists():
        shutil.rmtree(temporary_directory)

    temporary_directory.mkdir(parents=True, exist_ok=False)
    records: list[dict[str, object]] = []
    combined_lines: list[str] = []

    for rank, pose in enumerate(selected, start=1):
        run_text = (
            f"{pose.run_number:02d}"
            if pose.run_number is not None
            else "NA"
        )
        energy_text = f"{pose.energy:.2f}".replace("+", "")
        pose_filename = (
            f"{result_name}_pose{rank:02d}_run{run_text}_"
            f"E{energy_text}.pdbqt"
        )
        pose_path = temporary_directory / pose_filename
        lines = normalized_pose_lines(pose, rank)
        lines.insert(1, f"REMARK EXTRACTION_STATUS {extraction_status}")
        lines.insert(
            2,
            f"REMARK AVAILABLE_FINAL_POSES {available_pose_count}",
        )
        lines.insert(
            3,
            f"REMARK EXPORTED_POSES {exported_pose_count}",
        )
        lines.insert(
            4,
            f"REMARK MAXIMUM_POSES_REQUESTED {TOP_POSES}",
        )
        pose_path.write_text("\n".join(lines) + "\n")

        combined_lines.append(f"MODEL        {rank}")
        combined_lines.append(
            f"REMARK EXTRACTION_STATUS {extraction_status}"
        )
        combined_lines.append(
            f"REMARK AVAILABLE_FINAL_POSES {available_pose_count}"
        )
        combined_lines.append(
            f"REMARK EXPORTED_POSES {exported_pose_count}"
        )
        combined_lines.append(
            f"REMARK MAXIMUM_POSES_REQUESTED {TOP_POSES}"
        )
        combined_lines.append(f"REMARK STAGE2_RANK {rank}")
        combined_lines.append(
            f"REMARK AUTODOCK_RUN "
            f"{pose.run_number if pose.run_number is not None else 'unknown'}"
        )
        combined_lines.append(
            f"REMARK BINDING_ENERGY_KCAL_MOL {pose.energy:.4f}"
        )

        for line in pose.pdbqt_lines:
            if line.startswith("MODEL") or line.startswith("ENDMDL"):
                continue
            combined_lines.append(line)

        combined_lines.append("ENDMDL")

        records.append(
            {
                "result_name": result_name,
                "extraction_status": extraction_status,
                "available_pose_count": available_pose_count,
                "exported_pose_count": exported_pose_count,
                "maximum_pose_count": TOP_POSES,
                "pose_rank": rank,
                "run_number": pose.run_number,
                "binding_energy_kcal_mol": pose.energy,
                "pose_file": str(
                    (final_directory / pose_filename).resolve()
                ),
            }
        )

    combined_path = temporary_directory / f"{result_name}_top20.pdbqt"
    combined_path.write_text("\n".join(combined_lines) + "\n")

    if final_directory.exists():
        shutil.rmtree(final_directory)

    temporary_directory.rename(final_directory)
    return records


def extract_all_top_poses(
    ligands: list[Path],
    output_directory: Path,
    top_pose_root: Path,
) -> tuple[int, int]:
    top_pose_root.mkdir(parents=True, exist_ok=True)
    all_records: list[dict[str, object]] = []
    success_count = 0
    failure_count = 0
    failure_rows: list[tuple[str, str]] = []

    for ligand in ligands:
        result_name = result_name_for_ligand(ligand)
        dlg_path = output_directory / f"{result_name}.dlg"

        if not dlg_path.is_file() or dlg_path.stat().st_size == 0:
            failure_count += 1
            failure_rows.append((result_name, "DLG file missing or empty"))
            continue

        try:
            records = extract_top_poses_from_dlg(
                dlg_path=dlg_path,
                result_name=result_name,
                top_pose_root=top_pose_root,
            )
            all_records.extend(records)
            success_count += 1
        except Exception as error:
            failure_count += 1
            failure_rows.append((result_name, str(error)))

    summary_path = top_pose_root / "top20_pose_summary.csv"

    with summary_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "result_name",
                "extraction_status",
                "available_pose_count",
                "exported_pose_count",
                "maximum_pose_count",
                "pose_rank",
                "run_number",
                "binding_energy_kcal_mol",
                "pose_file",
            ]
        )

        for record in all_records:
            writer.writerow(
                [
                    record["result_name"],
                    record["extraction_status"],
                    record["available_pose_count"],
                    record["exported_pose_count"],
                    record["maximum_pose_count"],
                    record["pose_rank"],
                    record["run_number"],
                    record["binding_energy_kcal_mol"],
                    record["pose_file"],
                ]
            )

    failure_path = top_pose_root / "top20_extraction_failures.csv"

    with failure_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["result_name", "reason"])
        writer.writerows(failure_rows)

    return success_count, failure_count


def main() -> None:
    args = parse_arguments()
    ligand_folder = Path(args.ligand_folder).expanduser().resolve()

    ligands = validate_inputs(
        ligand_folder=ligand_folder,
        device=args.device,
    )

    try:
        grid_fld = locate_grid_fld()
    except (FileNotFoundError, RuntimeError) as error:
        print(error, file=sys.stderr)
        sys.exit(1)

    output_name = (
        safe_output_name(args.output_name)
        if args.output_name
        else f"{ligand_folder.name}_nrun{N_RUNS}"
    )
    output_directory = ligand_folder / output_name
    output_directory.mkdir(parents=True, exist_ok=True)

    top_pose_root = output_directory / "top20_conformers"
    manifest_path = output_directory / "ligand_manifest.csv"

    write_manifest(
        manifest_path=manifest_path,
        ligands=ligands,
        output_directory=output_directory,
        top_pose_root=top_pose_root,
    )

    pending_ligands: list[Path] = []
    skipped_count = 0

    for ligand in ligands:
        result_name = result_name_for_ligand(ligand)

        if is_docking_complete(output_directory, result_name):
            skipped_count += 1
        else:
            pending_ligands.append(ligand)

    batch_file = output_directory / "autodock_gpu_filelist.txt"

    with batch_file.open("w") as handle:
        for ligand in pending_ligands:
            result_name = result_name_for_ligand(ligand)
            handle.write(f"{grid_fld}\n")
            handle.write(f"{ligand.resolve()}\n")
            handle.write(f"{result_name}\n")

    command = [
        str(AUTODOCK_GPU),
        "--filelist",
        str(batch_file),
        "--nrun",
        str(N_RUNS),
        "--devnum",
        str(args.device),
        "--heuristics",
        "1",
        "--heurmax",
        str(HEURMAX),
        "--autostop",
        "1",
        "--stopstd",
        str(STOPSTD),
        "--xmloutput",
        "1",
        "--dlgoutput",
        "1",
        "--clustering",
        "1",
        "--gbest",
        "1",
    ]

    if args.contact_analysis:
        command.extend(["--contact_analysis", "1"])

    parameter_data = {
        "prepared": datetime.now().isoformat(timespec="seconds"),
        "conda_environment": os.environ.get("CONDA_DEFAULT_ENV", "unknown"),
        "autodock_gpu_executable": str(AUTODOCK_GPU),
        "grid_fld": str(grid_fld),
        "ligand_folder": str(ligand_folder),
        "ligand_pattern": "*-best.pdbqt",
        "total_ligands": len(ligands),
        "pending_ligands": len(pending_ligands),
        "previously_completed_and_skipped": skipped_count,
        "nrun": N_RUNS,
        "maximum_final_poses_exported_per_ligand": TOP_POSES,
        "pose_export_rule": (
            "Save all available poses when count is 1-20; "
            "save only the 20 lowest-energy poses when count is >20; "
            "zero poses is an extraction failure."
        ),
        "device": args.device,
        "heuristics": True,
        "heurmax": HEURMAX,
        "autostop": True,
        "stopstd": STOPSTD,
        "contact_analysis": args.contact_analysis,
        "output_name": output_name,
        "output_directory": str(output_directory),
        "top_pose_root": str(top_pose_root),
        "batch_file": str(batch_file),
        "command": command,
    }

    parameter_file = output_directory / "run_parameters.json"
    parameter_file.write_text(json.dumps(parameter_data, indent=2) + "\n")

    print("\nCD97 Stage-2 AutoDock-GPU docking")
    print("---------------------------------")
    print(f"Executable:          {AUTODOCK_GPU}")
    print(f"Grid maps:           {grid_fld}")
    print(f"Stage-2 input:       {ligand_folder}")
    print(f"Ligand pattern:      *-best.pdbqt")
    print(f"Total ligands:       {len(ligands)}")
    print(f"Pending ligands:     {len(pending_ligands)}")
    print(f"Already completed:   {skipped_count}")
    print(f"LGA runs/ligand:     {N_RUNS}")
    print(f"Exported poses:      all if <= {TOP_POSES}; best {TOP_POSES} if > {TOP_POSES}")
    print(f"GPU device:          {args.device}")
    print(f"Output folder:       {output_directory}")
    print(f"Top-pose folder:     {top_pose_root}")

    if args.extract_only:
        print("\nExtract-only mode: docking will not be started.")
        success_count, failure_count = extract_all_top_poses(
            ligands=ligands,
            output_directory=output_directory,
            top_pose_root=top_pose_root,
        )
        print(f"Top-20 extraction completed for {success_count} ligands.")
        print(f"Extraction failures: {failure_count}")
        return

    if not pending_ligands:
        print("\nAll ligand dockings already have DLG and XML output.")
        print("Rebuilding/verifying top-20 pose folders...")
        success_count, failure_count = extract_all_top_poses(
            ligands=ligands,
            output_directory=output_directory,
            top_pose_root=top_pose_root,
        )
        print(f"Top-20 extraction completed for {success_count} ligands.")
        print(f"Extraction failures: {failure_count}")
        return

    print("\nCommand:")
    print(" ".join(shlex.quote(part) for part in command))
    print()

    if args.dry_run:
        print("Dry run completed. Docking was not started.")
        return

    log_file = output_directory / "run.log"

    with log_file.open("a", buffering=1) as log:
        log.write("\n\n")
        log.write("=" * 78 + "\n")
        log.write(f"Started: {datetime.now().isoformat(timespec='seconds')}\n")
        log.write("Command: ")
        log.write(" ".join(shlex.quote(part) for part in command))
        log.write("\n")
        log.write("=" * 78 + "\n")
        log.flush()

        process = subprocess.Popen(
            command,
            cwd=output_directory,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )

        assert process.stdout is not None

        for line in process.stdout:
            print(line, end="")
            log.write(line)

        return_code = process.wait()

    parameter_data["finished"] = datetime.now().isoformat(timespec="seconds")
    parameter_data["return_code"] = return_code
    parameter_file.write_text(json.dumps(parameter_data, indent=2) + "\n")

    if return_code != 0:
        print(
            f"\nAutoDock-GPU stopped with exit code {return_code}.\n"
            f"Check the log:\n  {log_file}",
            file=sys.stderr,
        )
        sys.exit(return_code)

    print("\nDocking completed successfully.")
    print("Extracting the 20 lowest-energy final poses per ligand...")

    success_count, failure_count = extract_all_top_poses(
        ligands=ligands,
        output_directory=output_directory,
        top_pose_root=top_pose_root,
    )

    print(f"Top-20 extraction completed for {success_count} ligands.")
    print(f"Extraction failures: {failure_count}")
    print(f"Results:             {output_directory}")
    print(f"Top-20 poses:        {top_pose_root}")
    print(f"Pose summary:        {top_pose_root / 'top20_pose_summary.csv'}")
    print(f"Manifest:            {manifest_path}")
    print(f"Log:                 {log_file}")

    if failure_count:
        print(
            "\nSome DLG files contained no extractable energy-labelled poses. "
            "See:\n"
            f"  {top_pose_root / 'top20_extraction_failures.csv'}",
            file=sys.stderr,
        )
        sys.exit(2)


if __name__ == "__main__":
    main()
