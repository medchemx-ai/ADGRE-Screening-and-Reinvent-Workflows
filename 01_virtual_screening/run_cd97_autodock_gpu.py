#!/usr/bin/env python3
"""
CD97 batch docking with AutoDock-GPU.

Fixed:
- AutoDock-GPU executable from the drugdesign_env Conda environment
- receptor/grid-map directory
- output parent directory

Given from Bash:
1. ligand PDBQT folder
2. number of AutoDock-GPU LGA runs (--nrun)

Example:
    conda activate drugdesign_env

    python3 run_cd97_autodock_gpu.py \
        /home/linux/Ayan_Computational/Autodoc_prepared_ligand/pdbqt/adgpu_batch_00_smilies \
        10

Output:
    /home/linux/Ayan_Computational/dockresult/adgpu_batch_00_smilies_nrun10/

IMPORTANT:
AutoDock-GPU does not use receptor PDBQT + Vina center/size directly.
It requires an AutoGrid descriptor file ending in .maps.fld.
The grid center and dimensions are already encoded in those map files.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shlex
import subprocess
import sys
from datetime import datetime
from pathlib import Path


# ============================================================
# FIXED PATHS
# ============================================================

AUTODOCK_GPU = Path(
    "/media/linux/2c17f3da-5588-420c-a3a0-dab5c2b839ff/"
    "Home/anaconda3/envs/drugdesign_env/bin/autodock_gpu_128wi"
)

RECEPTOR_DIR = Path(
    "/home/linux/Ayan_Computational/receptor"
)

# Preferred AutoGrid descriptor name. If it does not exist, the script
# will accept the only *.maps.fld file present inside RECEPTOR_DIR.
PREFERRED_GRID_FLD = RECEPTOR_DIR / "CD_receptor.maps.fld"

RESULT_ROOT = Path(
    "/home/linux/Ayan_Computational/dockresult"
)

DEFAULT_DEVICE = 1


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run a folder of ligand PDBQT files against fixed CD97 "
            "AutoGrid maps using AutoDock-GPU."
        )
    )

    parser.add_argument(
        "ligand_folder",
        help="Folder containing ligand .pdbqt files.",
    )

    parser.add_argument(
        "nrun",
        type=int,
        help=(
            "Number of independent AutoDock-GPU LGA runs per ligand. "
            "Examples: 5, 10, 20."
        ),
    )

    parser.add_argument(
        "--device",
        type=int,
        default=DEFAULT_DEVICE,
        help=f"GPU device number; counting starts at 1. Default: {DEFAULT_DEVICE}.",
    )

    parser.add_argument(
        "--contact-analysis",
        action="store_true",
        help="Enable AutoDock-GPU distance-based contact analysis.",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate inputs and create batch files without starting docking.",
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
            f"Expected one inside:\n  {RECEPTOR_DIR}\n\n"
            "CD_receptor.pdbqt alone is not sufficient for AutoDock-GPU. "
            "Generate the AutoGrid maps first, then place the resulting "
            "*.maps.fld and companion map files in the receptor folder."
        )

    names = "\n".join(f"  {path}" for path in candidates)
    raise RuntimeError(
        "\nMore than one .maps.fld file was found. Rename the required file to:\n"
        f"  {PREFERRED_GRID_FLD}\n\n"
        f"Files found:\n{names}"
    )


def safe_output_name(name: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", name).strip("._")
    return cleaned or "ligand"


def validate_inputs(
    ligand_folder: Path,
    nrun: int,
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
            "Ligand folder was not found:\n"
            f"  {ligand_folder}"
        )

    ligands: list[Path] = []
    if ligand_folder.is_dir():
        ligands = sorted(ligand_folder.glob("*.pdbqt"))
        if not ligands:
            errors.append(
                "No .pdbqt files were found directly inside:\n"
                f"  {ligand_folder}"
            )

    if nrun < 1:
        errors.append("nrun must be at least 1.")

    if device < 1:
        errors.append("GPU device number must be at least 1.")

    output_names: dict[str, Path] = {}
    duplicates: list[str] = []

    for ligand in ligands:
        name = safe_output_name(ligand.stem)
        if name in output_names:
            duplicates.append(
                f"  {output_names[name].name} and {ligand.name} -> {name}"
            )
        else:
            output_names[name] = ligand

    if duplicates:
        errors.append(
            "Some ligand filenames would produce duplicate output names:\n"
            + "\n".join(duplicates)
        )

    if errors:
        print("\nINPUT ERROR", file=sys.stderr)
        print("-----------", file=sys.stderr)
        for error in errors:
            print(f"\n{error}", file=sys.stderr)
        sys.exit(1)

    return ligands


def is_completed(output_directory: Path, result_name: str) -> bool:
    """
    Treat a ligand as completed only when both its DLG and XML files exist
    and are non-empty. This allows safe restarting of interrupted batches.
    """
    dlg = output_directory / f"{result_name}.dlg"
    xml = output_directory / f"{result_name}.xml"

    return (
        dlg.is_file()
        and dlg.stat().st_size > 0
        and xml.is_file()
        and xml.stat().st_size > 0
    )


def write_manifest(
    manifest_path: Path,
    ligands: list[Path],
    output_directory: Path,
) -> None:
    with manifest_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "ligand_filename",
                "ligand_full_path",
                "result_name",
                "status_before_run",
            ]
        )

        for ligand in ligands:
            result_name = safe_output_name(ligand.stem)
            status = (
                "completed"
                if is_completed(output_directory, result_name)
                else "pending"
            )
            writer.writerow(
                [
                    ligand.name,
                    str(ligand.resolve()),
                    result_name,
                    status,
                ]
            )


def main() -> None:
    args = parse_arguments()

    ligand_folder = Path(args.ligand_folder).expanduser().resolve()

    ligands = validate_inputs(
        ligand_folder=ligand_folder,
        nrun=args.nrun,
        device=args.device,
    )

    try:
        grid_fld = locate_grid_fld()
    except (FileNotFoundError, RuntimeError) as error:
        print(error, file=sys.stderr)
        sys.exit(1)

    RESULT_ROOT.mkdir(parents=True, exist_ok=True)

    output_directory = (
        RESULT_ROOT / f"{ligand_folder.name}_nrun{args.nrun}"
    )
    output_directory.mkdir(parents=True, exist_ok=True)

    manifest_path = output_directory / "ligand_manifest.csv"
    write_manifest(
        manifest_path=manifest_path,
        ligands=ligands,
        output_directory=output_directory,
    )

    pending_ligands: list[Path] = []
    skipped_count = 0

    for ligand in ligands:
        result_name = safe_output_name(ligand.stem)
        if is_completed(output_directory, result_name):
            skipped_count += 1
        else:
            pending_ligands.append(ligand)

    if not pending_ligands:
        print("\nAll ligands already have completed DLG and XML results.")
        print(f"Output folder: {output_directory}")
        return

    # AutoDock-GPU batch format:
    # line 1 = receptor .maps.fld
    # line 2 = ligand .pdbqt
    # line 3 = output result name
    # Repeat the three lines for every ligand.
    batch_file = output_directory / "autodock_gpu_filelist.txt"

    with batch_file.open("w") as handle:
        for ligand in pending_ligands:
            result_name = safe_output_name(ligand.stem)
            handle.write(f"{grid_fld}\n")
            handle.write(f"{ligand.resolve()}\n")
            handle.write(f"{result_name}\n")

    command = [
        str(AUTODOCK_GPU),
        "--filelist", str(batch_file),
        "--nrun", str(args.nrun),
        "--devnum", str(args.device),
        "--heuristics", "1",
        "--autostop", "1",
        "--xmloutput", "1",
        "--dlgoutput", "1",
        "--gbest", "1",
    ]

    if args.contact_analysis:
        command.extend(["--contact_analysis", "1"])

    parameter_data = {
        "started": datetime.now().isoformat(timespec="seconds"),
        "conda_environment": os.environ.get("CONDA_DEFAULT_ENV", "unknown"),
        "autodock_gpu_executable": str(AUTODOCK_GPU),
        "grid_fld": str(grid_fld),
        "ligand_folder": str(ligand_folder),
        "total_ligands": len(ligands),
        "pending_ligands": len(pending_ligands),
        "previously_completed_and_skipped": skipped_count,
        "nrun": args.nrun,
        "device": args.device,
        "contact_analysis": args.contact_analysis,
        "output_directory": str(output_directory),
        "batch_file": str(batch_file),
        "command": command,
    }

    parameter_file = output_directory / "run_parameters.json"
    parameter_file.write_text(json.dumps(parameter_data, indent=2) + "\n")

    print("\nCD97 AutoDock-GPU batch docking")
    print("-------------------------------")
    print(f"Conda environment: {os.environ.get('CONDA_DEFAULT_ENV', 'unknown')}")
    print(f"Executable:        {AUTODOCK_GPU}")
    print(f"Grid maps:         {grid_fld}")
    print(f"Ligand folder:     {ligand_folder}")
    print(f"Total ligands:     {len(ligands)}")
    print(f"Pending ligands:   {len(pending_ligands)}")
    print(f"Already completed: {skipped_count}")
    print(f"LGA runs:          {args.nrun}")
    print(f"GPU device:        {args.device}")
    print(f"Output folder:     {output_directory}")
    print("\nCommand:")
    print(" ".join(shlex.quote(part) for part in command))
    print()

    if args.dry_run:
        print("Dry run completed. Docking was not started.")
        return

    log_file = output_directory / "run.log"

    with log_file.open("a", buffering=1) as log:
        log.write("\n\n")
        log.write("=" * 72 + "\n")
        log.write(f"Started: {datetime.now().isoformat(timespec='seconds')}\n")
        log.write("Command: ")
        log.write(" ".join(shlex.quote(part) for part in command))
        log.write("\n")
        log.write("=" * 72 + "\n")
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
    print(f"Results:  {output_directory}")
    print(f"Manifest: {manifest_path}")
    print(f"Log:      {log_file}")


if __name__ == "__main__":
    main()
