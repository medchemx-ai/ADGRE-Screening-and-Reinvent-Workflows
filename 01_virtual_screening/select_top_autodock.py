#!/usr/bin/env python3
"""
Select the best-scoring AutoDock-GPU ligands from one result directory.

Expected AutoDock-GPU files for each ligand:
    <basename>.xml
    <basename>.dlg
    <basename>-best.pdbqt

The script:
  1. Reads the minimum <free_NRG_binding> value from each XML file.
  2. Falls back to "Estimated Free Energy of Binding" in the DLG if needed.
  3. Sorts scores from most negative to least negative.
  4. Copies/links the selected ligand's XML, DLG, and best-pose PDBQT.
  5. Writes a CSV ranking and a parsing-failure report.

Output folder:
    <input_folder>/<input_folder_name>_best_<TOP_N>

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
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional


FLOAT_PATTERN = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[Ee][+-]?\d+)?"

XML_SCORE_RE = re.compile(
    rf"<\s*free_NRG_binding\s*>\s*({FLOAT_PATTERN})\s*<\s*/\s*free_NRG_binding\s*>",
    re.IGNORECASE,
)

DLG_SCORE_RE = re.compile(
    rf"Estimated\s+Free\s+Energy\s+of\s+Binding\s*=\s*({FLOAT_PATTERN})",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class Result:
    basename: str
    score: float
    source: str
    source_file: Path


def parse_xml_score(path: Path) -> Optional[float]:
    """Return the lowest free_NRG_binding score in an AutoDock-GPU XML file."""
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None

    values = []
    for match in XML_SCORE_RE.finditer(text):
        try:
            value = float(match.group(1))
        except ValueError:
            continue
        if math.isfinite(value):
            values.append(value)

    return min(values) if values else None


def parse_dlg_score(path: Path) -> Optional[float]:
    """Return the lowest Estimated Free Energy of Binding in a DLG file."""
    values = []

    try:
        with path.open("r", encoding="utf-8", errors="replace") as handle:
            for line in handle:
                match = DLG_SCORE_RE.search(line)
                if not match:
                    continue
                try:
                    value = float(match.group(1))
                except ValueError:
                    continue
                if math.isfinite(value):
                    values.append(value)
    except OSError:
        return None

    return min(values) if values else None


def collect_results(
    input_dir: Path,
    score_source: str,
    report_every: int,
) -> tuple[list[Result], list[tuple[str, str]]]:
    """
    Collect one best score per ligand basename.

    In auto mode, XML is preferred and DLG is used only when XML is absent
    or does not contain a readable free_NRG_binding value.
    """
    results: dict[str, Result] = {}
    failures: list[tuple[str, str]] = []

    xml_files = sorted(input_dir.glob("*.xml"))
    dlg_files = sorted(input_dir.glob("*.dlg"))

    if score_source in {"auto", "xml"}:
        total = len(xml_files)
        for index, path in enumerate(xml_files, start=1):
            score = parse_xml_score(path)
            if score is None:
                failures.append((path.name, "No readable <free_NRG_binding> value"))
            else:
                results[path.stem] = Result(
                    basename=path.stem,
                    score=score,
                    source="xml",
                    source_file=path,
                )

            if report_every > 0 and index % report_every == 0:
                print(f"Parsed XML: {index:,}/{total:,}", file=sys.stderr)

    if score_source in {"auto", "dlg"}:
        total = len(dlg_files)
        for index, path in enumerate(dlg_files, start=1):
            basename = path.stem

            if score_source == "auto" and basename in results:
                continue

            score = parse_dlg_score(path)
            if score is None:
                failures.append(
                    (path.name, "No readable Estimated Free Energy of Binding value")
                )
            else:
                results[basename] = Result(
                    basename=basename,
                    score=score,
                    source="dlg",
                    source_file=path,
                )

            if report_every > 0 and index % report_every == 0:
                print(f"Parsed DLG: {index:,}/{total:,}", file=sys.stderr)

    return list(results.values()), failures


def associated_files(input_dir: Path, basename: str) -> list[Path]:
    """Return the standard AutoDock-GPU output triplet that exists."""
    candidates = [
        input_dir / f"{basename}.xml",
        input_dir / f"{basename}.dlg",
        input_dir / f"{basename}-best.pdbqt",
    ]
    return [path for path in candidates if path.is_file()]


def transfer_file(source: Path, destination: Path, mode: str) -> None:
    """Copy, hard-link, or symlink a file."""
    if mode == "copy":
        shutil.copy2(source, destination)
    elif mode == "hardlink":
        os.link(source, destination)
    elif mode == "symlink":
        relative_source = os.path.relpath(source, start=destination.parent)
        destination.symlink_to(relative_source)
    else:
        raise ValueError(f"Unsupported mode: {mode}")


def write_csv(
    path: Path,
    selected: Iterable[Result],
    input_dir: Path,
    mode: str,
) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "rank",
                "basename",
                "autodock_score_kcal_mol",
                "score_source",
                "xml_present",
                "dlg_present",
                "best_pose_present",
                "transfer_mode",
            ]
        )

        for rank, result in enumerate(selected, start=1):
            writer.writerow(
                [
                    rank,
                    result.basename,
                    f"{result.score:.6f}",
                    result.source,
                    (input_dir / f"{result.basename}.xml").is_file(),
                    (input_dir / f"{result.basename}.dlg").is_file(),
                    (input_dir / f"{result.basename}-best.pdbqt").is_file(),
                    mode,
                ]
            )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Select the top-scoring AutoDock-GPU ligands and place their "
            "XML, DLG and -best.pdbqt files in a new subfolder."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "input_dir",
        type=Path,
        help="AutoDock-GPU output directory",
    )
    parser.add_argument(
        "top_n",
        type=int,
        help="Number of best-scoring ligands to select, for example 5000",
    )
    parser.add_argument(
        "--score-source",
        choices=("auto", "xml", "dlg"),
        default="auto",
        help="Use XML, DLG, or XML with DLG fallback",
    )
    parser.add_argument(
        "--score-lte",
        type=float,
        default=None,
        metavar="VALUE",
        help="Keep only scores less than or equal to VALUE, e.g. -6.0",
    )
    parser.add_argument(
        "--score-gte",
        type=float,
        default=None,
        metavar="VALUE",
        help="Keep only scores greater than or equal to VALUE",
    )
    parser.add_argument(
        "--mode",
        choices=("copy", "hardlink", "symlink"),
        default="copy",
        help="How selected files are placed in the output directory",
    )
    parser.add_argument(
        "--allow-missing-pose",
        action="store_true",
        help="Allow selection even when <basename>-best.pdbqt is missing",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete and recreate an existing output directory",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Rank ligands and write no output files",
    )
    parser.add_argument(
        "--report-every",
        type=int,
        default=10000,
        help="Print parsing progress after this many files; use 0 to disable",
    )

    return parser


def main() -> int:
    args = build_parser().parse_args()

    input_dir = args.input_dir.expanduser().resolve()

    if not input_dir.is_dir():
        print(f"ERROR: Input directory does not exist: {input_dir}", file=sys.stderr)
        return 2

    if args.top_n <= 0:
        print("ERROR: top_n must be greater than zero.", file=sys.stderr)
        return 2

    if (
        args.score_lte is not None
        and args.score_gte is not None
        and args.score_gte > args.score_lte
    ):
        print(
            "ERROR: --score-gte cannot be greater than --score-lte.",
            file=sys.stderr,
        )
        return 2

    output_dir = input_dir / f"{input_dir.name}_best_{args.top_n}"

    print(f"Input directory : {input_dir}")
    print(f"Requested top N : {args.top_n:,}")
    print(f"Score source    : {args.score_source}")
    print("Ranking rule    : lower/more-negative AutoDock score is better")

    results, failures = collect_results(
        input_dir=input_dir,
        score_source=args.score_source,
        report_every=args.report_every,
    )

    filtered = []
    for result in results:
        if args.score_lte is not None and result.score > args.score_lte:
            continue
        if args.score_gte is not None and result.score < args.score_gte:
            continue

        pose_path = input_dir / f"{result.basename}-best.pdbqt"
        if not args.allow_missing_pose and not pose_path.is_file():
            failures.append((result.basename, "Missing -best.pdbqt pose"))
            continue

        filtered.append(result)

    filtered.sort(key=lambda item: (item.score, item.basename))
    selected = filtered[: args.top_n]

    print(f"Readable scores : {len(results):,}")
    print(f"Eligible results: {len(filtered):,}")
    print(f"Selected        : {len(selected):,}")

    if not selected:
        print(
            "ERROR: No ligands were selected. Inspect one XML with:\n"
            "  grep -n 'free_NRG_binding' ligand.xml | head\n"
            "or one DLG with:\n"
            "  grep -n 'Estimated Free Energy of Binding' ligand.dlg | head",
            file=sys.stderr,
        )
        return 3

    if len(selected) < args.top_n:
        print(
            f"WARNING: Only {len(selected):,} eligible ligands were found.",
            file=sys.stderr,
        )

    if args.dry_run:
        print("\nDry run: no folder or files were created.")
        print("\nTop 10:")
        for rank, result in enumerate(selected[:10], start=1):
            print(
                f"{rank:>4}  {result.score:>10.3f}  "
                f"{result.source:<3}  {result.basename}"
            )
        return 0

    if output_dir.exists():
        if not args.overwrite:
            print(
                f"ERROR: Output directory already exists:\n  {output_dir}\n"
                "Use --overwrite to replace it.",
                file=sys.stderr,
            )
            return 4

        if output_dir.parent != input_dir:
            print("ERROR: Unsafe output path; refusing to delete.", file=sys.stderr)
            return 4
        shutil.rmtree(output_dir)

    output_dir.mkdir(parents=False, exist_ok=False)

    transferred_count = 0
    missing_files: list[tuple[str, str]] = []

    try:
        for result in selected:
            files = associated_files(input_dir, result.basename)

            for source in files:
                destination = output_dir / source.name
                transfer_file(source, destination, args.mode)
                transferred_count += 1

            expected = {
                f"{result.basename}.xml",
                f"{result.basename}.dlg",
                f"{result.basename}-best.pdbqt",
            }
            found = {path.name for path in files}

            for missing in sorted(expected - found):
                missing_files.append((result.basename, missing))

        write_csv(
            output_dir / "selected_scores.csv",
            selected,
            input_dir,
            args.mode,
        )

        with (output_dir / "score_parsing_failures.csv").open(
            "w", newline="", encoding="utf-8"
        ) as handle:
            writer = csv.writer(handle)
            writer.writerow(["file_or_basename", "reason"])
            writer.writerows(failures)

        with (output_dir / "missing_associated_files.csv").open(
            "w", newline="", encoding="utf-8"
        ) as handle:
            writer = csv.writer(handle)
            writer.writerow(["basename", "missing_file"])
            writer.writerows(missing_files)

    except Exception:
        print(
            f"ERROR: Transfer failed. Partial output remains at:\n  {output_dir}",
            file=sys.stderr,
        )
        raise

    print(f"Output directory : {output_dir}")
    print(f"Files transferred: {transferred_count:,}")
    print(f"Best score       : {selected[0].score:.3f} kcal/mol")
    print(f"Cutoff score     : {selected[-1].score:.3f} kcal/mol")
    print(f"Ranking CSV      : {output_dir / 'selected_scores.csv'}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
