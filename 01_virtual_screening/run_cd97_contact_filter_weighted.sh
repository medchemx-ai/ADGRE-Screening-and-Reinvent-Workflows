#!/usr/bin/env bash
set -euo pipefail

# ---------- EDIT THESE PATHS ----------
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_SCRIPT="${SCRIPT_DIR}/contact_filter_autodock_weighted.py"

# Exact receptor PDB/PDBQT used for docking.
RECEPTOR="$HOME/Ayan_Computational/receptor/CD_receptor.pdbqt"

# Folder containing the selected 5000 AutoDock outputs.
INPUT="$HOME/Ayan_Computational/dockresult/adgpu_batch_00_smiles_nrun10/adgpu_batch_00_smiles_nrun10_best_5000"
# --------------------------------------

TOP_N=1000
CUTOFF=4.0
MIN_TOTAL_CONTACTS=1
WORKERS=8
TRANSFER_MODE="copy"     # copy, hardlink, or symlink
DRY_RUN=false
REQUIRE_LID=false

# Highest importance.
LID=(
  W685
  F760
)

# Equal lower importance.
HYDROPHOBIC=(
  F597
  F623
  W753
  F771
  L555
  L757
)

# Equal lower importance.
POLAR=(
  N775
  Q778
  S558
  T752
  C622
)

# Lid highest (wtier = 4.0 as defined in Supplementary Section S1.2.6); hydrophobic and polar equal (wtier = 2.0).
LID_WEIGHT=4.0
HYDROPHOBIC_WEIGHT=2.0
POLAR_WEIGHT=2.0

EXTRA_ARGS=()

if [[ "$DRY_RUN" == "true" ]]; then
  EXTRA_ARGS+=(--dry-run)
fi

if [[ "$REQUIRE_LID" == "true" ]]; then
  EXTRA_ARGS+=(--require-lid)
fi

python3 "$PYTHON_SCRIPT" \
  "$RECEPTOR" \
  "$INPUT" \
  "$TOP_N" \
  --lid "${LID[@]}" \
  --hydrophobic "${HYDROPHOBIC[@]}" \
  --polar "${POLAR[@]}" \
  --lid-weight "$LID_WEIGHT" \
  --hydrophobic-weight "$HYDROPHOBIC_WEIGHT" \
  --polar-weight "$POLAR_WEIGHT" \
  --cutoff "$CUTOFF" \
  --min-total-contacts "$MIN_TOTAL_CONTACTS" \
  --workers "$WORKERS" \
  --mode "$TRANSFER_MODE" \
  "${EXTRA_ARGS[@]}"

echo
echo "Expected output folder:"
echo "$INPUT/$(basename "$INPUT")_top${TOP_N}_contact_analysis"
