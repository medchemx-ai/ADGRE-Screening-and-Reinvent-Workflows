#!/usr/bin/env bash
set -euo pipefail

# ==============================================================================
# Run Accurate, Non-Fabricated 2D Interaction Diagrams on the 40 GNINA Leads
# ==============================================================================

WORK_DIR="/home/linux/Ayan_Computational/schodinger"
GNINA_SRC="/home/linux/Ayan_Computational/dockresult/adgpu_batch_00_smiles_nrun10/adgpu_batch_00_smiles_nrun10_best_5000/adgpu_batch_00_smiles_nrun10_best_5000_top1000_contact_analysis/Stage2_input_nrun50/GNINA_OUT/gnina_lead"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PIPELINE_DIR="${SCRIPT_DIR}/../accurate_interaction_diagrams"

echo "======================================================================"
echo "Accurate 2D Protein-Ligand Interaction Diagram Generation (40 Leads)"
echo "======================================================================"

cd "${WORK_DIR}"

# 1. Ensure gnina_lead symlink exists in ~/Ayan_Computational/schodinger
if [ ! -e "./gnina_lead" ]; then
    echo "[+] Creating symlink to gnina_lead..."
    ln -s "${GNINA_SRC}" ./gnina_lead
else
    echo "[*] gnina_lead symlink already exists."
fi

# 2. Check pipeline location (script directory or local directory)
if [ ! -d "${PIPELINE_DIR}" ]; then
    if [ -d "./accurate_interaction_diagrams" ]; then
        PIPELINE_DIR="./accurate_interaction_diagrams"
    else
        echo "[-] Error: Pipeline directory not found at:"
        echo "    ${PIPELINE_DIR} or ./accurate_interaction_diagrams"
        echo "    Please ensure accurate_interaction_diagrams is copied to your script folder."
        exit 1
    fi
fi

# 3. Execute the pipeline on the 40 complexes
echo "[+] Running coordinate-based interaction pipeline on 40 complexes..."
python3 "${PIPELINE_DIR}/generate_interaction_diagrams.py" \
    --input ./gnina_lead/complexes \
    --output ./gnina_lead/accurate_interaction_results

echo ""
echo "======================================================================"
echo "[SUCCESS] Generated 2D Interaction Diagrams for the 40 Leads!"
echo "Master Summary: ${WORK_DIR}/gnina_lead/accurate_interaction_results/interaction_summary.csv"
echo "Diagrams Folder: ${WORK_DIR}/gnina_lead/accurate_interaction_results/"
echo "======================================================================"
