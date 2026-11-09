# ADGRE5 (CD97) screening and de novo design

Scripts for a computational workflow targeting human ADGRE5 (CD97), including virtual screening, interaction scoring, REINVENT 4 de novo design, and short molecular-dynamics screening. ZINC20 curation and BitBIRCH clustering were performed separately. Longer membrane simulations were prepared separately with CHARMM-GUI.

## Contents

- `01_virtual_screening/`: ligand preparation, AutoDock-GPU, strain filtering, PLIP interaction scoring, Butina diversity selection, and subsequent GNINA rescoring. See the folder README for details.
- `02_denovo_reinvent4/`: REINVENT 4 configuration, custom scoring adapter, and multi-objective oracle. The oracle includes GNINA. See its README for installation and input requirements.
- `select_top5_cd97_persistent.py` and `select_final_top5_cd97.py`: downstream trajectory and contact analysis. These expect separately generated screening MD outputs.



## Requirements

Depending on the stage, this project uses Python, RDKit, Open Babel, Meeko, NumPy, pandas, SciPy, PLIP, PyTorch, TorchANI, XGBoost, MDAnalysis, REINVENT 4, AutoDock-GPU, GNINA, GROMACS, AmberTools, and ACPYPE. The `environment.yml` and `requirements.txt` files are preliminary dependency lists.
