# ADGRE5 Screening and REINVENT Workflows

Scripts for virtual screening and de novo molecular design targeting ADGRE5.

## Virtual screening

Scripts are located in [`01_virtual_screening/`](01_virtual_screening/).

| Script | Purpose |
| --- | --- |
| `prepare_ligands.py` | Prepare ligands for docking. |
| `run_cd97_autodock_gpu.py` | Run AutoDock-GPU docking. |
| `select_top_autodock.py` | Select top AutoDock-GPU results. |
| `contact_filter_autodock_weighted.py` | Filter and rank poses by contacts with target residues. |
| `run_cd97_contact_filter_weighted.sh` | Run the weighted contact filter. |
| `run_cd97_stage2_autodock_gpu_updated.py` | Run stage 2 AutoDock-GPU docking. |
| `filter_pdbqt_strain_gpu.py` | Calculate and filter ligand conformational strain. |
| `cd97_plip_score_select_diverse.py` | Score PLIP interactions and select diverse compounds. |
| `filter_diversity_butina.py` | Cluster compounds by chemical similarity. |
| `run_cd97_gnina_minimize_cnn.py` | Minimize and rescore poses with GNINA. |
| `select_top_gnina_unique_for_plip.py` | Select unique GNINA results and prepare complexes. |
| `select_top_gnina_unique_for_plip_pose_safe.py` | Select unique GNINA results while preserving pose coordinates. |
| `run_accurate_interaction_diagrams.sh` | Run the interaction-diagram workflow. |

## De novo design

Scripts and configuration are located in [`02_denovo_reinvent4/`](02_denovo_reinvent4/).

| File | Purpose |
| --- | --- |
| `cd97_reinvent4_dap_REAL_ORACLE.py` | Calculate multi-objective scores for generated molecules. |
| `comp_cd97_real_oracle.py` | Connect the scoring engine to REINVENT 4. |
| `reinvent4_fresh_1000steps.toml` | Configure the REINVENT 4 run. |

## Molecular-dynamics analysis

| Script | Purpose |
| --- | --- |
| [`select_top5_cd97_persistent.py`](select_top5_cd97_persistent.py) | Rank ligands using trajectory stability and contact persistence. |
| [`select_final_top5_cd97.py`](select_final_top5_cd97.py) | Select final leads using contact retention and stability metrics. |
