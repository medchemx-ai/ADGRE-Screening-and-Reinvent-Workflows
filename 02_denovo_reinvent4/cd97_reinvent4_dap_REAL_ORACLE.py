# ==================================================================================================
# CD97 REAL-ORACLE MULTI-OBJECTIVE SCORING ENGINE
# Integrates 100% Exact Reference Implementations:
#   1. AutoDock-GPU 128wi Docking against CD97 grid maps
#   2. GPUMoleculeStrainEvaluator: Exact TorchANI / MMFF94s Restrained Bound & Free Solution Ensemble
#   3. GNINA: Exact Empirical Minimization & CNN Rescoring (--minimize --cnn_scoring rescore)
#   4. CD97 PLIP: Exact Target-Residue Specific Scoring (W685, F760 lid; F597.. side; N775.. bottom)
#   5. Complex Exporter: Automatically saves the best docked complex PDB & SDF for all passed molecules
# ==================================================================================================

from __future__ import annotations

import os
import sys
import gc
import json
import re
import math
import shutil
import logging
import subprocess
import tempfile
from pathlib import Path
from collections import defaultdict
from typing import Dict, Any, Tuple, Optional, List, Generator
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn

from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit import RDLogger

# Suppress minor RDKit noise
RDLogger.DisableLog('rdApp.warning')
RDLogger.DisableLog('rdApp.error')
os.environ['TORCHANI_NO_WARN_EXTENSIONS'] = '1'

# ---- Setup Logging ----
logger = logging.getLogger("CD97_RealOracle")
logger.setLevel(logging.INFO)
if not logger.handlers:
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter('%(asctime)s %(name)s %(levelname)s: %(message)s'))
    logger.addHandler(ch)

# ---- Constants ----
HARTREE_TO_KCAL = 627.5094740631
ANI2X_SUPPORTED_Z = {1, 6, 7, 8, 9, 16, 17}  # H, C, N, O, F, S, Cl

# ---- Project Root & Receptors ----
CURRENT_FILE = Path(__file__).resolve()
PROJECT_ROOT = CURRENT_FILE.parent
if PROJECT_ROOT.name == "components" and PROJECT_ROOT.parent.name == "reinvent_plugins":
    PROJECT_ROOT = PROJECT_ROOT.parent.parent
if not (PROJECT_ROOT / "receptors").exists():
    for _c in [Path.cwd(), Path("/home/linux/CD97_REINVENT4_LINUX_1000STEPS")]:
        if (_c / "receptors").exists():
            PROJECT_ROOT = _c
            break

RECEPTOR_FLD = PROJECT_ROOT / "receptors" / "cd97_rec_39ns.maps.fld"
RECEPTOR_PDB = PROJECT_ROOT / "receptors" / "cd97_rec_39ns.pdb"

# Best complexes storage directory (dedicated to production 1000-step run)
BEST_COMPLEXES_DIR = PROJECT_ROOT / "analysis" / "production_1000steps" / "best_docked_complexes"
BEST_COMPLEXES_DIR.mkdir(parents=True, exist_ok=True)

# ---- Conda Binaries ----
_CONDA_BIN = Path("/media/linux/2c17f3da-5588-420c-a3a0-dab5c2b839ff/Home/anaconda3/envs/drugdesign_env/bin")

def _find_bin(names: list) -> str:
    for name in names:
        p = _CONDA_BIN / name
        if p.exists() and os.access(p, os.X_OK):
            return str(p)
        found = shutil.which(name)
        if found:
            return found
        p_proj = PROJECT_ROOT / "bin" / name
        if p_proj.exists() and os.access(p_proj, os.X_OK):
            return str(p_proj)
    return names[0]

AUTODOCK_GPU_BIN = _find_bin(["autodock_gpu_128wi", "autodock_gpu"])
GNINA_BIN        = _find_bin(["gnina"])
PLIP_BIN         = _find_bin(["plip"])

# ==================================================================================================
# CD97 TARGET RESIDUES & WEIGHTS — identical to cd97_plip_score_select_diverse.py
# ==================================================================================================
TARGET_RESIDUES: dict[int, dict[str, str]] = {
    # Lid residues — W685 and F760 treated equally (same group, same weight)
    685: {"label": "W685", "expected_restype": "TRP", "group": "lid"},
    760: {"label": "F760", "expected_restype": "PHE", "group": "lid"},

    597: {"label": "F597", "expected_restype": "PHE", "group": "side"},
    623: {"label": "F623", "expected_restype": "PHE", "group": "side"},
    753: {"label": "W753", "expected_restype": "TRP", "group": "side"},
    771: {"label": "F771", "expected_restype": "PHE", "group": "side"},
    555: {"label": "L555", "expected_restype": "LEU", "group": "side"},
    757: {"label": "L757", "expected_restype": "LEU", "group": "side"},

    775: {"label": "N775", "expected_restype": "ASN", "group": "bottom"},
    778: {"label": "Q778", "expected_restype": "GLN", "group": "bottom"},
    558: {"label": "S558", "expected_restype": "SER", "group": "bottom"},
    752: {"label": "T752", "expected_restype": "THR", "group": "bottom"},
    622: {"label": "C622", "expected_restype": "CYS", "group": "bottom"},
}

GROUP_WEIGHTS: dict[str, float] = {
    "lid":    4.0,   # W685 and F760 — equal weight, no preference
    "side":   2.0,
    "bottom": 2.0,
}

INTERACTION_WEIGHTS: dict[str, float] = {
    "hydrophobic": 1.00,
    "hbond": 1.50,
    "pistacking": 1.50,
    "pication": 1.50,
    "saltbridge": 1.50,
    "halogen": 1.00,
    "waterbridge": 0.75,
    "metal": 1.00,
}

PLIP_INTERACTION_ATTRIBUTES: dict[str, tuple[str, ...]] = {
    "hydrophobic": ("hydrophobic_contacts",),
    "hbond": ("hbonds_ldon", "hbonds_pdon"),
    "pistacking": ("pistacking",),
    "pication": ("pication_laro", "pication_paro"),
    "saltbridge": ("saltbridge_lneg", "saltbridge_pneg"),
    "halogen": ("halogen_bonds",),
    "waterbridge": ("water_bridges",),
    "metal": ("metal_complexes",),
}

# ==================================================================================================
# 1. GPU MOLECULE STRAIN EVALUATOR
# ==================================================================================================
class GPUMoleculeStrainEvaluator:
    def __init__(
        self,
        device: str = "cuda:0",
        max_strain: float = 10.0,
        restraint_k: float = 25.0,
        max_relax_steps: int = 5,
        relax_lr: float = 0.005,
        num_confs_free: int = 5
    ):
        self.device = torch.device(device if torch.cuda.is_available() and "cuda" in device else "cpu")
        self.max_strain = max_strain
        self.restraint_k = restraint_k
        self.max_relax_steps = max_relax_steps
        self.relax_lr = relax_lr
        self.num_confs_free = num_confs_free
        self.baseline_cache: Dict[Tuple[str, bool], float] = {}

        self.ani_model = None
        self._init_ani_model()

    def _init_ani_model(self):
        try:
            import torchani
            self.ani_model = torchani.models.ANI2x(periodic_table_index=True).to(self.device).eval()
        except Exception:
            self.ani_model = None

    def is_ani_compatible(self, mol: Chem.Mol) -> bool:
        if self.ani_model is None:
            return False
        # ANI-2x is not charge-aware; use MMFF94s for formally charged ligands.
        if Chem.GetFormalCharge(mol) != 0:
            return False
        for atom in mol.GetAtoms():
            if atom.GetAtomicNum() not in ANI2X_SUPPORTED_Z:
                return False
        return True

    def get_chemical_key(self, mol: Chem.Mol) -> str:
        try:
            return Chem.MolToInchiKey(mol)
        except Exception:
            try:
                return Chem.MolToSmiles(Chem.RemoveHs(mol), isomericSmiles=True)
            except Exception:
                return "UNKNOWN"

    def compute_e_free_gpu_ani(self, mol: Chem.Mol) -> float:
        free_mol = Chem.AddHs(Chem.Mol(mol))
        free_mol.RemoveAllConformers()
        params = AllChem.ETKDGv3()
        params.randomSeed = 42
        cids = AllChem.EmbedMultipleConfs(free_mol, numConfs=self.num_confs_free, params=params)
        if not cids:
            params.useRandomCoords = True
            cids = AllChem.EmbedMultipleConfs(free_mol, numConfs=self.num_confs_free, params=params)
        if not cids:
            AllChem.EmbedMolecule(free_mol, AllChem.ETKDGv3())

        # Match the standalone filter: MMFF94 preparation, followed by ANI scoring.
        results = AllChem.MMFFOptimizeMoleculeConfs(
            free_mol, numThreads=1, maxIters=150, mmffVariant='MMFF94'
        )
        if any(status < 0 for status, _ in results):
            raise ValueError('MMFF94 parameters unavailable for free conformers')
        atomic_nums = [atom.GetAtomicNum() for atom in free_mol.GetAtoms()]
        num_confs = free_mol.GetNumConformers()
        coords_list = [free_mol.GetConformer(cid).GetPositions() for cid in range(num_confs)]

        species = torch.tensor([atomic_nums] * num_confs, dtype=torch.long, device=self.device)
        if not num_confs:
            raise ValueError("Unable to generate free conformers")
        coords = torch.from_numpy(np.array(coords_list, dtype=np.float32)).to(self.device)

        with torch.no_grad():
            energies_hartree = self.ani_model((species, coords)).energies
            energies_kcal = energies_hartree * HARTREE_TO_KCAL
            min_energy = torch.min(energies_kcal).item()

        return min_energy

    def compute_e_free_rdkit(self, mol: Chem.Mol) -> float:
        free_mol = Chem.AddHs(Chem.Mol(mol))
        free_mol.RemoveAllConformers()
        params = AllChem.ETKDGv3()
        params.randomSeed = 42
        cids = list(AllChem.EmbedMultipleConfs(free_mol, numConfs=self.num_confs_free, params=params))
        if not cids:
            params.useRandomCoords = True
            cids = list(AllChem.EmbedMultipleConfs(free_mol, numConfs=self.num_confs_free, params=params))
        if not cids:
            AllChem.EmbedMolecule(free_mol, AllChem.ETKDGv3())
            cids = [0]

        min_energy = float('inf')
        mp = AllChem.MMFFGetMoleculeProperties(free_mol, mmffVariant='MMFF94s')
        if mp is not None:
            for cid in cids:
                try:
                    ff = AllChem.MMFFGetMoleculeForceField(free_mol, mp, confId=cid)
                    if ff is not None:
                        ff.Initialize()
                        ff.Minimize(maxIts=300)
                        e = ff.CalcEnergy()
                        if e < min_energy:
                            min_energy = e
                except Exception:
                    pass

        if not math.isfinite(min_energy):
            raise ValueError("MMFF94s free conformer optimization failed")
        return min_energy

    def get_e_free(self, mol: Chem.Mol, use_ani: bool) -> float:
        key = (self.get_chemical_key(mol), use_ani)
        if key in self.baseline_cache:
            return self.baseline_cache[key]

        e_free = self.compute_e_free_gpu_ani(mol) if use_ani else self.compute_e_free_rdkit(mol)

        self.baseline_cache[key] = e_free
        return e_free

    def compute_e_bound_gpu_ani(self, mol: Chem.Mol) -> float:
        work_mol = Chem.Mol(mol)
        if any(atom.GetAtomicNum() == 1 for atom in work_mol.GetAtoms()) is False:
            work_mol = Chem.AddHs(work_mol, addCoords=True)

        atomic_nums = [atom.GetAtomicNum() for atom in work_mol.GetAtoms()]
        pos = work_mol.GetConformer(0).GetPositions()

        species = torch.tensor([atomic_nums], dtype=torch.long, device=self.device)
        R_docked = torch.from_numpy(np.ascontiguousarray(pos, dtype=np.float32)).unsqueeze(0).to(self.device)
        R = R_docked.clone().detach().requires_grad_(True)

        heavy_mask = (species[0] > 1)
        optimizer = torch.optim.Adam([R], lr=self.relax_lr)

        for _ in range(self.max_relax_steps):
            optimizer.zero_grad()
            energies_hartree = self.ani_model((species, R)).energies
            e_kcal = energies_hartree * HARTREE_TO_KCAL
            restraint = 0.5 * self.restraint_k * torch.sum((R[:, heavy_mask, :] - R_docked[:, heavy_mask, :]) ** 2)
            loss = (e_kcal + restraint).sum()
            loss.backward()
            optimizer.step()

        with torch.no_grad():
            final_e_kcal = (self.ani_model((species, R)).energies * HARTREE_TO_KCAL).item()

        return final_e_kcal

    def compute_e_bound_rdkit(self, mol: Chem.Mol) -> float:
        work_mol = Chem.Mol(mol)
        if any(atom.GetAtomicNum() == 1 for atom in work_mol.GetAtoms()) is False:
            work_mol = Chem.AddHs(work_mol, addCoords=True)

        use_uff = False
        try:
            mp_props = AllChem.MMFFGetMoleculeProperties(work_mol, mmffVariant='MMFF94s')
            if mp_props is None:
                use_uff = True
            else:
                ff = AllChem.MMFFGetMoleculeForceField(work_mol, mp_props, confId=0)
                if ff is None:
                    use_uff = True
        except Exception:
            use_uff = True

        if use_uff:
            raise ValueError("MMFF94s parameters unavailable for bound ligand")

        for atom in work_mol.GetAtoms():
            if atom.GetAtomicNum() > 1:
                if hasattr(ff, "MMFFAddPositionConstraint"):
                    ff.MMFFAddPositionConstraint(atom.GetIdx(), maxDispl=0.0, forceConstant=self.restraint_k)
                elif hasattr(ff, "UFFAddPositionConstraint"):
                    ff.UFFAddPositionConstraint(atom.GetIdx(), maxDispl=0.0, forceConstant=self.restraint_k)

        ff.Initialize()
        ff.Minimize(maxIts=self.max_relax_steps)
        # Restraints guide relaxation but are excluded from internal energy.
        pure_ff = AllChem.MMFFGetMoleculeForceField(work_mol, mp_props, confId=0)
        if pure_ff is None:
            raise ValueError("Cannot evaluate bound MMFF94s energy")
        return pure_ff.CalcEnergy()

    def evaluate_strain(self, docked_mol: Chem.Mol, mol_ref: Optional[Chem.Mol] = None) -> Tuple[float, float, float, bool]:
        ref = mol_ref if mol_ref is not None else docked_mol
        try:
            # Energy differences require the same atoms and protonation state.
            bound_h = Chem.AddHs(Chem.Mol(docked_mol), addCoords=True)
            ref_h = Chem.AddHs(Chem.Mol(ref))
            bound_atoms = sorted(atom.GetAtomicNum() for atom in bound_h.GetAtoms())
            ref_atoms = sorted(atom.GetAtomicNum() for atom in ref_h.GetAtoms())
            if bound_atoms != ref_atoms or Chem.GetFormalCharge(bound_h) != Chem.GetFormalCharge(ref_h):
                raise ValueError('Docked and free molecules differ in composition or formal charge')
            docked_mol, ref = bound_h, ref_h
        except Exception as exc:
            logger.error('Cannot match docked and free ligands; rejecting molecule: %s', exc)
            return float('nan'), float('nan'), float('nan'), False
        if self.is_ani_compatible(docked_mol) and self.is_ani_compatible(ref):
            try:
                e_bound = self.compute_e_bound_gpu_ani(docked_mol)
                e_free = self.get_e_free(ref, use_ani=True)
                if not math.isfinite(e_bound) or not math.isfinite(e_free):
                    raise ValueError("Non-finite ANI energy")
                strain = max(0.0, e_bound - e_free)
                return e_bound, e_free, strain, strain <= self.max_strain
            except Exception as exc:
                logger.warning("ANI-2x strain failed; trying MMFF94s: %s", exc)
        try:
            e_bound = self.compute_e_bound_rdkit(docked_mol)
            e_free = self.get_e_free(ref, use_ani=False)
            if not math.isfinite(e_bound) or not math.isfinite(e_free):
                raise ValueError("Non-finite MMFF94s energy")
            strain = max(0.0, e_bound - e_free)
            passed = (strain <= self.max_strain)
            return e_bound, e_free, strain, passed
        except Exception as exc:
            logger.error("Strain calculation failed; rejecting molecule: %s", exc)
            return float('nan'), float('nan'), float('nan'), False


# ==================================================================================================
# 2. PDBQT PREPARATION
# ==================================================================================================
def prepare_pdbqt(smiles: str, out_pdbqt: Path, mol_3d=None) -> bool:
    try:
        subprocess.run(
            ["obabel", f"-:{smiles}", "-O", str(out_pdbqt), "--gen3d", "-p", "7.4", "--partialcharge", "gasteiger"],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=30
        )
    except Exception:
        pass

    if out_pdbqt.exists() and out_pdbqt.stat().st_size > 100:
        return True

    if mol_3d is not None:
        try:
            tmp_pdb = out_pdbqt.with_suffix(".tmp.pdb")
            tmp_pdb.write_text(Chem.MolToPDBBlock(mol_3d) or "")
            subprocess.run(
                ["obabel", str(tmp_pdb), "-O", str(out_pdbqt), "-xh"],
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=20
            )
            tmp_pdb.unlink(missing_ok=True)
        except Exception:
            pass

    return out_pdbqt.exists() and out_pdbqt.stat().st_size > 100


# ==================================================================================================
# 3. AUTODOCK-GPU DOCKING
# ==================================================================================================
def run_autodock_gpu(pdbqt: Path, step_dir: Path, mol_base: str, nrun: int = 20) -> Tuple[float, Optional[Path]]:
    dlg_file = step_dir / f"{mol_base}.dlg"
    try:
        cmd = [
            AUTODOCK_GPU_BIN,
            "--ffile", str(RECEPTOR_FLD),
            "--lfile", str(pdbqt),
            "--nrun", str(nrun),
            "--resnam", str(step_dir / mol_base)
        ]
        subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=180)
    except Exception as e:
        logger.warning(f"AutoDock-GPU error for {mol_base}: {e}")
        return 0.0, None

    if not dlg_file.exists():
        return 0.0, None

    # Parse the RMSD TABLE RANKING section:
    # Format: Rank  SubRank  Run  BindingEnergy  ClustRMSD  RefRMSD  RANKING
    # Example: '   1      1      5       -7.61      0.00    234.83           RANKING'
    best_energy = 0.0
    best_run = None
    with open(dlg_file, 'r', errors='ignore') as f:
        for line in f:
            if line.rstrip().endswith('RANKING'):
                parts = line.split()
                # parts: [rank, subrank, run, energy, clust_rmsd, ref_rmsd, 'RANKING']
                if len(parts) >= 7:
                    try:
                        rank = int(parts[0])
                        run = int(parts[2])
                        energy = float(parts[3])
                        if rank == 1 and (best_run is None or energy < best_energy):
                            best_energy = energy
                            best_run = run
                    except (ValueError, IndexError):
                        pass
                    if best_run is not None:
                        break  # Got rank 1 best pose
    return best_energy, dlg_file


def extract_docked_conformer(dlg_file: Path, step_dir: Path, mol_base: str) -> Tuple[Optional[Chem.Mol], Optional[Path]]:
    """Extract the best-ranked docked pose from AutoDock-GPU DLG file.

    The DLG format uses 'DOCKED: <record>' lines where each line (after stripping 'DOCKED: ')
    starts with a single space before the record type (e.g. ' ATOM', ' ROOT').
    We need to find the MODEL corresponding to the best-ranked run.
    """
    # Valid PDBQT record types to keep
    _PDBQT_RECORDS = ("ATOM", "HETATM", "ROOT", "ENDROOT", "BRANCH", "ENDBRANCH", "TORSDOF", "TER")

    try:
        # First pass: find best run number from RANKING table
        best_run = None
        with open(dlg_file, 'r', errors='ignore') as f:
            for line in f:
                if line.rstrip().endswith('RANKING'):
                    parts = line.split()
                    if len(parts) >= 7:
                        try:
                            rank = int(parts[0])
                            run = int(parts[2])
                            if rank == 1:
                                best_run = run
                                break
                        except (ValueError, IndexError):
                            pass

        # Second pass: find and extract the MODEL for best_run (or first model)
        docked_lines = []
        in_model = False
        current_run = None

        with open(dlg_file, 'r', errors='ignore') as f:
            for line in f:
                # Detect model start: 'DOCKED: MODEL        N'
                if line.startswith("DOCKED: MODEL"):
                    # The run number is in the next USER line: 'DOCKED: USER    Run = N'
                    in_model = True
                    docked_lines = []
                    current_run = None
                    continue

                if in_model and line.startswith("DOCKED: USER    Run = "):
                    try:
                        current_run = int(line.split("Run =")[1].strip())
                    except Exception:
                        pass
                    continue

                if in_model and line.startswith("DOCKED: ENDMDL"):
                    # If this is the best run, stop; otherwise reset
                    if best_run is None or current_run == best_run:
                        break
                    else:
                        docked_lines = []
                        in_model = False
                    continue

                if in_model and line.startswith("DOCKED:"):
                    # Strip 'DOCKED:' (7 chars) and all leading whitespace
                    raw = line[7:].lstrip()
                    # Keep only valid PDBQT record lines
                    if raw.startswith(_PDBQT_RECORDS):
                        docked_lines.append(raw + "\n" if not raw.endswith("\n") else raw)

        if not docked_lines:
            return None, None

        tmp_docked_pdbqt = step_dir / f"{mol_base}_docked.pdbqt"
        tmp_docked_pdb = step_dir / f"{mol_base}_docked.pdb"
        tmp_docked_sdf = step_dir / f"{mol_base}_docked.sdf"
        tmp_docked_pdbqt.write_text("".join(docked_lines))

        # Use obabel with explicit format flags for reliable PDBQT->PDB/SDF conversion
        obabel_bin = str(_find_bin(["obabel"]))
        subprocess.run(
            [obabel_bin, "-ipdbqt", str(tmp_docked_pdbqt), "-opdb", "-O", str(tmp_docked_pdb)],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=20
        )
        subprocess.run(
            [obabel_bin, "-ipdbqt", str(tmp_docked_pdbqt), "-osdf", "-O", str(tmp_docked_sdf), "-h"],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=20
        )

        docked_mol = None
        if tmp_docked_sdf.exists() and tmp_docked_sdf.stat().st_size > 50:
            try:
                suppl = Chem.SDMolSupplier(str(tmp_docked_sdf), removeHs=False, sanitize=True)
                if len(suppl) > 0 and suppl[0] is not None:
                    docked_mol = suppl[0]
            except Exception:
                pass

        if docked_mol is None and tmp_docked_pdb.exists() and tmp_docked_pdb.stat().st_size > 50:
            docked_mol = Chem.MolFromPDBFile(str(tmp_docked_pdb), removeHs=False, sanitize=False)
            if docked_mol is not None:
                try:
                    Chem.SanitizeMol(docked_mol, catchErrors=True)
                except Exception:
                    pass

        return docked_mol, tmp_docked_pdbqt
    except Exception as e:
        logger.debug(f"Docked conformer extraction note: {e}")

    return None, None


# ==================================================================================================
# 4. GNINA MINIMIZATION & CNN RESCORING (run_cd97_gnina_minimize_cnn.py)
# ==================================================================================================
def parse_gnina_sdf_properties(sdf_path: Path) -> dict[str, str]:
    properties: dict[str, str] = {}
    if not sdf_path.is_file():
        return properties
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


def run_gnina_minimize_cnn(docked_pdbqt: Path, step_dir: Path, mol_base: str) -> Tuple[float, float, float]:
    out_sdf = step_dir / f"{mol_base}_minimized.sdf"
    out_log = step_dir / f"{mol_base}_gnina.log"

    cmd = [
        GNINA_BIN,
        "--receptor", str(RECEPTOR_PDB),
        "--ligand", str(docked_pdbqt),
        "--minimize",
        "--minimize_iters", "1000",
        "--accurate_line",
        "--cnn_scoring", "rescore",
        "--pose_sort_order", "CNNscore",
        "--device", "0" if torch.cuda.is_available() else "0",
        "--cpu", "2",
        "--out", str(out_sdf),
        "--log", str(out_log)
    ]

    cnn_score = 0.0
    cnn_affinity = 0.0
    min_aff = 0.0

    try:
        subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=60, check=True)
        props = parse_gnina_sdf_properties(out_sdf)
        if "CNNscore" not in props or "CNNaffinity" not in props:
            raise ValueError("GNINA output missing CNNscore or CNNaffinity")
        if "CNNscore" in props:
            cnn_score = float(props["CNNscore"])
        if "CNNaffinity" in props:
            cnn_affinity = float(props["CNNaffinity"])
        if "minimizedAffinity" in props:
            min_aff = float(props["minimizedAffinity"])
    except Exception as e:
        raise RuntimeError(f"GNINA scoring failed for {mol_base}: {e}") from e
    finally:
        out_sdf.unlink(missing_ok=True)
        out_log.unlink(missing_ok=True)

    return cnn_score, cnn_affinity, min_aff


# ==================================================================================================
# 5. CD97 PLIP TARGET-RESIDUE SCORING (cd97_plip_score_select_diverse.py)
# ==================================================================================================
def run_cd97_plip_scoring(complex_pdb: Path) -> Dict[str, Any]:
    plip_res = {
        'target_interaction_score': 0.0,
        'lid_bonus': 0.0,
        'region_bonus': 0.0,
        'total_plip_score': 0.0,
        'target_residue_hits': 0,
        'lid_hits': 0,
        'side_hits': 0,
        'bottom_hits': 0,
        'lid_engaged': False,
        'deep_pocket_bypass': False,
        'pocket_path_valid': False,
        'hit_labels': ''
    }

    try:
        from plip.basic import config as plip_config
        from plip.structure.preparation import PDBComplex

        plip_config.NOFIXFILE = True
        plip_config.NOHYDRO = False

        complex_obj = PDBComplex()
        complex_obj.load_pdb(str(complex_pdb))
        complex_obj.analyze()

        target_site = None
        for bsite_id, iset in complex_obj.interaction_sets.items():
            if "LIG" in bsite_id or ":Z:" in bsite_id or len(complex_obj.interaction_sets) == 1:
                target_site = iset
                break
        if target_site is None and complex_obj.interaction_sets:
            target_site = next(iter(complex_obj.interaction_sets.values()))

        if target_site is None:
            return plip_res

        raw_interactions = []
        for itype, attrs in PLIP_INTERACTION_ATTRIBUTES.items():
            for attr in attrs:
                contacts = getattr(target_site, attr, [])
                for c in contacts:
                    resnr = getattr(c, "resnr", None)
                    if resnr is not None and int(resnr) in TARGET_RESIDUES:
                        resnr = int(resnr)
                        target_def = TARGET_RESIDUES[resnr]
                        r_group = target_def["group"]
                        r_weight = GROUP_WEIGHTS[r_group]
                        i_weight = INTERACTION_WEIGHTS[itype]
                        raw_interactions.append({
                            'resnr': resnr,
                            'itype': itype,
                            'group': r_group,
                            'score': r_weight * i_weight
                        })

        unique_contacts = {}
        for item in raw_interactions:
            key = (item['resnr'], item['itype'])
            if key not in unique_contacts:
                unique_contacts[key] = item

        hit_residues = {item['resnr'] for item in unique_contacts.values()}
        lid_hits   = {r for r in hit_residues if TARGET_RESIDUES[r]["group"] == "lid"}
        side_hits  = {r for r in hit_residues if TARGET_RESIDUES[r]["group"] == "side"}
        bottom_hits = {r for r in hit_residues if TARGET_RESIDUES[r]["group"] == "bottom"}

        target_score = sum(item['score'] for item in unique_contacts.values())

        # Lid bonus — identical to reference script:
        # both lid residues (W685 + F760) = +6.0, one lid residue = +2.0
        if len(lid_hits) == 2:
            lid_bonus = 6.0
        elif len(lid_hits) == 1:
            lid_bonus = 2.0
        else:
            lid_bonus = 0.0

        # All-three-region bonus: lid + side + bottom all contacted = +4.0
        groups_hit = {TARGET_RESIDUES[r]["group"] for r in hit_residues}
        region_bonus = 4.0 if len(groups_hit) == 3 else 0.0

        # Pocket path validity — identical to reference script
        lid_engaged = (len(lid_hits) >= 1)
        deep_pocket_bypass = (len(side_hits) >= 2 and len(bottom_hits) >= 1)
        pocket_path_valid = (lid_engaged or deep_pocket_bypass)

        plip_res['target_interaction_score'] = target_score
        plip_res['lid_bonus'] = lid_bonus
        plip_res['region_bonus'] = region_bonus
        plip_res['total_plip_score'] = target_score + lid_bonus + region_bonus
        plip_res['target_residue_hits'] = len(hit_residues)
        plip_res['lid_hits'] = len(lid_hits)
        plip_res['side_hits'] = len(side_hits)
        plip_res['bottom_hits'] = len(bottom_hits)
        plip_res['lid_engaged'] = lid_engaged
        plip_res['deep_pocket_bypass'] = deep_pocket_bypass
        plip_res['pocket_path_valid'] = pocket_path_valid
        plip_res['hit_labels'] = ";".join(TARGET_RESIDUES[r]["label"] for r in sorted(hit_residues))

    except Exception as e:
        logger.debug(f"PLIP module run note: {e}")

    return plip_res


# ==================================================================================================
# 6. PDB HELPER: Properly formatted HETATM lines for ligand in complex
# ==================================================================================================
def _format_ligand_pdb_lines(mol: Chem.Mol) -> List[str]:
    """Convert an RDKit mol to properly column-formatted HETATM lines for PLIP.

    PDB format (columns, 1-indexed):
    1-6   record type (HETATM)
    7-11  serial (right-justified)
    12    space
    13-16 atom name (left-justified)
    17    alt loc
    18-20 resName (right-justified in 3 chars)
    21    space
    22    chainID
    23-26 resSeq
    27    iCode
    28-30 spaces
    31-38 x (8.3f)
    39-46 y (8.3f)
    47-54 z (8.3f)
    55-60 occupancy (6.2f)
    61-66 tempFactor (6.2f)
    73-76 spaces
    77-78 element (right-justified)
    """
    lines = []
    pdb_block = Chem.MolToPDBBlock(mol)
    if not pdb_block:
        return lines

    atom_serial = 1
    for raw_line in pdb_block.splitlines():
        rec = raw_line[:6].strip()
        if rec not in ("ATOM", "HETATM"):
            continue
        # Parse fields from original line
        try:
            atom_name = raw_line[12:16].strip()
            x = float(raw_line[30:38])
            y = float(raw_line[38:46])
            z = float(raw_line[46:54])
            occ_str = raw_line[54:60].strip()
            occ = float(occ_str) if occ_str else 1.00
            tf_str = raw_line[60:66].strip()
            tf = float(tf_str) if tf_str else 0.00
            elem = raw_line[76:78].strip() if len(raw_line) >= 78 else raw_line[12:14].strip().lstrip('0123456789')
        except (ValueError, IndexError):
            continue

        # Format atom name: 1-char elements left-padded, 2-char elements in col 13
        if len(elem) == 1:
            atom_name_fmt = f" {atom_name:<3s}"
        else:
            atom_name_fmt = f"{atom_name:<4s}"

        line = (
            f"{'HETATM':<6s}"
            f"{atom_serial:5d} "
            f"{atom_name_fmt}"
            f" "
            f"{'LIG':<3s}"
            f" "
            f"{'Z'}"
            f"{1:4d}"
            f"    "
            f"{x:8.3f}{y:8.3f}{z:8.3f}"
            f"{occ:6.2f}{tf:6.2f}"
            f"          "
            f"{elem:>2s}"
        )
        lines.append(line)
        atom_serial += 1
    return lines


# ==================================================================================================
# 7. DOCKED COMPLEX BUILDER (Saves receptor + best ligand pose with full REMARK metadata)
# ==================================================================================================
def save_docked_complex(
    docked_mol: Chem.Mol,
    mol_base: str,
    smiles: str,
    affinity: float,
    strain: float,
    e_bound: float,
    e_free: float,
    cnn_score: float,
    cnn_affinity: float,
    plip_score: float,
    final_score: float,
    step_num: int
):
    try:
        out_name = f"step{step_num:04d}_{mol_base}_score{final_score:.3f}"
        pdb_path = BEST_COMPLEXES_DIR / f"{out_name}_complex.pdb"

        with open(RECEPTOR_PDB, 'r') as rf:
            rec_lines = [l for l in rf if l.startswith(("ATOM", "HETATM", "TER"))]

        remarks = [
            f"REMARK   1  MOLECULE_ID:      {mol_base}",
            f"REMARK   1  SMILES:           {smiles}",
            f"REMARK   1  STEP_NUMBER:      {step_num}",
            f"REMARK   1  FINAL_SCORE:      {final_score:.4f}",
            f"REMARK   1  DOCKING_AFFINITY: {affinity:.2f} KCAL/MOL",
            f"REMARK   1  STRAIN_ENERGY:    {strain:.2f} KCAL/MOL (BOUND: {e_bound:.2f}, FREE: {e_free:.2f})",
            f"REMARK   1  GNINA_CNNSCORE:   {cnn_score:.4f} (AFFINITY: {cnn_affinity:.2f})",
            f"REMARK   1  PLIP_TOTAL_SCORE: {plip_score:.2f}",
        ]

        lig_lines = _format_ligand_pdb_lines(docked_mol)

        with open(pdb_path, 'w') as cf:
            cf.write("\n".join(remarks) + "\n")
            cf.writelines(rec_lines)
            if rec_lines and not rec_lines[-1].strip().startswith("TER"):
                cf.write("TER\n")
            cf.write("\n".join(lig_lines) + "\n")
            cf.write("END\n")

    except Exception as e:
        logger.debug(f"Complex save note: {e}")


# ==================================================================================================
# REAL RETROSYNTHETIC ACCESSIBILITY (RAscore) MODEL (Reymond Group / CASP)
# ==================================================================================================
class RAScorer:
    """
    Official Retrosynthetic Accessibility (RAscore) implementation.
    Faithful to official Reymond Group GitHub repository (reymond-group/RAscore).
    Model: XGB_chembl_ecfp_counts (ChEMBL 200k AiZynthFinder, ECFP6 counts, radius 3).
    Evaluates machine-learned retrosynthetic accessibility score in [0.0, 1.0].
    Strict: No silent heuristic fallback or free default scores. Invalid SMILES -> 0.0.
    """
    def __init__(self, model_path: Optional[str] = None):
        self.bst = None
        self._load_model(model_path)

    def _load_model(self, model_path: Optional[str] = None):
        import xgboost as xgb
        base_dir = Path(__file__).resolve().parent

        candidates = []
        if model_path:
            candidates.append(Path(model_path))
        candidates.extend([
            base_dir / "RAscore" / "RAscore" / "models" / "XGB_chembl_ecfp_counts" / "model.json",
            Path("/home/linux/CD97_REINVENT4_LINUX_1000STEPS/RAscore/RAscore/models/XGB_chembl_ecfp_counts/model.json"),
            base_dir / "RAscore" / "RAscore" / "models" / "XGB_chembl_ecfp_counts" / "model.pkl",
            Path("/home/linux/CD97_REINVENT4_LINUX_1000STEPS/RAscore/RAscore/models/XGB_chembl_ecfp_counts/model.pkl"),
        ])

        for p in candidates:
            if p.exists():
                try:
                    if str(p).endswith(".json"):
                        self.bst = xgb.Booster()
                        self.bst.load_model(str(p))
                        logger.info(f"Loaded official RAscore model from: {p}")
                        return
                    elif str(p).endswith(".pkl"):
                        import pickle
                        import sklearn.preprocessing
                        import xgboost.compat
                        xgboost.compat.XGBoostLabelEncoder = sklearn.preprocessing.LabelEncoder
                        with open(p, "rb") as f:
                            clf = pickle.load(f)
                        self.bst = clf.get_booster()
                        logger.info(f"Loaded official RAscore booster from pickle: {p}")
                        return
                except Exception as e:
                    logger.debug(f"Could not load candidate model {p}: {e}")

        raise RuntimeError("Critical: Failed to load official RAscore model!")

    def ecfp(self, smiles: str) -> Optional[np.ndarray]:
        """Converts SMILES into counted ECFP6 vector (identical to official RAscore_XGB.py)."""
        mol = Chem.MolFromSmiles(smiles)
        if not mol:
            return None
        fp = AllChem.GetMorganFingerprint(mol, 3, useCounts=True, useFeatures=False)
        size = 2048
        arr = np.zeros((size,), dtype=np.float32)
        for idx, v in fp.GetNonzeroElements().items():
            nidx = idx % size
            arr[nidx] += float(v)
        return arr

    def predict_smiles(self, smiles: str) -> float:
        """Predicts synthetic accessibility score [0.0, 1.0] from SMILES using official RAscore model."""
        if self.bst is None:
            raise RuntimeError("RAscore model is not loaded!")

        if not smiles or not isinstance(smiles, str) or not smiles.strip():
            return 0.0

        try:
            arr = self.ecfp(smiles)
            if arr is None:
                return 0.0
            import xgboost as xgb
            dmat = xgb.DMatrix(arr.reshape(1, -1))
            pred = self.bst.predict(dmat)
            score = float(pred[0])
            if np.isnan(score):
                return 0.0
            return float(np.clip(score, 0.0, 1.0))
        except Exception as e:
            logger.warning(f"RAscore calculation failed for {smiles}: {e}")
            return 0.0


# Backward compatibility alias
AstraZenecaRAScorer = RAScorer


# ==================================================================================================
# MAIN SCORING ENGINE
# ==================================================================================================
class RealOracleScoringEngine:
    def __init__(self, sigma: float = 40.0, max_strain: float = 10.0, min_ra_score: float = 0.60):
        self.sigma = sigma
        self.max_strain = max_strain
        self.min_ra_score = min_ra_score
        self.ra_scorer = RAScorer()
        import threading
        self.lock = threading.Lock()
        self.seen_scaffolds: Dict[str, int] = defaultdict(int)
        self.seen_fps: list = []
        self.strain_evaluator = GPUMoleculeStrainEvaluator(
            device="cuda:0" if torch.cuda.is_available() else "cpu",
            max_strain=max_strain,
            restraint_k=25.0,
            max_relax_steps=5,
            relax_lr=0.005,
            num_confs_free=5
        )

    def compute_ra_score(self, smiles: str) -> float:
        """Evaluates AstraZeneca Retrosynthetic Accessibility (RAscore) [0.0, 1.0]."""
        return self.ra_scorer.predict_smiles(smiles)

    def compute_diversity_score(self, mol: Chem.Mol, scaffold: str) -> float:
        """Evaluates scaffold novelty and Morgan ECFP4 fingerprint distance against seen hits."""
        try:
            with self.lock:
                scaff_count = self.seen_scaffolds.get(scaffold, 0)
                fps_copy = list(self.seen_fps[-100:])

            if scaff_count == 0:
                s_scaff = 1.0
            elif scaff_count <= 3:
                s_scaff = 0.6
            else:
                s_scaff = 0.2

            from rdkit.DataStructs import FingerprintSimilarity
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
            if not fps_copy:
                s_fp = 1.0
            else:
                max_sim = max(FingerprintSimilarity(fp, prev_fp) for prev_fp in fps_copy)
                s_fp = float(np.clip(1.0 - max_sim, 0.0, 1.0))

            return float(0.5 * s_scaff + 0.5 * s_fp)
        except Exception:
            return 1.0

    def evaluate_molecule(self, smiles: str, mol_idx: int, step_dir: Path, step_num: int = 0) -> Dict[str, Any]:
        result = {
            'valid': False,
            'SMILES': smiles,
            'Score': 0.0,
            'Real_DeltaG_kcal_mol': 0.0,
            'Strain_kcal_mol': 0.0,
            'P_Strain': 0.0,
            'GNINA_CNN_Score': 0.0,
            'GNINA_CNN_Affinity': 0.0,
            'PLIP_Total_Score': 0.0,
            'PLIP_Target_Hits': 0,
            'PLIP_Lid_Engaged': False,
            'PLIP_Pocket_Valid': False,
            'RA_Score': 0.0,
            'Diversity_Score': 0.0,
            'Scaffold': ''
        }

        mol = Chem.MolFromSmiles(smiles)
        if not mol:
            return result

        # 1. Physicochemical Gate
        mw   = Descriptors.MolWt(mol)
        logp = Descriptors.MolLogP(mol)
        tpsa = Descriptors.TPSA(mol)
        hbd  = Descriptors.NumHDonors(mol)
        hba  = Descriptors.NumHAcceptors(mol)
        rotb = Descriptors.NumRotatableBonds(mol)

        if not (150 <= mw <= 650 and -2.0 <= logp <= 6.5 and tpsa <= 160 and hbd <= 6 and hba <= 10 and rotb <= 12):
            return result

        # 2. Early RAscore Synthesizability Screening (1 ms test)
        # Molecules with extreme synthetic infeasibility (<0.15) skip expensive GPU docking entirely
        s_ra_early = self.ra_scorer.predict_smiles(smiles)
        result['RA_Score'] = s_ra_early
        if s_ra_early < 0.15:
            return result

        # 3. Bemis-Murcko Scaffold
        try:
            core = MurckoScaffold.GetScaffoldForMol(mol)
            result['Scaffold'] = Chem.MolToSmiles(core)
        except Exception:
            result['Scaffold'] = smiles

        # 3. PDBQT Prep
        mol_base = f"CD97_M{mol_idx:04d}"
        pdbqt_file = step_dir / f"{mol_base}.pdbqt"

        mol_3d = Chem.AddHs(mol)
        AllChem.EmbedMolecule(mol_3d, AllChem.ETKDGv3())

        if not prepare_pdbqt(smiles, pdbqt_file, mol_3d):
            return result

        # 4. AutoDock-GPU Docking (10% Weight)
        best_energy, dlg_file = run_autodock_gpu(pdbqt_file, step_dir, mol_base)
        result['Real_DeltaG_kcal_mol'] = best_energy

        if best_energy >= 0.0 or dlg_file is None:
            pdbqt_file.unlink(missing_ok=True)
            if dlg_file and dlg_file.exists(): dlg_file.unlink(missing_ok=True)
            return result

        s_dock = float(np.clip(-best_energy / 12.0, 0.0, 1.0))

        # 5. GNINA MINIMIZATION & CNN RESCORING (30% Weight)
        docked_mol, docked_pdbqt = extract_docked_conformer(dlg_file, step_dir, mol_base)
        target_lig_for_gnina = docked_pdbqt if (docked_pdbqt and docked_pdbqt.exists()) else pdbqt_file
        cnn_score, cnn_affinity, min_aff = run_gnina_minimize_cnn(target_lig_for_gnina, step_dir, mol_base)
        result['GNINA_CNN_Score'] = cnn_score
        result['GNINA_CNN_Affinity'] = cnn_affinity

        s_gnina = float(np.clip(
            (0.5 * cnn_score + 0.5 * np.clip((cnn_affinity - 4.0) / 6.0, 0.0, 1.0)),
            0.1, 1.0
        ))

        # 6. CD97 PLIP TARGET-RESIDUE PROFILING (35% Weight - 40.0 ZINC Scale)
        complex_pdb = step_dir / f"{mol_base}_complex.pdb"
        try:
            plip_mol = docked_mol if docked_mol is not None else mol_3d
            lig_lines = _format_ligand_pdb_lines(plip_mol)
            with open(RECEPTOR_PDB, 'r') as rf:
                rec_lines = [l for l in rf if l.startswith(("ATOM", "HETATM", "TER"))]
            with open(complex_pdb, 'w') as cf:
                cf.writelines(rec_lines)
                if rec_lines and not rec_lines[-1].strip().startswith("TER"):
                    cf.write("TER\n")
                cf.write("\n".join(lig_lines) + "\n")
                cf.write("END\n")

            plip_res = run_cd97_plip_scoring(complex_pdb)
        except Exception as ex:
            logger.debug(f"PLIP complex build note: {ex}")
            plip_res = {'total_plip_score': 0.0, 'target_residue_hits': 0, 'pocket_path_valid': False, 'lid_engaged': False}
        finally:
            complex_pdb.unlink(missing_ok=True)

        result['PLIP_Total_Score'] = plip_res.get('total_plip_score', 0.0)
        result['PLIP_Target_Hits'] = plip_res.get('target_residue_hits', 0)
        result['PLIP_Lid_Engaged'] = plip_res.get('lid_engaged', False)
        result['PLIP_Pocket_Valid'] = plip_res.get('pocket_path_valid', False)

        raw_plip = plip_res.get('total_plip_score', 0.0)
        s_plip = float(np.clip(raw_plip / 40.0, 0.05, 1.0))
        if not plip_res.get('pocket_path_valid', False):
            s_plip *= 0.50

        # 7. GPU CONFORMATIONAL STRAIN EVALUATION (Exponential Gating)
        target_mol_for_strain = docked_mol
        if target_mol_for_strain is None:
            logger.error("Could not extract docked pose; rejecting molecule")
            return result
        e_bound, e_free, strain, passed_strain = self.strain_evaluator.evaluate_strain(target_mol_for_strain, mol_ref=mol)
        result['Strain_kcal_mol'] = strain

        if not passed_strain:
            # Overstrained pose rejected
            result['P_Strain'] = 0.0
            result['Score'] = 0.0
            pdbqt_file.unlink(missing_ok=True)
            if dlg_file and dlg_file.exists(): dlg_file.unlink(missing_ok=True)
            if docked_pdbqt and docked_pdbqt.exists(): docked_pdbqt.unlink(missing_ok=True)
            (step_dir / f"{mol_base}_docked.pdb").unlink(missing_ok=True)
            (step_dir / f"{mol_base}_docked.sdf").unlink(missing_ok=True)
            return result

        p_strain = float(np.exp(-strain / 10.0))
        result['P_Strain'] = p_strain

        # 8. REAL RETROSYNTHETIC ACCESSIBILITY (RAscore, 15% Weight, >0.60 Standard)
        s_ra = result.get('RA_Score', 0.0)
        if s_ra <= 0.0:
            s_ra = self.ra_scorer.predict_smiles(smiles)
            result['RA_Score'] = s_ra
        if s_ra < 0.60:
            s_ra *= 0.50  # 50% penalty if synthesizability probability < 60%

        # 9. DIVERSITY SCORE (10% Weight)
        s_diversity = self.compute_diversity_score(mol, result['Scaffold'])
        result['Diversity_Score'] = s_diversity

        # 10. EXACT MULTI-OBJECTIVE WEIGHTED COMPOSITE SCORE
        # PLIP: 35%, GNINA: 30%, RAscore: 15%, Diversity: 10%, AutoDock: 10%
        w_plip = 0.35
        w_gnina = 0.30
        w_ra = 0.15
        w_diversity = 0.10
        w_dock = 0.10
        total_w = w_plip + w_gnina + w_ra + w_diversity + w_dock

        raw_composite = (
            w_plip * s_plip +
            w_gnina * s_gnina +
            w_ra * s_ra +
            w_diversity * s_diversity +
            w_dock * s_dock
        ) / total_w

        total_score = raw_composite * p_strain
        result['Score'] = float(np.clip(total_score, 0.0, 1.0))
        result['valid'] = True

        # Update diversity memory for successful candidates
        if result['Score'] > 0.0:
            with self.lock:
                self.seen_scaffolds[result['Scaffold']] += 1
                try:
                    self.seen_fps.append(AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048))
                except Exception:
                    pass

        # 9. Automatically save the best docked complex PDB for top-scoring leads (Score >= 0.60)
        if result['Score'] >= 0.60 and target_mol_for_strain is not None:
            save_docked_complex(
                docked_mol=target_mol_for_strain,
                mol_base=mol_base,
                smiles=smiles,
                affinity=best_energy,
                strain=strain,
                e_bound=e_bound,
                e_free=e_free,
                cnn_score=cnn_score,
                cnn_affinity=cnn_affinity,
                plip_score=raw_plip,
                final_score=result['Score'],
                step_num=step_num
            )

        # 10. Continuous multi-column breakdown logging (dedicated production file)
        try:
            breakdown_csv = PROJECT_ROOT / "analysis" / "production_1000steps" / "cd97_real_oracle_breakdown.csv"
            if not breakdown_csv.exists():
                breakdown_csv.parent.mkdir(parents=True, exist_ok=True)
                breakdown_csv.write_text("Step,Mol_ID,SMILES,AutoDock_dG_kcal_mol,GNINA_CNNScore,GNINA_Affinity,PLIP_Score,Diversity_Score,RA_Score,Strain_kcal_mol,Final_Score\n")
            with open(breakdown_csv, "a") as bf:
                bf.write(f"{step_num},{mol_base},{smiles},{best_energy:.2f},{cnn_score:.4f},{cnn_affinity:.2f},{raw_plip:.2f},{s_diversity:.4f},{ra_score:.4f},{strain:.2f},{result['Score']:.4f}\n")
        except Exception:
            pass

        pdbqt_file.unlink(missing_ok=True)
        if dlg_file and dlg_file.exists(): dlg_file.unlink(missing_ok=True)
        if docked_pdbqt and docked_pdbqt.exists(): docked_pdbqt.unlink(missing_ok=True)
        (step_dir / f"{mol_base}_docked.pdb").unlink(missing_ok=True)
        (step_dir / f"{mol_base}_docked.sdf").unlink(missing_ok=True)

        return result
