#!/usr/bin/env python3
"""
GPU-Accelerated Batch PDBQT Molecule Strain Filter & PDB Exporter
=================================================================

Evaluates ligand conformational reorganization strain:
    Delta_E_strain = E_bound (mild relaxation) - E_free (solution global minimum)

Hardware Engine:
    - PyTorch CUDA when available; CPU otherwise
    - TorchANI ANI-2x Neural Network Potential
    - Autograd Restrained Relaxation directly as PyTorch CUDA tensors
    - Batched Conformer Free Energy Evaluation on GPU
    - MMFF94s is used for both energies if ANI is unavailable or fails;
      molecules without valid MMFF94s parameters are reported as errors

AutoDock-GPU PDBQT Parsing:
    - Purpose-built Meeko & OpenBabel dual-engine reader for AutoDock-GPU format
    - Extracts AutoDock-GPU binding affinity from REMARK headers
    - Outputs standard PDB files (.pdb) with REMARK metadata
    - Generates real-time batch success and error logs
"""

import os
import sys
import time
import argparse
import logging
import json
import re
from typing import Dict, Tuple, Optional, List, Generator
from dataclasses import dataclass

# Force UTF-8 encoding across Windows and Linux terminals
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import torch

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import RDLogger

# Suppress minor RDKit noise
RDLogger.DisableLog('rdApp.warning')
RDLogger.DisableLog('rdApp.error')

# Suppress TorchANI extension compiler warning
os.environ['TORCHANI_NO_WARN_EXTENSIONS'] = '1'

HARTREE_TO_KCAL = 627.5094740631
ANI2X_SUPPORTED_Z = {1, 6, 7, 8, 9, 16, 17}  # H, C, N, O, F, S, Cl

# AutoDock-GPU & Schrödinger energy remark regexes
REMARK_AD_ENERGY_REGEX = re.compile(r"REMARK\s+AUTODOCK_BINDING_ENERGY_KCAL_MOL\s+([-\d.]+)", re.IGNORECASE)
REMARK_SCHRODINGER_REGEX = re.compile(r"REMARK\s+SCHRODINGER_HTVS_DOCKING_SCORE\s+([-\d.]+)", re.IGNORECASE)
REMARK_PIPELINE_LIG_REGEX = re.compile(r"REMARK\s+PIPELINE_LIGAND_ID\s+(\S+)", re.IGNORECASE)
REMARK_VINA_REGEX = re.compile(r"REMARK\s+VINA\s+RESULT:\s+([-\d.]+)", re.IGNORECASE)
REMARK_ENERGY_REGEX = re.compile(r"REMARK\s+.*ENERGY\s*=\s*([-\d.]+)", re.IGNORECASE)
USER_ENERGY_REGEX = re.compile(r"USER\s+Estimated\s+Free\s+Energy\s+of\s+Binding\s+=\s+([-\d.]+)", re.IGNORECASE)


@dataclass
class GPUMoleculeStrainResult:
    mol_idx: int
    mol_name: str
    batch_name: str
    affinity: Optional[float]
    smiles: str
    e_bound: float
    e_free: float
    strain: float
    passed: bool
    engine: str
    pdb_block: Optional[str] = None
    error: Optional[str] = None
    status_detail: str = ""


class GPUMoleculeStrainEvaluator:
    """
    GPU-accelerated strain evaluator for AutoDock-GPU PDBQT files using PyTorch CUDA & TorchANI.
    """

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
        """Initializes TorchANI ANI-2x on the RTX 4050 GPU."""
        try:
            import torchani
            logger = logging.getLogger("GPUEvaluator")
            logger.info(f"Loading TorchANI ANI-2x neural network potential onto {self.device}...")
            self.ani_model = torchani.models.ANI2x(periodic_table_index=True).to(self.device).eval()
            logger.info("TorchANI ANI-2x successfully loaded on GPU.")
        except Exception as e:
            logging.getLogger("GPUEvaluator").warning(f"TorchANI initialization note: {e}")
            self.ani_model = None

    def extract_autodock_affinity(self, pdbqt_str: str, filename: str = "") -> Optional[float]:
        """Extracts binding score from AutoDock-GPU headers or filename (e.g. _E-5.78)."""
        if filename:
            m_fn = re.search(r"_E([-\d.]+)", filename)
            if m_fn:
                try:
                    return float(m_fn.group(1))
                except ValueError:
                    pass

        m_ad = REMARK_AD_ENERGY_REGEX.search(pdbqt_str)
        if m_ad:
            try:
                return float(m_ad.group(1))
            except ValueError:
                pass
        m_schrod = REMARK_SCHRODINGER_REGEX.search(pdbqt_str)
        if m_schrod:
            try:
                return float(m_schrod.group(1))
            except ValueError:
                pass
        m = REMARK_VINA_REGEX.search(pdbqt_str)
        if m:
            try:
                return float(m.group(1))
            except ValueError:
                pass
        m2 = USER_ENERGY_REGEX.search(pdbqt_str)
        if m2:
            try:
                return float(m2.group(1))
            except ValueError:
                pass
        m3 = REMARK_ENERGY_REGEX.search(pdbqt_str)
        if m3:
            try:
                return float(m3.group(1))
            except ValueError:
                pass
        return None

    def pdbqt_to_rdkit_mol(self, pdbqt_str: str) -> Tuple[Optional[Chem.Mol], str]:
        """Converts AutoDock-GPU PDBQT text into 3D molecule with correct bond orders."""
        # Method 1: Meeko (developed by AutoDock team)
        try:
            from meeko import PDBQTMolecule, RDKitMolCreate
            pdbqt_mol = PDBQTMolecule(pdbqt_str)
            mols = RDKitMolCreate.from_pdbqt_mol(pdbqt_mol)
            if mols and len(mols) > 0 and mols[0] is not None:
                mol = mols[0]
                if mol.GetNumConformers() > 0:
                    return mol, "Meeko"
        except Exception:
            pass

        # Method 2: OpenBabel fallback
        try:
            from openbabel import pybel
            ob_mol = pybel.readstring('pdbqt', pdbqt_str)
            pdb_str = ob_mol.write('pdb')
            mol = Chem.MolFromPDBBlock(pdb_str, removeHs=False, sanitize=False)
            if mol is not None:
                mol.UpdatePropertyCache(strict=False)
                Chem.GetSSSR(mol)
                return mol, "OpenBabel"
        except Exception:
            pass

        return None, "None"

    def is_ani_compatible(self, mol: Chem.Mol) -> bool:
        """Checks if all atoms in molecule are supported by TorchANI ANI-2x (H, C, N, O, F, S, Cl)."""
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
        """Computes unique chemical identifier for baseline caching."""
        try:
            return Chem.MolToInchiKey(mol)
        except Exception:
            try:
                return Chem.MolToSmiles(Chem.RemoveHs(mol), isomericSmiles=True)
            except Exception:
                return mol.GetProp("_Name") if mol.HasProp("_Name") else "UNKNOWN"

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

    def compute_e_bound_gpu_ani(self, mol: Chem.Mol) -> Tuple[float, np.ndarray]:
        """
        Executes mild local relaxation on GPU using PyTorch CUDA autograd
        with heavy-atom harmonic positional restraints.
        """
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

        relaxed_coords = R.detach().cpu().numpy()[0]
        return final_e_kcal, relaxed_coords

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

    def build_pdb_block_with_remarks(
        self,
        mol: Chem.Mol,
        mol_idx: int,
        mol_name: str,
        batch_name: str,
        affinity: Optional[float],
        strain: float,
        e_bound: float,
        e_free: float,
        passed: bool,
        engine: str
    ) -> str:
        """Builds standard PDB text block with strain & affinity metadata in REMARK headers."""
        base_pdb = Chem.MolToPDBBlock(mol)

        remarks = []
        remarks.append(f"REMARK   1  MOLECULE_INDEX:   {mol_idx}")
        remarks.append(f"REMARK   1  MOLECULE_NAME:    {mol_name}")
        remarks.append(f"REMARK   1  BATCH_SOURCE:     {batch_name}")
        if affinity is not None:
            remarks.append(f"REMARK   1  DOCKING_AFFINITY: {affinity:.2f} KCAL/MOL")
        remarks.append(f"REMARK   1  STRAIN_ENERGY:    {strain:.2f} KCAL/MOL (THRESHOLD <= {self.max_strain:.1f})")
        remarks.append(f"REMARK   1  E_BOUND_RELAXED:  {e_bound:.2f} KCAL/MOL")
        remarks.append(f"REMARK   1  E_FREE_GLOBAL_MIN:{e_free:.2f} KCAL/MOL")
        remarks.append(f"REMARK   1  COMPUTE_ENGINE:   {engine}")
        remarks.append(f"REMARK   1  STRAIN_STATUS:    {'PASSED_LOW_STRAIN' if passed else 'REJECTED_STRAINED'}")

        header = "\n".join(remarks) + "\n"
        return header + base_pdb

    def evaluate_molecule(self, pdbqt_str: str, mol_idx: int, default_name: str, batch_name: str) -> GPUMoleculeStrainResult:
        """Evaluates a single AutoDock-GPU PDBQT molecule."""
        affinity = self.extract_autodock_affinity(pdbqt_str, filename=default_name)
        mol, parse_engine = self.pdbqt_to_rdkit_mol(pdbqt_str)

        if mol is None:
            return GPUMoleculeStrainResult(
                mol_idx=mol_idx,
                mol_name=default_name,
                batch_name=batch_name,
                affinity=affinity,
                smiles="INVALID",
                e_bound=0.0,
                e_free=0.0,
                strain=999.9,
                passed=False,
                engine="PARSER_FAIL",
                error="PDBQT Parsing Failed: Could not build 3D molecule.",
                status_detail="PARSER_FAILURE"
            )

        m_id = REMARK_PIPELINE_LIG_REGEX.search(pdbqt_str)
        if m_id:
            mol_name = m_id.group(1)
        elif mol.HasProp("_Name") and mol.GetProp("_Name"):
            mol_name = mol.GetProp("_Name")
        else:
            mol_name = default_name
        try:
            smiles = Chem.MolToSmiles(Chem.RemoveHs(mol))
        except Exception:
            smiles = "N/A"

        use_ani = self.is_ani_compatible(mol)
        engine_name = "RDKit_MMFF94s"

        try:
            mol = Chem.AddHs(mol, addCoords=True)
            if use_ani:
                try:
                    e_bound, _ = self.compute_e_bound_gpu_ani(mol)
                    e_free = self.get_e_free(mol, use_ani=True)
                    if not np.isfinite(e_bound) or not np.isfinite(e_free):
                        raise ValueError("Non-finite ANI energy")
                    engine_name = f"TorchANI_ANI2x ({self.device})"
                except Exception as exc:
                    logging.getLogger("GPUEvaluator").warning("ANI failed; recomputing both energies with MMFF94s: %s", exc)
                    e_bound = self.compute_e_bound_rdkit(mol)
                    e_free = self.get_e_free(mol, use_ani=False)
            else:
                e_bound = self.compute_e_bound_rdkit(mol)
                e_free = self.get_e_free(mol, use_ani=False)
            if not np.isfinite(e_bound) or not np.isfinite(e_free):
                raise RuntimeError("Non-finite ligand energy")
            strain = max(0.0, e_bound - e_free)
            passed = strain <= self.max_strain

            pdb_block = self.build_pdb_block_with_remarks(
                mol, mol_idx, mol_name, batch_name, affinity, strain, e_bound, e_free, passed, engine_name
            )

            status_str = "PASSED_LOW_STRAIN" if passed else f"REJECTED_HIGH_STRAIN_{strain:.2f}_KCAL"

            return GPUMoleculeStrainResult(
                mol_idx=mol_idx,
                mol_name=mol_name,
                batch_name=batch_name,
                affinity=affinity,
                smiles=smiles,
                e_bound=e_bound,
                e_free=e_free,
                strain=strain,
                passed=passed,
                engine=engine_name,
                pdb_block=pdb_block,
                status_detail=status_str
            )

        except Exception as err:
            return GPUMoleculeStrainResult(
                mol_idx=mol_idx,
                mol_name=mol_name,
                batch_name=batch_name,
                affinity=affinity,
                smiles=smiles,
                e_bound=0.0,
                e_free=0.0,
                strain=999.9,
                passed=False,
                engine="ERROR",
                error=f"Energy Minimization / Evaluation Exception: {str(err)}",
                status_detail="ENERGY_CALC_FAILURE"
            )


def stream_batch_molecules(batch_dir: str, batch_name: str) -> Generator[Tuple[str, int, str, str], None, None]:
    """Streams PDBQT molecules from batch folder."""
    mol_idx = 0
    pdbqt_files = [f for f in os.listdir(batch_dir) if f.lower().endswith('.pdbqt')]
    pdbqt_files.sort()

    for fname in pdbqt_files:
        fpath = os.path.join(batch_dir, fname)
        mol_name = os.path.splitext(fname)[0]
        try:
            with open(fpath, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            if content.strip():
                mol_idx += 1
                yield (content, mol_idx, mol_name, batch_name)
        except Exception:
            mol_idx += 1
            yield ("", mol_idx, mol_name, batch_name)


def discover_batches(input_path: str) -> List[Tuple[str, str]]:
    """Discovers batch directories."""
    if not os.path.isdir(input_path):
        return []

    sub_dirs = [d for d in os.listdir(input_path) if os.path.isdir(os.path.join(input_path, d))]
    batch_subdirs = [d for d in sub_dirs if "batch" in d.lower()]

    if batch_subdirs:
        batch_subdirs.sort()
        return [(d, os.path.join(input_path, d)) for d in batch_subdirs]
    elif any(f.lower().endswith('.pdbqt') for f in os.listdir(input_path)):
        base = os.path.basename(os.path.abspath(input_path))
        return [(base, input_path)]
    elif sub_dirs:
        sub_dirs.sort()
        valid = []
        for d in sub_dirs:
            p = os.path.join(input_path, d)
            if any(f.lower().endswith('.pdbqt') for f in os.listdir(p)):
                valid.append((d, p))
        return valid

    return []


def run_gpu_batch_pipeline(
    input_path: str,
    output_dir: str,
    device: str = "cuda:0",
    max_strain: float = 10.0,
    restraint_k: float = 25.0,
    max_relax_steps: int = 5,
    relax_lr: float = 0.005,
    num_confs_free: int = 5,
    split_pdb: bool = False,
    sort_by: str = "affinity",
    resume: bool = True,
    max_mols: Optional[int] = None
):
    """
    Executes GPU batch strain filtering pipeline.
    """
    main_logger = logging.getLogger("GPUBatchPipeline")
    main_logger.setLevel(logging.INFO)
    if not main_logger.handlers:
        ch = logging.StreamHandler(sys.stdout)
        ch.setFormatter(logging.Formatter('[%(asctime)s] [%(levelname)s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S'))
        main_logger.addHandler(ch)

    batches = discover_batches(input_path)
    if not batches:
        main_logger.error(f"No batch directories or .pdbqt files found in: {input_path}")
        sys.exit(1)

    device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"

    os.makedirs(output_dir, exist_ok=True)
    master_csv = os.path.join(output_dir, "ALL_BATCHES_SUMMARY_METRICS.csv")
    master_succ_log = os.path.join(output_dir, "master_success.log")
    master_err_log = os.path.join(output_dir, "master_error.log")

    main_logger.info("=" * 80)
    main_logger.info("GPU-ACCELERATED AUTODOCK-GPU MOLECULE STRAIN FILTER & PDB EXPORTER")
    main_logger.info("=" * 80)
    main_logger.info(f"Hardware Compute Engine  : {device_name} (PyTorch {torch.__version__}, Device: {device})")
    main_logger.info(f"Root Input Path          : {input_path}")
    main_logger.info(f"Discovered Batches       : {len(batches)} batches")
    main_logger.info(f"Output Directory         : {output_dir}")
    main_logger.info(f"PDB Export Format        : Individual .pdb files per passed molecule")
    main_logger.info(f"Max Strain Threshold     : {max_strain:.1f} kcal/mol")
    main_logger.info(f"Positional Restraint (k) : {restraint_k:.1f} kcal/(mol*A^2)")
    main_logger.info(f"GPU Relaxation Steps     : {max_relax_steps} steps (lr={relax_lr})")
    main_logger.info(f"Free Minimum Ensemble    : {num_confs_free} conformers (GPU Batched)")
    main_logger.info(f"Resume Mode              : {'ENABLED' if resume else 'DISABLED'}")
    main_logger.info("=" * 80)

    evaluator = GPUMoleculeStrainEvaluator(
        device=device,
        max_strain=max_strain,
        restraint_k=restraint_k,
        max_relax_steps=max_relax_steps,
        relax_lr=relax_lr,
        num_confs_free=num_confs_free
    )

    grand_total = 0
    grand_passed = 0
    grand_rejected = 0
    grand_errors = 0
    start_time_all = time.time()
    all_passed_mols_global = []

    f_master_succ = open(master_succ_log, 'w', encoding='utf-8')
    f_master_err = open(master_err_log, 'w', encoding='utf-8')
    f_master_csv = open(master_csv, 'w', encoding='utf-8')

    f_master_succ.write("=" * 80 + "\nGLOBAL MASTER SUCCESS LOG - PASSED LOW-STRAIN MOLECULES\n" + "=" * 80 + "\n")
    f_master_err.write("=" * 80 + "\nGLOBAL MASTER ERROR LOG - REJECTED STRAINED MOLECULES\n" + "=" * 80 + "\n")
    f_master_csv.write("mol_idx,mol_name,batch,docking_affinity,smiles,e_bound,e_free,strain_kcal_mol,passed,engine,status,error\n")

    try:
        for b_idx, (b_name, b_dir) in enumerate(batches, 1):
            batch_out_dir = os.path.join(output_dir, b_name)
            batch_csv = os.path.join(batch_out_dir, f"{b_name}_metrics.csv")

            if resume and os.path.isfile(batch_csv) and os.path.getsize(batch_csv) > 100:
                main_logger.info(f"[{b_idx}/{len(batches)}] Skipping {b_name} (Already completed, metrics exist).")
                try:
                    with open(batch_csv, 'r', encoding='utf-8') as fc:
                        lines = fc.readlines()[1:]
                        grand_total += len(lines)
                        for line in lines:
                            f_master_csv.write(line)
                            parts = line.split(',')
                            if len(parts) >= 9 and parts[8] == '1':
                                grand_passed += 1
                            else:
                                grand_rejected += 1
                except Exception:
                    pass
                continue

            main_logger.info(f"\n>>> Starting GPU Batch [{b_idx}/{len(batches)}]: {b_name} ({b_dir})")

            os.makedirs(batch_out_dir, exist_ok=True)
            succ_log_path = os.path.join(batch_out_dir, f"{b_name}_success.log")
            err_log_path = os.path.join(batch_out_dir, f"{b_name}_error.log")
            split_dir_path = os.path.join(batch_out_dir, "passed_pdbs")
            os.makedirs(split_dir_path, exist_ok=True)

            f_succ = open(succ_log_path, 'w', encoding='utf-8')
            f_err = open(err_log_path, 'w', encoding='utf-8')
            f_csv = open(batch_csv, 'w', encoding='utf-8')

            f_succ.write("=" * 80 + "\n")
            f_succ.write(f"BATCH: {b_name} - GPU SUCCESS LOG (LOW-STRAIN PASSED MOLECULES)\nStarted: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f_succ.write("=" * 80 + "\n")

            f_err.write("=" * 80 + "\n")
            f_err.write(f"BATCH: {b_name} - GPU ERROR & REJECTION LOG\nStarted: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f_err.write("=" * 80 + "\n")

            f_csv.write("mol_idx,mol_name,batch,docking_affinity,smiles,e_bound,e_free,strain_kcal_mol,passed,engine,status,error\n")

            b_start = time.time()
            total_mols = 0
            passed_mols = 0
            rejected_mols = 0
            error_mols = 0
            batch_passed_results = []

            for content, m_idx, m_name, b_label in stream_batch_molecules(b_dir, b_name):
                if max_mols is not None and total_mols >= max_mols:
                    main_logger.info(f"Reached requested limit of {max_mols} molecules for test run.")
                    break
                total_mols += 1
                now_str = time.strftime('%Y-%m-%d %H:%M:%S')

                res = evaluator.evaluate_molecule(content, m_idx, m_name, b_label)

                if res.error:
                    error_mols += 1
                    rejected_mols += 1
                    f_err.write(f"[{now_str}] [ERROR] Molecule #{res.mol_idx:<5} | {res.mol_name}: {res.error}\n")
                elif res.passed:
                    passed_mols += 1
                    batch_passed_results.append(res)
                    aff_text = f"{res.affinity:.2f} kcal/mol" if res.affinity is not None else "N/A"
                    f_succ.write(
                        f"[{now_str}] [PASSED] Molecule #{res.mol_idx:<5} | Name: {res.mol_name:<25} | "
                        f"Affinity: {aff_text:<12} | Strain: {res.strain:>6.2f} kcal/mol | "
                        f"E_bound: {res.e_bound:>8.2f} | E_free: {res.e_free:>8.2f} | Engine: {res.engine}\n"
                    )
                    # Immediately write individual PDB file for passed molecule
                    if res.pdb_block:
                        single_pdb_path = os.path.join(split_dir_path, f"{res.mol_name}.pdb")
                        with open(single_pdb_path, 'w', encoding='utf-8') as f_indiv:
                            f_indiv.write(res.pdb_block)
                else:
                    rejected_mols += 1
                    aff_text = f"{res.affinity:.2f} kcal/mol" if res.affinity is not None else "N/A"
                    f_err.write(
                        f"[{now_str}] [REJECTED_STRAINED] Molecule #{res.mol_idx:<5} | Name: {res.mol_name:<25} | "
                        f"Affinity: {aff_text:<12} | Strain: {res.strain:>6.2f} kcal/mol "
                        f"(> {max_strain:.1f} kcal/mol) | E_bound: {res.e_bound:>8.2f} | E_free: {res.e_free:>8.2f}\n"
                    )

                aff_str = f"{res.affinity:.2f}" if res.affinity is not None else ""
                err_clean = f'"{res.error}"' if res.error else ""
                csv_line = (
                    f"{res.mol_idx},\"{res.mol_name}\",\"{res.batch_name}\",{aff_str},\"{res.smiles}\","
                    f"{res.e_bound:.2f},{res.e_free:.2f},{res.strain:.2f},"
                    f"{1 if res.passed else 0},\"{res.engine}\",\"{res.status_detail}\",{err_clean}\n"
                )
                f_csv.write(csv_line)
                f_master_csv.write(csv_line)
                f_csv.flush()
                f_master_csv.flush()
                f_succ.flush()
                f_err.flush()

                # Progress logging every 20 molecules
                if total_mols % 20 == 0:
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    elapsed_m = time.time() - b_start
                    rate_m = total_mols / elapsed_m if elapsed_m > 0 else 0
                    main_logger.info(
                        f"GPU Progress [{b_name}]: {total_mols:,} mols | "
                        f"Passed: {passed_mols:,} ({(passed_mols/total_mols)*100:.1f}%) | "
                        f"Speed: {rate_m:.1f} mols/sec | Device: {device_name}"
                    )

            b_elapsed = time.time() - b_start
            b_rate = total_mols / b_elapsed if b_elapsed > 0 else 0

            f_succ.write("\n" + "=" * 80 + "\n")
            f_succ.write(f"BATCH {b_name} GPU SUMMARY:\n")
            f_succ.write(f"Total Evaluated  : {total_mols:,}\n")
            f_succ.write(f"Passed Low-Strain: {passed_mols:,} ({(passed_mols/max(1, total_mols))*100:.2f}%)\n")
            f_succ.write(f"Time Taken       : {b_elapsed:.2f} s ({b_rate:.1f} mols/sec)\n")
            f_succ.write(f"Passed PDBs Dir  : {split_dir_path}\n")
            f_succ.write("=" * 80 + "\n")

            f_err.write("\n" + "=" * 80 + "\n")
            f_err.write(f"BATCH {b_name} REJECTION SUMMARY:\n")
            f_err.write(f"Strained Mols    : {rejected_mols - error_mols:,}\n")
            f_err.write(f"Parser Errors    : {error_mols:,}\n")
            f_err.write("=" * 80 + "\n")

            f_succ.close()
            f_err.close()
            f_csv.close()

            grand_total += total_mols
            grand_passed += passed_mols
            grand_rejected += rejected_mols
            grand_errors += error_mols

            f_master_succ.write(f"[{b_name}] Total: {total_mols} | Passed: {passed_mols} | PDBs: {split_dir_path}\n")
            f_master_err.write(f"[{b_name}] Rejected: {rejected_mols} | Errors: {error_mols} | Log: {err_log_path}\n")
            f_master_succ.flush()
            f_master_err.flush()

            main_logger.info(
                f"<<< Finished GPU {b_name} | Evaluated: {total_mols:,} | "
                f"Passed: {passed_mols:,} ({(passed_mols/max(1, total_mols))*100:.1f}%) | "
                f"Individual PDBs in: {split_dir_path}"
            )

    except KeyboardInterrupt:
        main_logger.warning("GPU pipeline interrupted by user. Finalizing writes...")
    finally:
        pass

    f_master_succ.close()
    f_master_err.close()
    f_master_csv.close()

    total_wall_time = time.time() - start_time_all

    main_logger.info("=" * 80)
    main_logger.info("ALL GPU BATCHES PROCESSING COMPLETED")
    main_logger.info("=" * 80)
    main_logger.info(f"Total Batches Evaluated       : {len(batches)}")
    main_logger.info(f"Total Molecules Evaluated     : {grand_total:,}")
    main_logger.info(f"Total Passed Low-Strain Mols  : {grand_passed:,} ({(grand_passed/max(1, grand_total))*100:.2f}%)")
    main_logger.info(f"Total Rejected Strained Mols  : {grand_rejected:,} ({(grand_rejected/max(1, grand_total))*100:.2f}%)")
    main_logger.info(f"Total Setup / Parse Errors    : {grand_errors:,}")
    main_logger.info(f"Total Execution Time          : {total_wall_time:.1f} s ({total_wall_time/60.0:.2f} min)")
    main_logger.info(f"Consolidated Summary CSV      : {master_csv}")
    main_logger.info(f"Master Success Log            : {master_succ_log}")
    main_logger.info(f"Master Error Log              : {master_err_log}")
    main_logger.info("=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description="GPU-accelerated batch PDBQT strain filtering and PDB export on NVIDIA RTX 4050."
    )
    parser.add_argument("-i", "--input", required=True, help="Path to parent folder containing batch_* folders, or a single batch folder")
    parser.add_argument("-o", "--output_dir", required=True, help="Directory to save per-batch outputs and consolidated master PDB")
    parser.add_argument("--device", default="cuda:0", help="Compute device (default: cuda:0)")
    parser.add_argument("--max_strain", type=float, default=10.0, help="Maximum strain threshold in kcal/mol (default: 10.0)")
    parser.add_argument("--restraint_k", type=float, default=25.0, help="Positional restraint constant k (default: 25.0)")
    parser.add_argument("--max_relax_steps", type=int, default=5, help="ANI autograd relaxation steps (default: 5)")
    parser.add_argument("--relax_lr", type=float, default=0.005, help="GPU autograd learning rate (default: 0.005)")
    parser.add_argument("--num_confs_free", type=int, default=5, help="Requested free conformers (default: 5)")
    parser.add_argument("--split_pdb", action="store_true", help="Save individual .pdb files for each passed molecule inside its batch folder")
    parser.add_argument("--sort_by", choices=["affinity", "strain"], default="affinity", help="Sort order for passed molecules (default: affinity)")
    parser.add_argument("--no_resume", action="store_true", help="Do not skip already completed batches")
    parser.add_argument("--max_mols", type=int, default=None, help="Maximum number of molecules to evaluate per batch for test run")

    args = parser.parse_args()

    run_gpu_batch_pipeline(
        input_path=args.input,
        output_dir=args.output_dir,
        device=args.device,
        max_strain=args.max_strain,
        restraint_k=args.restraint_k,
        max_relax_steps=args.max_relax_steps,
        relax_lr=args.relax_lr,
        num_confs_free=args.num_confs_free,
        split_pdb=args.split_pdb,
        sort_by=args.sort_by,
        resume=not args.no_resume,
        max_mols=args.max_mols
    )


if __name__ == "__main__":
    main()
