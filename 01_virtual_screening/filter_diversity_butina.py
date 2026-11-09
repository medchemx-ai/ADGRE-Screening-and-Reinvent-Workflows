#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
BEMIS-MURCKO & TAYLOR-BUTINA CHEMICAL DIVERSITY FILTERING PIPELINE
================================================================================
Scientific Purpose:
    Eliminates analogue redundancy from docked/strain-filtered PDB libraries.
    Groups structurally similar molecules and bioisosteres into clusters using:
      1. Bemis-Murcko Scaffold Extraction (Ring framework topology)
      2. Morgan Circular Fingerprints (ECFP4, radius=2, 2048 bits)
      3. Taylor-Butina Sphere Exclusion Clustering (Tanimoto distance <= 0.60)
    Selects the single highest-affinity champion from each cluster and discards
    redundant analogues.
    Performs rigorous post-selection Quality Control to guarantee zero chemical
    redundancy (Tc < 0.40 across all output leads).

Outputs:
    - diverse_leads_pdbs/     : Individual 3D PDBs for each cluster champion
    - diversity_clusters.csv  : Cluster breakdown, Murcko scaffolds, sizes & leads
    - diversity_champions.csv : Selected lead ranking table
    - diversity_summary.log   : Full audit log and QC metrics
================================================================================
"""

import os
import sys
import re
import time
import argparse
import logging
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass

# Force UTF-8 encoding across Windows and Linux terminals
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit import DataStructs
from rdkit.ML.Cluster import Butina
from rdkit import RDLogger

# Suppress minor RDKit sanitization warnings during high-throughput ingest
RDLogger.DisableLog('rdApp.warning')
RDLogger.DisableLog('rdApp.error')

# Regex patterns for extracting affinity, strain, and PLIP from PDB remarks
RE_AFFINITY = re.compile(r"REMARK\s+.*(?:AFFINITY|ENERGY|SCORE).*?([-\d.]+)", re.IGNORECASE)
RE_STRAIN = re.compile(r"REMARK\s+.*STRAIN.*?([-\d.]+)", re.IGNORECASE)
RE_PLIP = re.compile(r"REMARK\s+.*(?:PLIP|INTERACTION).*?SCORE.*?([-\d.]+)", re.IGNORECASE)
RE_NAME = re.compile(r"REMARK\s+.*(?:NAME|ID).*?\s+(\S+)", re.IGNORECASE)
RE_FN_ENERGY = re.compile(r"_E([-\d.]+)")
RE_FN_PLIP = re.compile(r"_PLIP([-\d.]+)", re.IGNORECASE)


@dataclass
class MoleculeRecord:
    idx: int
    name: str
    pdb_path: str
    pdb_block: str
    mol: Chem.Mol
    smiles: str
    murcko_scaffold: str
    generic_scaffold: str
    affinity: float          # Lower is better (e.g. -8.5 kcal/mol)
    strain: float            # In kcal/mol
    plip_score: float        # Higher is better (e.g. 24.5)
    num_heavy_atoms: int
    cluster_id: int = -1
    is_champion: bool = False


def setup_logger(output_dir: str) -> logging.Logger:
    """Configures multi-channel logging to both stdout and a log file."""
    os.makedirs(output_dir, exist_ok=True)
    logger = logging.getLogger("DiversityFilter")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")

    fh = logging.FileHandler(os.path.join(output_dir, "diversity_summary.log"), encoding="utf-8")
    fh.setLevel(logging.INFO)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    return logger


def load_plip_csv_scores(csv_path: Optional[str]) -> Dict[str, float]:
    """Loads molecule PLIP interaction scores from a CSV table (e.g. all_100_plip_ranked.csv)."""
    scores: Dict[str, float] = {}
    if not csv_path or not os.path.isfile(csv_path):
        return scores

    import csv
    try:
        with open(csv_path, 'r', encoding='utf-8', errors='replace') as f:
            reader = csv.DictReader(f)
            for row in reader:
                mol_id = (
                    row.get("molecule_id")
                    or row.get("mol_name")
                    or row.get("molecule")
                    or row.get("ligand_filename")
                    or ""
                ).strip()
                score_str = (
                    row.get("total_plip_score")
                    or row.get("target_interaction_score")
                    or row.get("plip_score")
                    or row.get("score")
                    or "0"
                ).strip()
                try:
                    sc = float(score_str)
                    if mol_id:
                        scores[mol_id] = sc
                        scores[mol_id.lower()] = sc
                        base_clean = os.path.splitext(os.path.basename(mol_id))[0]
                        scores[base_clean] = sc
                        scores[base_clean.lower()] = sc
                except ValueError:
                    pass
    except Exception:
        pass

    return scores


def parse_pdb_file(pdb_path: str, idx: int, plip_scores_map: Optional[Dict[str, float]] = None) -> Optional[MoleculeRecord]:
    """Parses an individual PDB file, extracting 3D coordinates, affinity, strain, and scaffolds."""
    try:
        with open(pdb_path, 'r', encoding='utf-8', errors='replace') as f:
            pdb_content = f.read()
    except Exception:
        return None

    if not pdb_content.strip():
        return None

    # Parse RDKit molecule from PDB text
    mol = Chem.MolFromPDBBlock(pdb_content, removeHs=False, sanitize=False)
    if mol is None or mol.GetNumAtoms() == 0:
        # Fallback sanitize
        mol = Chem.MolFromPDBBlock(pdb_content, removeHs=False, sanitize=True)
    if mol is None or mol.GetNumAtoms() == 0:
        return None

    # Sanitize if possible
    try:
        Chem.SanitizeMol(mol)
    except Exception:
        pass

    # Extract Molecule Name
    fn = os.path.splitext(os.path.basename(pdb_path))[0]
    name = fn
    m_name = RE_NAME.search(pdb_content)
    if m_name:
        name = m_name.group(1).strip()

    # Extract Docking Affinity
    affinity = 0.0
    m_aff = RE_AFFINITY.search(pdb_content)
    if m_aff:
        try:
            affinity = float(m_aff.group(1))
        except ValueError:
            pass
    else:
        m_fn = RE_FN_ENERGY.search(fn)
        if m_fn:
            try:
                affinity = float(m_fn.group(1))
            except ValueError:
                pass

    # Extract Strain Energy
    strain = 0.0
    m_str = RE_STRAIN.search(pdb_content)
    if m_str:
        try:
            strain = float(m_str.group(1))
        except ValueError:
            pass

    # Extract PLIP Interaction Score
    plip_score = 0.0
    if plip_scores_map:
        if name in plip_scores_map:
            plip_score = plip_scores_map[name]
        elif name.lower() in plip_scores_map:
            plip_score = plip_scores_map[name.lower()]
        elif fn in plip_scores_map:
            plip_score = plip_scores_map[fn]
        elif fn.lower() in plip_scores_map:
            plip_score = plip_scores_map[fn.lower()]

    if plip_score == 0.0:
        m_plip = RE_PLIP.search(pdb_content)
        if m_plip:
            try:
                plip_score = float(m_plip.group(1))
            except ValueError:
                pass
        else:
            m_fn_plip = RE_FN_PLIP.search(fn)
            if m_fn_plip:
                try:
                    plip_score = float(m_fn_plip.group(1))
                except ValueError:
                    pass

    # Generate 2D SMILES (no hydrogens)
    try:
        no_h = Chem.RemoveHs(mol)
        smiles = Chem.MolToSmiles(no_h)
    except Exception:
        smiles = "N/A"

    # Layer 1: Bemis-Murcko Framework Extraction
    try:
        scaffold_mol = MurckoScaffold.GetScaffoldForMol(no_h)
        murcko_scaffold = Chem.MolToSmiles(scaffold_mol) if scaffold_mol else ""
        if not murcko_scaffold:
            murcko_scaffold = "[ACYCLIC]"
    except Exception:
        murcko_scaffold = "[UNKNOWN]"

    try:
        generic_mol = MurckoScaffold.MakeScaffoldGeneric(scaffold_mol) if scaffold_mol else None
        generic_scaffold = Chem.MolToSmiles(generic_mol) if generic_mol else murcko_scaffold
    except Exception:
        generic_scaffold = murcko_scaffold

    return MoleculeRecord(
        idx=idx,
        name=name,
        pdb_path=pdb_path,
        pdb_block=pdb_content,
        mol=mol,
        smiles=smiles,
        murcko_scaffold=murcko_scaffold,
        generic_scaffold=generic_scaffold,
        affinity=affinity,
        strain=strain,
        plip_score=plip_score,
        num_heavy_atoms=mol.GetNumHeavyAtoms()
    )


def discover_input_pdbs(input_path: str) -> List[str]:
    """Finds all .pdb files in a folder or recursively across batch passed_pdbs subfolders."""
    pdb_files = []
    if os.path.isfile(input_path) and input_path.lower().endswith('.pdb'):
        return [input_path]

    if not os.path.isdir(input_path):
        return []

    # Priority 1: Check for passed_pdbs folders (from strain filter)
    for root, dirs, files in os.walk(input_path):
        for f in files:
            if f.lower().endswith('.pdb'):
                # Avoid multi-model or raw receptor PDBs if obvious
                if "receptor" not in f.lower() and "protein" not in f.lower():
                    pdb_files.append(os.path.join(root, f))

    pdb_files.sort()
    return pdb_files


def run_diversity_clustering(
    input_path: str,
    output_dir: str,
    tanimoto_cutoff: float = 0.40,
    sort_by: str = "plip",
    max_leads: Optional[int] = None,
    plip_csv: Optional[str] = None
) -> None:
    """Executes the complete Bemis-Murcko and Taylor-Butina diversity workflow."""
    logger = setup_logger(output_dir)

    dist_cutoff = 1.0 - tanimoto_cutoff

    logger.info("=" * 80)
    logger.info("BEMIS-MURCKO & TAYLOR-BUTINA CHEMICAL DIVERSITY FILTER")
    logger.info("=" * 80)
    logger.info(f"Input Path              : {input_path}")
    logger.info(f"Output Directory        : {output_dir}")
    if plip_csv:
        logger.info(f"PLIP Score CSV Source   : {plip_csv}")
    logger.info(f"Tanimoto Similarity (Tc): >= {tanimoto_cutoff:.2f} (Distance D <= {dist_cutoff:.2f})")
    logger.info(f"Fingerprint Type        : Morgan ECFP4 (radius=2, 2048 bits)")
    metric_desc = "Highest PLIP Interaction Score" if sort_by == "plip" else f"Lowest {sort_by}"
    logger.info(f"Lead Champion Metric    : {metric_desc}")
    if max_leads:
        logger.info(f"Max Leads Retained      : Top {max_leads} champions")
    logger.info("=" * 80)

    # 1. Discover PDB files
    pdb_paths = discover_input_pdbs(input_path)
    logger.info(f"Discovered {len(pdb_paths):,} input PDB files across directories.")
    if not pdb_paths:
        logger.error("No valid .pdb files found. Please check your input directory path.")
        return

    # Load PLIP scores if CSV provided
    plip_scores_map = load_plip_csv_scores(plip_csv)
    if plip_scores_map:
        logger.info(f"Loaded PLIP interaction scores for {len(plip_scores_map):,} molecules from CSV.")

    # 2. Ingest Molecules & Extract Scaffolds
    logger.info("Ingesting 3D PDBs and extracting Bemis-Murcko frameworks...")
    records: List[MoleculeRecord] = []
    fps = []
    t_start = time.time()

    for idx, p in enumerate(pdb_paths, 1):
        rec = parse_pdb_file(p, idx, plip_scores_map=plip_scores_map)
        if rec and rec.mol:
            # Generate ECFP4 (Morgan radius=2, 2048 bits)
            try:
                fp = AllChem.GetMorganFingerprintAsBitVect(rec.mol, radius=2, nBits=2048)
                records.append(rec)
                fps.append(fp)
            except Exception:
                continue

        if idx % 500 == 0 or idx == len(pdb_paths):
            logger.info(f"  Ingest progress: {idx:,} / {len(pdb_paths):,} PDBs processed ({len(records):,} valid)")

    n_mols = len(records)
    logger.info(f"Successfully loaded {n_mols:,} valid molecules for diversity clustering.")
    if n_mols == 0:
        logger.error("No valid molecules could be parsed from PDBs. Exiting.")
        return

    # 3. Compute 1D Compressed Pairwise Tanimoto Distance Matrix
    logger.info("Computing 1D Pairwise Tanimoto Distance Matrix (Taylor-Butina format)...")
    t_dist_start = time.time()
    dist_matrix = []
    for i in range(1, n_mols):
        sims = DataStructs.BulkTanimotoSimilarity(fps[i], fps[:i])
        dist_matrix.extend([1.0 - s for s in sims])

    t_dist_elapsed = time.time() - t_dist_start
    logger.info(f"Pairwise distance matrix generated ({len(dist_matrix):,} comparisons) in {t_dist_elapsed:.2f} s.")

    # 4. Taylor-Butina Sphere Exclusion Clustering
    logger.info(f"Running Taylor-Butina sphere exclusion clustering (cutoff D <= {dist_cutoff:.2f})...")
    t_clust_start = time.time()
    clusters = Butina.ClusterData(dist_matrix, n_mols, dist_cutoff, isDistData=True)
    t_clust_elapsed = time.time() - t_clust_start

    num_clusters = len(clusters)
    logger.info(f"Taylor-Butina completed in {t_clust_elapsed:.2f} s -> Partitioned into {num_clusters:,} distinct chemical families.")

    # 5. Cluster Champion Selection & Analogue Pruning
    logger.info(f"Selecting cluster champions based on: {metric_desc}...")
    champions: List[MoleculeRecord] = []
    cluster_records_map: Dict[int, List[MoleculeRecord]] = {}

    for c_id, cluster_indices in enumerate(clusters, 1):
        mols_in_cluster = [records[i] for i in cluster_indices]
        for m in mols_in_cluster:
            m.cluster_id = c_id

        # Sort within cluster: highest PLIP score wins (tie-breaker: lowest affinity, then lowest strain)
        if sort_by == "plip":
            mols_in_cluster.sort(key=lambda x: (-x.plip_score, x.affinity, x.strain))
        elif sort_by == "affinity":
            mols_in_cluster.sort(key=lambda x: (x.affinity, x.strain))
        else:
            mols_in_cluster.sort(key=lambda x: (x.strain, x.affinity))

        # Champion is the #1 ranked molecule in this cluster
        champion = mols_in_cluster[0]
        champion.is_champion = True
        champions.append(champion)
        cluster_records_map[c_id] = mols_in_cluster

    # Global sort across all champions
    if sort_by == "plip":
        champions.sort(key=lambda x: (-x.plip_score, x.affinity, x.strain))
    elif sort_by == "affinity":
        champions.sort(key=lambda x: (x.affinity, x.strain))
    else:
        champions.sort(key=lambda x: (x.strain, x.affinity))

    if max_leads and max_leads < len(champions):
        logger.info(f"Capping top champions to --max_leads={max_leads}")
        champions = champions[:max_leads]

    # 6. Post-Selection Quality Control (QC Verification)
    logger.info("=" * 80)
    logger.info("QUALITY CONTROL VERIFICATION: ZERO DIVERSITY REDUNDANCY CHECK")
    logger.info("=" * 80)
    champ_fps = [AllChem.GetMorganFingerprintAsBitVect(c.mol, radius=2, nBits=2048) for c in champions]
    qc_passed = True
    max_pairwise_tc = 0.0
    violation_pairs = []

    for i in range(len(champions)):
        for j in range(i + 1, len(champions)):
            tc = DataStructs.TanimotoSimilarity(champ_fps[i], champ_fps[j])
            if tc > max_pairwise_tc:
                max_pairwise_tc = tc
            if tc >= tanimoto_cutoff:
                qc_passed = False
                violation_pairs.append((champions[i].name, champions[j].name, tc))

    if qc_passed:
        logger.info(f"[PASS] QC Verification Successful: 100% of lead pairs satisfy Tc < {tanimoto_cutoff:.2f}!")
        logger.info(f"       Maximum Pairwise Tanimoto across all leads: {max_pairwise_tc:.3f}")
    else:
        logger.warning(f"[NOTE] Found {len(violation_pairs)} border pairs with Tc >= {tanimoto_cutoff:.2f}. Merging cluster borders...")
        # Auto-prune duplicate champions if any slip through sphere boundaries
        filtered_champs = []
        filtered_fps = []
        for c, fp in zip(champions, champ_fps):
            conflict = False
            for f_fp in filtered_fps:
                if DataStructs.TanimotoSimilarity(fp, f_fp) >= tanimoto_cutoff:
                    conflict = True
                    break
            if not conflict:
                filtered_champs.append(c)
                filtered_fps.append(fp)
        champions = filtered_champs
        logger.info(f"       Pruned to {len(champions):,} strictly non-redundant champions (Max Tc guaranteed < {tanimoto_cutoff:.2f}).")

    # 7. Export Outputs
    diverse_pdbs_dir = os.path.join(output_dir, "diverse_leads_pdbs")
    os.makedirs(diverse_pdbs_dir, exist_ok=True)

    logger.info(f"Exporting {len(champions):,} diverse lead PDBs into: {diverse_pdbs_dir}")
    for rank, champ in enumerate(champions, 1):
        aff_str = f"E{champ.affinity:.2f}" if champ.affinity != 0.0 else "E_NA"
        plip_str = f"_PLIP{champ.plip_score:.1f}" if champ.plip_score > 0 else ""
        out_pdb_name = f"lead_{rank:03d}_{champ.name}{plip_str}_{aff_str}_strain_{champ.strain:.1f}.pdb"
        out_pdb_path = os.path.join(diverse_pdbs_dir, out_pdb_name)

        # Write clean 3D PDB with diversity headers
        remarks = [
            f"REMARK   1  DIVERSE_LEAD_RANK:    {rank}",
            f"REMARK   1  MOLECULE_NAME:        {champ.name}",
            f"REMARK   1  ORIGINAL_CLUSTER_ID:  {champ.cluster_id}",
            f"REMARK   1  CLUSTER_ANALOGUE_SIZE:{len(cluster_records_map.get(champ.cluster_id, [champ]))}",
            f"REMARK   1  PLIP_SCORE:           {champ.plip_score:.2f}",
            f"REMARK   1  DOCKING_AFFINITY:     {champ.affinity:.2f} KCAL/MOL",
            f"REMARK   1  CONFORMATIONAL_STRAIN:{champ.strain:.2f} KCAL/MOL",
            f"REMARK   1  MURCKO_SCAFFOLD:      {champ.murcko_scaffold}",
            f"REMARK   1  GENERIC_SCAFFOLD:     {champ.generic_scaffold}",
            f"REMARK   1  DIVERSITY_STATUS:     PURE_NON_REDUNDANT_LEAD"
        ]
        with open(out_pdb_path, 'w', encoding='utf-8') as f_out:
            f_out.write("\n".join(remarks) + "\n")
            f_out.write(champ.pdb_block)

    # Export Champions Table CSV
    champ_csv = os.path.join(output_dir, "diversity_champions.csv")
    with open(champ_csv, 'w', encoding='utf-8') as f_csv:
        f_csv.write("rank,molecule_name,cluster_id,cluster_family_size,plip_score,docking_affinity,strain_kcal_mol,heavy_atoms,murcko_scaffold,generic_scaffold,smiles,pdb_file\n")
        for rank, c in enumerate(champions, 1):
            c_size = len(cluster_records_map.get(c.cluster_id, [c]))
            f_csv.write(
                f"{rank},\"{c.name}\",{c.cluster_id},{c_size},{c.plip_score:.2f},{c.affinity:.2f},{c.strain:.2f},"
                f"{c.num_heavy_atoms},\"{c.murcko_scaffold}\",\"{c.generic_scaffold}\",\"{c.smiles}\","
                f"\"{os.path.basename(c.pdb_path)}\"\n"
            )

    # Export Full Clusters Table CSV
    clusters_csv = os.path.join(output_dir, "diversity_all_clusters.csv")
    with open(clusters_csv, 'w', encoding='utf-8') as f_cls:
        f_cls.write("cluster_id,is_champion,molecule_name,plip_score,docking_affinity,strain_kcal_mol,murcko_scaffold,smiles\n")
        for c_id, mol_list in sorted(cluster_records_map.items()):
            for m in mol_list:
                f_cls.write(
                    f"{c_id},{1 if m.is_champion else 0},\"{m.name}\",{m.plip_score:.2f},{m.affinity:.2f},{m.strain:.2f},"
                    f"\"{m.murcko_scaffold}\",\"{m.smiles}\"\n"
                )

    total_wall_time = time.time() - t_start
    total_redundant_pruned = n_mols - len(champions)

    logger.info("=" * 80)
    logger.info("DIVERSITY FILTERING COMPLETED SUCCESSFULLY")
    logger.info("=" * 80)
    logger.info(f"Total Input Molecules       : {n_mols:,}")
    logger.info(f"Total Clusters Formed       : {num_clusters:,}")
    logger.info(f"Total Champions Selected    : {len(champions):,} (1 lead per chemical family)")
    logger.info(f"Redundant Analogues Pruned  : {total_redundant_pruned:,} ({(total_redundant_pruned/max(1, n_mols))*100:.1f}% redundancy removed)")
    logger.info(f"Total Wall Execution Time   : {total_wall_time:.2f} s")
    logger.info(f"Diverse Leads Output Dir    : {diverse_pdbs_dir}")
    logger.info(f"Champions Summary CSV       : {champ_csv}")
    logger.info(f"All Clusters CSV Breakdown  : {clusters_csv}")
    logger.info("=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description="Filter docked & strain-relaxed PDBs using Bemis-Murcko scaffolds and Taylor-Butina clustering."
    )
    parser.add_argument("-i", "--input", required=True, help="Path to input directory containing passed PDBs or batch output directories")
    parser.add_argument("-o", "--output_dir", required=True, help="Directory to save diverse leads and clustering reports")
    parser.add_argument("--plip_csv", default=None, help="Path to PLIP summary CSV (e.g. all_100_plip_ranked.csv) to extract exact PLIP scores")
    parser.add_argument("--tanimoto_cutoff", type=float, default=0.40, help="Tanimoto similarity threshold for clustering analogues (default: 0.40)")
    parser.add_argument("--sort_by", choices=["plip", "affinity", "strain"], default="plip", help="Criterion to select cluster champions (default: plip)")
    parser.add_argument("--max_leads", type=int, default=None, help="Maximum number of diverse leads to retain (default: all unique champions)")

    args = parser.parse_args()

    run_diversity_clustering(
        input_path=args.input,
        output_dir=args.output_dir,
        tanimoto_cutoff=args.tanimoto_cutoff,
        sort_by=args.sort_by,
        max_leads=args.max_leads,
        plip_csv=args.plip_csv
    )


if __name__ == "__main__":
    main()
