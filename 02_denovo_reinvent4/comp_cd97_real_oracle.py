# Native PEP 420 CD97 Real-Oracle Scoring Plugin for REINVENT 4
import os
import sys
import gc
import logging
from pathlib import Path
from typing import List, Optional
from dataclasses import dataclass
import numpy as np

try:
    import torch
except ImportError:
    torch = None

from reinvent_plugins.normalize import normalize_smiles
from reinvent_plugins.components.add_tag import add_tag

from cd97_reinvent4_dap_REAL_ORACLE import RealOracleScoringEngine

try:
    from reinvent_plugins.components.component_results import ComponentResults
except ImportError:
    class ComponentResults:
        def __init__(self, scores: List[np.ndarray], metadata: Optional[dict] = None):
            self.scores = scores
            self.metadata = metadata or {}


@add_tag("__parameters")
@dataclass
class Parameters:
    output_dir: Optional[List[str]] = None
    sigma: Optional[List[float]] = None
    max_strain: Optional[List[float]] = None
    min_ra_score: Optional[List[float]] = None


@add_tag("__component")
class CD97RealOracleScorer:
    def __init__(self, params: Optional[Parameters] = None):
        out_dir = None
        sigma = 40.0
        max_strain = 10.0
        min_ra = 0.60

        if params is not None:
            raw_dir = getattr(params, "output_dir", None)
            if raw_dir:
                while isinstance(raw_dir, (list, tuple)) and len(raw_dir) > 0:
                    raw_dir = raw_dir[0]
                if isinstance(raw_dir, (str, Path)):
                    out_dir = Path(raw_dir)

            if getattr(params, "sigma", None) is not None:
                s = params.sigma
                while isinstance(s, (list, tuple)) and len(s) > 0:
                    s = s[0]
                sigma = float(s)

            if getattr(params, "max_strain", None) is not None:
                s = params.max_strain
                while isinstance(s, (list, tuple)) and len(s) > 0:
                    s = s[0]
                max_strain = float(s)

            if getattr(params, "min_ra_score", None) is not None:
                s = params.min_ra_score
                while isinstance(s, (list, tuple)) and len(s) > 0:
                    s = s[0]
                min_ra = float(s)

        self.output_dir = out_dir or (Path(__file__).resolve().parent / "analysis" / "production_1000steps" / "docking_runs")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.engine = RealOracleScoringEngine(sigma=sigma, max_strain=max_strain, min_ra_score=min_ra)
        self.smiles_type = "rdkit_smiles"

        # Persistent step counter across restarts/shutdowns
        existing_steps = []
        for p in self.output_dir.glob("step_*"):
            if p.is_dir():
                s_name = p.name.replace("step_", "")
                if s_name.isdigit():
                    existing_steps.append(int(s_name))
        self._step_counter = max(existing_steps) if existing_steps else 0

    @normalize_smiles
    def __call__(self, smilies: List[str]) -> ComponentResults:
        self._step_counter += 1
        step_dir = self.output_dir / f"step_{self._step_counter:04d}"
        step_dir.mkdir(parents=True, exist_ok=True)

        from concurrent.futures import ThreadPoolExecutor

        def _eval_worker(item):
            idx, smi = item
            try:
                res = self.engine.evaluate_molecule(smi, idx + 1, step_dir, step_num=self._step_counter)
                return idx, (res['Score'] if res['valid'] else 0.0)
            except Exception:
                logging.getLogger("CD97_RealOracle").exception("Oracle failed for molecule index %s", idx)
                return idx, 0.0

        # Highly-parallel batch evaluation across AMD Ryzen 9 7950X threads (16 parallel workers)
        num_workers = min(16, max(1, len(smilies)))
        scores_map = {}
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            for idx, sc in executor.map(_eval_worker, enumerate(smilies)):
                scores_map[idx] = sc

        scores = [scores_map.get(i, 0.0) for i in range(len(smilies))]

        # 1. Cleanup all temporary step files to prevent disk exhaustion
        for ext in ["*.dlg", "*.xml", "*.pdbqt", "*.pdb", "*.sdf"]:
            for f in step_dir.glob(ext):
                try:
                    f.unlink()
                except Exception:
                    pass

        # 2. Remove empty step directory
        try:
            if not any(step_dir.iterdir()):
                step_dir.rmdir()
        except Exception:
            pass

        # 3. Purge GPU memory & Python garbage collection
        gc.collect()
        if torch is not None and torch.cuda.is_available():
            torch.cuda.empty_cache()

        return ComponentResults([np.array(scores, dtype=np.float32)])
