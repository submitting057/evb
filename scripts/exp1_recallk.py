# -*- coding: utf-8 -*-
"""Exp1: stage-1 shortlist recall@K from cached probability matrices.

Question: as a cheap stage-1 candidate generator, how large must the LGBM
segs-v2 top-K shortlist be so the true VID is inside with high recall?
DF / ensemble recall@K are computed as reference (expensive upper bound).

Usage: python scripts/exp1_recallk.py [10s|5s]
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import PROJECT_ROOT

DF_TAGS = {"10s": ["df_branch_all_e100", "df_branch_all_dt"],
           "5s": ["df5_branch_all_e100"]}
KS = [1, 2, 3, 4, 5, 8, 10, 16, 24, 32, 48, 64, 96, 128, 192, 256]
TARGETS = [0.95, 0.98, 0.99, 0.995]


def recall_at_k(P: np.ndarray, y: np.ndarray, k: int) -> float:
    order = np.argsort(-P, axis=1)[:, :k]
    return float(np.mean(np.any(order == y[:, None], axis=1)))


def main(dataset: str = "10s") -> None:
    lgbm_z = np.load(PROJECT_ROOT / "runs" / "cache" / f"lgbm_probs_{dataset}.npz")
    out = {"dataset": dataset, "ks": KS, "splits": {}}

    for split in ("val", "test"):
        d0 = np.load(PROJECT_ROOT / "runs" / DF_TAGS[dataset][0] / f"probs_{split}.npz")
        y = d0["probe_y"]
        pl = lgbm_z[f"{split}_probe"]
        pl = pl / pl.sum(1, keepdims=True)
        dfs = [np.load(PROJECT_ROOT / "runs" / t / f"probs_{split}.npz")["probe"]
               for t in DF_TAGS[dataset]]
        dfs = [p / p.sum(1, keepdims=True) for p in dfs]
        ens = (pl + np.mean(dfs, 0)) / 2   # same fusion as run_ensemble.py

        res = {}
        for name, P in (("lgbm", pl), ("ens", ens)):
            r = {k: recall_at_k(P, y, k) for k in KS}
            min_k = {f"R>={t}": next((k for k in KS if r[k] >= t), None) for t in TARGETS}
            res[name] = {"recall": r, "min_k": min_k}
        out["splits"][split] = res

        print(f"\n=== {dataset} {split}  n_probes={len(y)} ===")
        print(f"{'K':>5} | {'LGBM R@K':>9} | {'Ens R@K':>9}")
        for k in KS:
            print(f"{k:>5} | {res['lgbm']['recall'][k]:>9.4f} | {res['ens']['recall'][k]:>9.4f}")
        for t in TARGETS:
            print(f"min K with R>={t}: lgbm={res['lgbm']['min_k'][f'R>={t}']} "
                  f"ens={res['ens']['min_k'][f'R>={t}']}")

    dest = PROJECT_ROOT / "runs" / "cascade" / f"exp1_recallk_{dataset}.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(f"\n[wrote] {dest}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "10s")
