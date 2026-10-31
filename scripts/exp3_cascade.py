# -*- coding: utf-8 -*-
"""Exp3: cascade open-set evaluation — LGBM top-K shortlist masks stage-2.

Stage-1: LGBM segv2 top-K candidate set (cheap). Stage-2: DF ensemble /
mean-prob ensemble evaluated only on the shortlist. Impostor rows are masked
by their own LGBM top-K (deployment-realistic). Thresholds re-derived from
the masked impostor score distribution at the same FPR grid.

Variants:
  df_maskrenorm  DF-ens probs renormalized over the mask, score = max
  ens_maskrenorm LGBM+DF mean-prob renormalized over the mask, score = max
  ens_maskabs    identification restricted to mask, score = RAW ens prob
                 of the restricted argmax (keeps absolute calibration)

Usage: python scripts/exp3_cascade.py [10s|5s]
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import PROJECT_ROOT
from src.data import load_splits, session_pool
from src.evaluate import open_set_metrics, format_open_set_report

DF_TAGS = {"10s": ["df_branch_all_e100", "df_branch_all_dt"],
           "5s": ["df5_branch_all_e100"]}
KS = [16, 32, 64, 128, 256]
FPRS = (0.005, 0.01, 0.05)


def renorm_masked(M: np.ndarray, mask: np.ndarray) -> np.ndarray:
    out = np.where(mask, M, 0.0)
    s = out.sum(1, keepdims=True)
    s[s == 0] = 1.0
    return out / s


def abs_masked(M: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Identification restricted to mask, score = raw prob of restricted argmax."""
    out = np.where(mask, M, 0.0)
    return out


def summarize(m: dict, r1: float | None = None) -> dict:
    d = {"top1": m["closed"]["top1"], "auroc": m["auroc_detection"],
         **{f"tpr@{f}": m["fpr"][f]["tpr"] for f in m["fpr"]},
         **{f"ci@{f}": m["fpr"][f].get("tpr_ci95") for f in m["fpr"]}}
    if r1 is not None:
        d["stage1_recall"] = r1
    return d


def main(dataset: str = "10s") -> None:
    splits = load_splits(PROJECT_ROOT / "splits" / f"splits_{dataset}_s42.json")
    known = sorted(splits["known_videos"])

    lgbm_z = np.load(PROJECT_ROOT / "runs" / "cache" / f"lgbm_probs_{dataset}.npz")
    out = {"dataset": dataset, "n_known": len(known), "splits": {}}

    for split in ("test", "val"):
        d0 = np.load(PROJECT_ROOT / "runs" / DF_TAGS[dataset][0] / f"probs_{split}.npz")
        y = d0["probe_y"]
        pl = lgbm_z[f"{split}_probe"] / lgbm_z[f"{split}_probe"].sum(1, keepdims=True)
        pil = lgbm_z[f"{split}_imp"] / lgbm_z[f"{split}_imp"].sum(1, keepdims=True)
        dfs_p, dfs_i = [], []
        for t in DF_TAGS[dataset]:
            z = np.load(PROJECT_ROOT / "runs" / t / f"probs_{split}.npz")
            dfs_p.append(z["probe"] / z["probe"].sum(1, keepdims=True))
            dfs_i.append(z["imp"] / z["imp"].sum(1, keepdims=True))
        pdf = np.mean(dfs_p, 0)
        pid = np.mean(dfs_i, 0)
        # same fusion as run_ensemble.py: (LGBM + mean(DFs)) / 2
        ens_p = (pl + pdf) / 2
        ens_i = (pil + pid) / 2

        res = {}

        # ---- baselines (sanity vs findings.md) ----
        res["baseline_df"] = summarize(open_set_metrics(pdf, y, pid.max(1), n_boot=1000))
        res["baseline_ens"] = summarize(open_set_metrics(ens_p, y, ens_i.max(1), n_boot=1000))

        order_p = np.argsort(-pl, axis=1)
        order_i = np.argsort(-pil, axis=1)

        for K in KS:
            mp = np.zeros_like(pl, dtype=bool)
            mi = np.zeros_like(pil, dtype=bool)
            np.put_along_axis(mp, order_p[:, :K], True, axis=1)
            np.put_along_axis(mi, order_i[:, :K], True, axis=1)
            r1 = float(np.mean(np.any(order_p[:, :K] == y[:, None], axis=1)))

            v = {}
            v["df_maskrenorm"] = open_set_metrics(renorm_masked(pdf, mp), y,
                                                  renorm_masked(pid, mi).max(1), n_boot=1000)
            v["ens_maskrenorm"] = open_set_metrics(renorm_masked(ens_p, mp), y,
                                                   renorm_masked(ens_i, mi).max(1), n_boot=1000)
            # abs variant: restricted argmax, absolute score
            rp, ri = abs_masked(ens_p, mp), abs_masked(ens_i, mi)
            v["ens_maskabs"] = open_set_metrics(rp, y, ri.max(1), n_boot=1000)
            res[f"K{K}"] = {name: summarize(m, r1) for name, m in v.items()}

        out["splits"][split] = res

        print(f"\n================ {dataset} {split.upper()} ================")
        for name in ("baseline_df", "baseline_ens"):
            r = res[name]
            print(f"{name:>16}: top1={r['top1']:.4f} AUROC={r['auroc']:.4f} "
                  f"@0.5%={r['tpr@0.005']:.4f} @1%={r['tpr@0.01']:.4f} @5%={r['tpr@0.05']:.4f}")
        for K in KS:
            r1 = res[f"K{K}"]["ens_maskrenorm"]["stage1_recall"]
            print(f"-- K={K} (stage-1 recall R@K={r1:.4f}) --")
            for name in ("df_maskrenorm", "ens_maskrenorm", "ens_maskabs"):
                r = res[f"K{K}"][name]
                print(f"{name:>16}: top1={r['top1']:.4f} AUROC={r['auroc']:.4f} "
                      f"@0.5%={r['tpr@0.005']:.4f} @1%={r['tpr@0.01']:.4f} @5%={r['tpr@0.05']:.4f}")

    dest = PROJECT_ROOT / "runs" / "cascade" / f"exp3_cascade_{dataset}.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(f"\n[wrote] {dest}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "10s")
