# -*- coding: utf-8 -*-
"""Speed-oriented LGBM variants: cap the boosting rounds (tree budget).

The run_lgbm_v2 replica has best_iteration=1500 -> 1500*386 = 579k trees;
single-row predict traverses every tree, so predict cost is linear in the
round cap.  This script trains round-capped variants (identical params,
no early stop -> exact tree budget), writes the strict probs caches that
run_combiner consumes via --lgbm-suffix _fast{R}, and prints the member's
own open-set numbers + single-row predict latency.

Usage:
  python scripts/run_lgbm_fast.py --dataset 10s_nogro --rounds 50,100,200,400
  python scripts/run_combiner.py --dataset 10s_nogro --protocol strict \
      --df-tags ngL_dfs_branch_all_e100,ngL_dfs_branch_all_dt,ngL_dfs_branch_aux \
      --lovo --lgbm-suffix _fast200 --out combo_10s_nogro_L8k_fast200
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import lightgbm as lgb

from src.config import PROJECT_ROOT, RunConfig
from src.data import SessionIndex, load_splits, session_pool


def norm(P):
    return P / (P.sum(1, keepdims=True) + 1e-12)


def t_ms(f, n=50, warm=3):
    for _ in range(warm):
        f()
    t0 = time.perf_counter()
    for _ in range(n):
        f()
    return (time.perf_counter() - t0) / n * 1e3


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="10s_nogro")
    ap.add_argument("--rounds", default="50,100,200,400")
    ap.add_argument("--leaves", type=int, default=63)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rcfg = RunConfig(dataset=args.dataset, protocol="strict")
    rcfg.split.seed = args.seed
    splits = load_splits(rcfg.splits_path)
    index = SessionIndex(rcfg.dataset_dir, rcfg.bin_size_ms)
    known = sorted(splits["known_videos"])
    vidx = {v: i for i, v in enumerate(known)}

    pools = dict(
        train=session_pool(splits, "known", "train"),
        val_probe=session_pool(splits, "known", "val", gallery=False),
        val_imp=session_pool(splits, "unknown", "val"),
        test_probe=session_pool(splits, "known", "test", gallery=False),
        test_imp=session_pool(splits, "unknown", "test"),
    )
    all_sids = sorted({s for v in pools.values() for s in v})
    X = np.load(PROJECT_ROOT / "runs" / "cache" / f"segv2_{args.dataset}.npy")
    assert X.shape[0] == len(all_sids)
    pos = {s: i for i, s in enumerate(all_sids)}
    A = lambda pool: X[[pos[s] for s in pool]]

    def y_of(pool):
        return np.array([vidx[splits["sessions"][s]["video"]] for s in pool],
                        dtype=np.int64)

    summary = []
    for R in [int(x) for x in args.rounds.split(",")]:
        params = dict(objective="multiclass", num_class=len(known),
                      metric="multi_logloss", learning_rate=args.lr,
                      num_leaves=args.leaves, min_data_in_leaf=20,
                      feature_fraction=0.7, bagging_fraction=0.8, bagging_freq=1,
                      lambda_l2=1.0, verbosity=-1, num_threads=8, seed=args.seed)
        t0 = time.time()
        model = lgb.train(params, lgb.Dataset(A(pools["train"]),
                                              y_of(pools["train"])),
                          num_boost_round=R)
        tr_s = time.time() - t0

        probs_cache = {}
        for split in ("val", "test"):
            probs_cache[f"{split}_probe"] = model.predict(A(pools[f"{split}_probe"]))
            probs_cache[f"{split}_imp"] = model.predict(A(pools[f"{split}_imp"]))
        row = {"rounds": R, "trees": int(model.num_trees()),
               "train_s": tr_s}
        row["predict_ms"] = float(t_ms(
            lambda: model.predict(A(pools["val_probe"])[0:1])))
        for split in ("val", "test"):
            P = norm(probs_cache[f"{split}_probe"])
            I = norm(probs_cache[f"{split}_imp"])
            y = y_of(pools[f"{split}_probe"])
            row[f"top1_{split}"] = float((P.argmax(1) == y).mean())
            sp, co = P.max(1), P.argmax(1) == y
            si = I.max(1)
            from src.evaluate import _threshold_for_fpr
            row[f"tpr05_{split}"] = float(
                np.mean(co & (sp >= _threshold_for_fpr(si, 0.005))))
        summary.append(row)
        print(f"[fast{R:4d}] trees={row['trees']:6d} train={tr_s:5.0f}s "
              f"predict={row['predict_ms']:7.2f}ms  "
              f"val top1={row['top1_val']:.3f} @.5%={row['tpr05_val']:.3f}  "
              f"test top1={row['top1_test']:.3f} @.5%={row['tpr05_test']:.3f}")

        np.savez(PROJECT_ROOT / "runs" / "cache"
                 / f"lgbm_probs_{args.dataset}_strict_fast{R}.npz",
                 best_iter=R, **probs_cache)

    out = PROJECT_ROOT / "runs" / f"{args.dataset}_lgbm_fast"
    out.mkdir(parents=True, exist_ok=True)
    (out / "report.json").write_text(json.dumps(
        {"config": vars(args), "variants": summary}, indent=1, default=str),
        encoding="utf-8")
    print(f"[wrote] {out / 'report.json'}")


if __name__ == "__main__":
    main()
