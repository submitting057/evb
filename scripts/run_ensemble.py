# -*- coding: utf-8 -*-
"""Ensemble evaluation: DF CNN probs + LGBM segv2 probs, same pools/order.

Usage: python scripts/run_ensemble.py --dataset 10s --df-tag df_branch_vtc
Retrains LGBM (same hyperparams as run_lgbm_v2), saves its probs, then
evaluates DF / LGBM / mean / rank-mean fusion under the open-set protocol.
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
from src.evaluate import format_open_set_report, open_set_metrics
from src.segment import segment_features_v2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="10s")
    ap.add_argument("--df-tag", required=True)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rcfg = RunConfig(dataset=args.dataset)
    rcfg.split.seed = args.seed
    index = SessionIndex(rcfg.dataset_dir, rcfg.bin_size_ms)
    splits = load_splits(rcfg.splits_path)
    known = sorted(splits["known_videos"])
    vidx = {v: i for i, v in enumerate(known)}

    pools = dict(
        train=session_pool(splits, "known", "train"),
        val_probe=session_pool(splits, "known", "val", gallery=False),
        val_imp=session_pool(splits, "unknown", "val"),
        test_probe=session_pool(splits, "known", "test", gallery=False),
        test_imp=session_pool(splits, "unknown", "test"),
    )

    seg_cache = PROJECT_ROOT / "runs" / "cache" / f"segv2_{args.dataset}.npy"
    X = np.load(seg_cache)
    all_sids = sorted(set().union(*pools.values()))
    if X.shape[0] != len(all_sids):
        X = np.stack([segment_features_v2(index, s) for s in all_sids])
        np.save(seg_cache, X)
    pos = {s: i for i, s in enumerate(all_sids)}
    A = lambda pool: X[[pos[s] for s in pool]]
    y_of = lambda pool: np.array([vidx[splits["sessions"][s]["video"]] for s in pool],
                                 dtype=np.int64)

    params = dict(objective="multiclass", num_class=len(known), metric="multi_logloss",
                  learning_rate=0.05, num_leaves=63, min_data_in_leaf=20,
                  feature_fraction=0.7, bagging_fraction=0.8, bagging_freq=1,
                  lambda_l2=1.0, verbosity=-1, num_threads=8, seed=args.seed)
    lgbm_cache = PROJECT_ROOT / "runs" / "cache" / f"lgbm_probs_{args.dataset}.npz"
    if lgbm_cache.exists():
        z = np.load(lgbm_cache)
        lgbm_probs = {s: (z[f"{s}_probe"], z[f"{s}_imp"]) for s in ("val", "test")}
        best_iter = int(z["best_iter"])
        print("[lgbm] probs loaded from cache")
    else:
        t0 = time.time()
        model = lgb.train(params, lgb.Dataset(A(pools["train"]), y_of(pools["train"])),
                          num_boost_round=1500,
                          valid_sets=[lgb.Dataset(A(pools["val_probe"]), y_of(pools["val_probe"]))],
                          callbacks=[lgb.early_stopping(100, verbose=False)])
        print(f"[lgbm] trained in {time.time() - t0:.0f}s best_iter={model.best_iteration}")
        lgbm_probs, best_iter = {}, model.best_iteration
        for split in ("val", "test"):
            lgbm_probs[split] = (
                model.predict(A(pools[f"{split}_probe"]), num_iteration=best_iter),
                model.predict(A(pools[f"{split}_imp"]), num_iteration=best_iter))
        np.savez(lgbm_cache, best_iter=best_iter,
                 **{f"{s}_{k}": v for s in lgbm_probs for k, v in
                    zip(("probe", "imp"), lgbm_probs[s])})

    tags = args.df_tag.split(",")
    report = {"df_tags": tags, "lgbm_best_iter": best_iter, "splits": {}}
    for split in ("val", "test"):
        first = np.load(PROJECT_ROOT / "runs" / tags[0] / f"probs_{split}.npz")
        y = first["probe_y"]
        assert y_of(pools[f"{split}_probe"]).tolist() == y.tolist(), "row order mismatch"
        pl, pi_l = lgbm_probs[split]
        pl = pl / pl.sum(1, keepdims=True)
        pi_l = pi_l / pi_l.sum(1, keepdims=True)
        dfs = []
        for tag in tags:
            z = np.load(PROJECT_ROOT / "runs" / tag / f"probs_{split}.npz")
            dfs.append((z["probe"] / z["probe"].sum(1, keepdims=True),
                        z["imp"] / z["imp"].sum(1, keepdims=True)))
        pd_ = np.mean([d[0] for d in dfs], 0)
        pi_d = np.mean([d[1] for d in dfs], 0)

        def rank01(M):
            R = np.argsort(np.argsort(M, 1), 1).astype(np.float64)
            return R / (M.shape[1] - 1)

        variants = {
            "lgbm": (pl, pi_l),
            "df": (pd_, pi_d),
            "mean": ((pl + pd_) / 2, (pi_l + pi_d) / 2),
            "mean_df_only": (pd_, pi_d),
        }
        del variants["mean_df_only"]
        results = {}
        for name, (pv, pi) in variants.items():
            m = open_set_metrics(pv, y, pi.max(1), n_boot=1000)
            print(format_open_set_report(f"{split.upper()} [{name}]", m))
            results[name] = m
        report["splits"][split] = results
    out = PROJECT_ROOT / "runs" / f"ens_{args.df_tag}"
    out.mkdir(parents=True, exist_ok=True)
    (out / "report.json").write_text(json.dumps(report, indent=1, default=str),
                                     encoding="utf-8")
    print(f"[wrote] {out / 'report.json'}")


if __name__ == "__main__":
    main()
