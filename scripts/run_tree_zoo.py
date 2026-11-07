# -*- coding: utf-8 -*-
"""Tree-ensemble zoo: replace the LGBM member with other tabular learners on
the SAME 120-d segment-v2 fingerprint (fair head-to-head, strict protocol).

Each algorithm writes runs/cache/lgbm_probs_{ds}_strict_{algo}.npz in the exact
format run_combiner.py expects (--lgbm-suffix _{algo}), plus a single-model
report under runs/tree_zoo_{algo}/.

Algorithms:
  xgb : XGBoost (same boosting family as LGBM -> homogeneity control)
  cat : CatBoost (ordered boosting, different regularisation path)
  et  : ExtraTrees (bagging axis: random splits, variance reduction)
  rf  : RandomForest (bagging axis, bootstrap)
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import PROJECT_ROOT, RunConfig
from src.data import SessionIndex, load_splits, session_pool, split_summary
from src.evaluate import format_open_set_report, open_set_metrics
from src.segment import segment_features_v2


def get_X(index, sids, cache_key):
    cache = PROJECT_ROOT / "runs" / "cache" / f"segv2_{cache_key}.npy"
    if cache.exists():
        X = np.load(cache)
        if X.shape[0] == len(sids):
            return X
    X = np.stack([segment_features_v2(index, s) for s in sids])
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache, X)
    return X


def build_model(algo, seed, n_class):
    if algo == "xgb":
        from xgboost import XGBClassifier
        return XGBClassifier(n_estimators=2000, learning_rate=0.05, max_depth=8,
                             subsample=0.8, colsample_bytree=0.7, reg_lambda=1.0,
                             tree_method="hist", early_stopping_rounds=100,
                             eval_metric="mlogloss", random_state=seed,
                             n_jobs=8, verbosity=0)
    if algo == "cat":
        from catboost import CatBoostClassifier
        return CatBoostClassifier(iterations=3000, learning_rate=0.05, depth=8,
                                  l2_leaf_reg=3.0, loss_function="MultiClass",
                                  random_seed=seed, thread_count=8,
                                  early_stopping_rounds=100, verbose=False)
    from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
    cls = ExtraTreesClassifier if algo == "et" else RandomForestClassifier
    return cls(n_estimators=1000, n_jobs=8, random_state=seed,
               max_features=0.5, min_samples_leaf=2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="10s_nogro")
    ap.add_argument("--algo", required=True, choices=["xgb", "cat", "et", "rf"])
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rcfg = RunConfig(dataset=args.dataset, protocol="strict")
    rcfg.split.seed = args.seed
    index = SessionIndex(rcfg.dataset_dir, rcfg.bin_size_ms)
    splits = load_splits(rcfg.splits_path)
    print(split_summary(splits))
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
    X = get_X(index, all_sids, args.dataset)
    pos = {s: i for i, s in enumerate(all_sids)}
    A = lambda pool: X[[pos[s] for s in pool]]

    def y_of(pool):
        return np.array([vidx[splits["sessions"][s]["video"]] for s in pool],
                        dtype=np.int64)

    t0 = time.time()
    clf = build_model(args.algo, args.seed, len(known))
    Xtr, ytr = A(pools["train"]), y_of(pools["train"])
    Xva, yva = A(pools["val_probe"]), y_of(pools["val_probe"])
    if args.algo in ("xgb", "cat"):
        clf.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False)
    else:
        clf.fit(Xtr, ytr)
    print(f"[{args.algo}] trained in {time.time() - t0:.0f}s")

    def pred(pool):
        P = clf.predict_proba(A(pool))
        assert P.shape[1] == len(known), P.shape
        return P

    out_dir = PROJECT_ROOT / "runs" / f"tree_zoo_{args.algo}"
    out_dir.mkdir(parents=True, exist_ok=True)
    report = {"algo": args.algo, "splits": {}}
    cache_out = {}
    for split, probes, imps in (("val", pools["val_probe"], pools["val_imp"]),
                                ("test", pools["test_probe"], pools["test_imp"])):
        y = y_of(probes)
        P, Pi = pred(probes), pred(imps)
        m = open_set_metrics(P, y, Pi.max(1), n_boot=1000)
        print(format_open_set_report(f"{split.upper()} [{args.algo}]", m))
        report["splits"][split] = {"prob": m}
        cache_out[f"{split}_probe"] = P
        cache_out[f"{split}_imp"] = Pi
    (out_dir / "report.json").write_text(json.dumps(report, indent=1, default=str),
                                         encoding="utf-8")
    np.savez(PROJECT_ROOT / "runs" / "cache" /
             f"lgbm_probs_{args.dataset}_strict_{args.algo}.npz", **cache_out)
    print(f"[wrote] {out_dir / 'report.json'} + probs cache")


if __name__ == "__main__":
    main()
