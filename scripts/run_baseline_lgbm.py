# -*- coding: utf-8 -*-
"""Pipeline C: statistical features + LightGBM (closed-set multi-class, open-set via max prob)."""
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import PROJECT_ROOT, RunConfig
from src.data import SessionIndex, load_splits, session_pool, split_summary
from src.evaluate import format_open_set_report, open_set_metrics
from src.features import build_feature_matrix


def get_features(index: SessionIndex, splits: dict, dataset: str) -> dict:
    """Feature matrix for ALL sessions, cached on disk keyed by dataset."""
    cache = PROJECT_ROOT / "runs" / "cache" / f"stats_features_{dataset}.npz"
    cache.parent.mkdir(parents=True, exist_ok=True)
    all_sids = list(splits["sessions"].keys())
    if cache.exists():
        z = np.load(cache, allow_pickle=True)
        cached_ids = list(z["session_ids"])
        if cached_ids == all_sids:
            print(f"[cache] {cache}")
            return {"X": z["X"], "names": list(z["names"]), "session_ids": cached_ids}
    print(f"[build] features for {len(all_sids)} sessions ...")
    X, names = build_feature_matrix(index, all_sids)
    np.savez_compressed(cache, X=X, names=np.array(names, dtype=object),
                        session_ids=np.array(all_sids, dtype=object))
    return {"X": X, "names": names, "session_ids": all_sids}


def main(dataset: str = "10s", seed: int = 42):
    cfg = RunConfig(dataset=dataset)
    cfg.split.seed = seed
    index = SessionIndex(cfg.dataset_dir, cfg.bin_size_ms)
    splits = load_splits(cfg.splits_path)
    print(split_summary(splits))

    feats = get_features(index, splits, dataset)
    X, sids = feats["X"], feats["session_ids"]
    pos = {sid: i for i, sid in enumerate(sids)}
    video_of = splits["sessions"]
    known_videos = sorted(splits["known_videos"])
    cls_idx = {v: i for i, v in enumerate(known_videos)}

    def pack(pool):
        rows = [pos[sid] for sid in pool]
        y = np.array([cls_idx[video_of[sid]["video"]] for sid in pool], dtype=np.int64)
        return X[rows], y

    tr_sids = session_pool(splits, "known", "train")
    va_sids = session_pool(splits, "known", "val")
    te_sids = session_pool(splits, "known", "test", gallery=False)
    imp_va = session_pool(splits, "unknown", "val")
    imp_te = session_pool(splits, "unknown", "test")
    Xtr, ytr = pack(tr_sids)
    Xva, yva = pack(va_sids)
    Xte, yte = pack(te_sids)
    Xiv = X[[pos[s] for s in imp_va]]
    Xit = X[[pos[s] for s in imp_te]]
    print(f"train={Xtr.shape} val={Xva.shape} test={Xte.shape} classes={len(known_videos)}")

    import lightgbm as lgb
    params = dict(objective="multiclass", num_class=len(known_videos),
                  metric="multi_logloss", learning_rate=0.1, num_leaves=127,
                  feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1,
                  min_data_in_leaf=5, verbosity=-1, num_threads=8, seed=seed)
    t0 = time.time()
    model = lgb.train(params, lgb.Dataset(Xtr, ytr), num_boost_round=300,
                      valid_sets=[lgb.Dataset(Xva, yva)],
                      callbacks=[lgb.early_stopping(50, verbose=True), lgb.log_evaluation(50)])
    print(f"trained in {time.time() - t0:.0f}s, best_iter={model.best_iteration}")

    def score_matrix(X_):
        return model.predict(X_, num_iteration=model.best_iteration)

    # model selection on val, final numbers on test
    va_mat = score_matrix(Xva)
    m_val = open_set_metrics(va_mat, yva, score_matrix(Xiv).max(axis=1))
    print(format_open_set_report("VAL (lgbm max-prob)", m_val))
    te_mat = score_matrix(Xte)
    m_test = open_set_metrics(te_mat, yte, score_matrix(Xit).max(axis=1))
    print(format_open_set_report("TEST (lgbm max-prob)", m_test))

    out_dir = PROJECT_ROOT / "runs" / f"{dataset}_lgbm_s{seed}"
    out_dir.mkdir(parents=True, exist_ok=True)
    report = {"config": cfg.to_dict(), "best_iteration": model.best_iteration,
              "n_features": X.shape[1],
              "feature_importance_top20": sorted(
                  zip(feats["names"], model.feature_importance().tolist()),
                  key=lambda kv: -kv[1])[:20],
              "val": m_val, "test": m_test}
    (out_dir / "report.json").write_text(json.dumps(report, indent=1, default=str),
                                         encoding="utf-8")
    print(f"[wrote] {out_dir / 'report.json'}")


if __name__ == "__main__":
    ds = sys.argv[1] if len(sys.argv) > 1 else "10s"
    main(ds)
