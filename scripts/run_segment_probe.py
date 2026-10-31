# -*- coding: utf-8 -*-
"""Probe the segment-fingerprint representation: kNN + LightGBM on 10s dataset."""
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import RunConfig
from src.data import SessionIndex, load_splits, session_pool
from src.evaluate import open_set_metrics
from src.segment import segment_features


def main(dataset="10s"):
    rcfg = RunConfig(dataset=dataset)
    index = SessionIndex(rcfg.dataset_dir, rcfg.bin_size_ms)
    splits = load_splits(rcfg.splits_path)
    known = sorted(splits["known_videos"])
    vidx = {v: i for i, v in enumerate(known)}

    pools = dict(
        train=session_pool(splits, "known", "train"),
        val_gal=session_pool(splits, "known", "val", gallery=True),
        val_probe=session_pool(splits, "known", "val", gallery=False),
        val_imp=session_pool(splits, "unknown", "val"),
        test_gal=session_pool(splits, "known", "test", gallery=True),
        test_probe=session_pool(splits, "known", "test", gallery=False),
        test_imp=session_pool(splits, "unknown", "test"),
    )
    all_sids = sorted({s for v in pools.values() for s in v})
    print(f"computing segment features for {len(all_sids)} sessions ...")
    t0 = time.time()
    X_all = np.stack([segment_features(index, s) for s in all_sids])
    pos = {s: i for i, s in enumerate(all_sids)}
    print(f"done in {time.time() - t0:.0f}s, dim={X_all.shape[1]}")

    def arr(pool):
        return X_all[[pos[s] for s in pool]]

    # ---- probe 1: max-similarity over gallery sessions (handles multi-modal classes)
    for split in ("val", "test"):
        G, P, I = arr(pools[f"{split}_gal"]), arr(pools[f"{split}_probe"]), arr(pools[f"{split}_imp"])
        Gv = [splits["sessions"][s]["video"] for s in pools[f"{split}_gal"]]
        y = np.array([vidx[splits["sessions"][s]["video"]] for s in pools[f"{split}_probe"]],
                     dtype=np.int64)
        # z-score columns using gallery stats for cosine comparability
        mu, sd = G.mean(0), G.std(0) + 1e-6
        Gz, Pz, Iz = (G - mu) / sd, (P - mu) / sd, (I - mu) / sd
        sim_pg = Pz @ Gz.T / (np.linalg.norm(Pz, axis=1, keepdims=True) *
                              np.linalg.norm(Gz, axis=1, keepdims=True).T + 1e-9)
        sim_ig = Iz @ Gz.T / (np.linalg.norm(Iz, axis=1, keepdims=True) *
                              np.linalg.norm(Gz, axis=1, keepdims=True).T + 1e-9)
        col = {}
        mat = np.full((len(P), len(known)), -1.0, dtype=np.float32)
        imp = np.full(len(I), -1.0, dtype=np.float32)
        for j, v in enumerate(Gv):
            c = vidx[v]
            if c not in col:
                col[c] = []
            col[c].append(j)
        for c, js in col.items():
            mat[:, c] = sim_pg[:, js].max(axis=1)
            imp = np.maximum(imp, sim_ig[:, js].max(axis=1))
        m = open_set_metrics(mat, y, imp, n_boot=200)
        print(f"[{split} kNN-max over gallery] top1={m['closed']['top1']:.4f} "
              f"top5={m['closed']['top5']:.4f} AUROC={m['auroc_detection']:.4f} "
              f"TPR@0.5%={m['fpr'][0.005]['tpr']:.4f} TPR@1%={m['fpr'][0.01]['tpr']:.4f} "
              f"TPR@5%={m['fpr'][0.05]['tpr']:.4f}")

    # ---- probe 2: LightGBM on segment features
    import lightgbm as lgb
    Xtr, ytr = arr(pools["train"]), np.array(
        [vidx[splits["sessions"][s]["video"]] for s in pools["train"]], dtype=np.int64)
    Xva, yva = arr(pools["val_probe"]), np.array(
        [vidx[splits["sessions"][s]["video"]] for s in pools["val_probe"]], dtype=np.int64)
    Xte, yte = arr(pools["test_probe"]), np.array(
        [vidx[splits["sessions"][s]["video"]] for s in pools["test_probe"]], dtype=np.int64)
    params = dict(objective="multiclass", num_class=len(known), metric="multi_logloss",
                  learning_rate=0.05, num_leaves=63, min_data_in_leaf=20,
                  feature_fraction=0.7, bagging_fraction=0.8, bagging_freq=1,
                  lambda_l2=1.0, verbosity=-1, num_threads=8, seed=42)
    model = lgb.train(params, lgb.Dataset(Xtr, ytr), num_boost_round=400,
                      valid_sets=[lgb.Dataset(Xva, yva)],
                      callbacks=[lgb.early_stopping(50, verbose=False)])
    for split, X_, y_ in [("val", Xva, yva), ("test", Xte, yte)]:
        prob = model.predict(X_)
        imp = model.predict(arr(pools[f"{split}_imp" if split == "test" else "val_imp"]))
        m = open_set_metrics(prob, y_, imp.max(axis=1), n_boot=200)
        print(f"[{split} LGBM] top1={m['closed']['top1']:.4f} top5={m['closed']['top5']:.4f} "
              f"AUROC={m['auroc_detection']:.4f} "
              f"TPR@0.5%={m['fpr'][0.005]['tpr']:.4f} TPR@1%={m['fpr'][0.01]['tpr']:.4f} "
              f"TPR@5%={m['fpr'][0.05]['tpr']:.4f}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "10s")
