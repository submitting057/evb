# -*- coding: utf-8 -*-
"""LGBM on packet-aware segment fingerprints v3 (GRO re-split data).

Same protocol/scorers as run_lgbm_v2; features see per-packet structure.

Scorers:
  prob : max class probability (weak, standard floor)
  leaf : cosine similarity in LGBM leaf-embedding space (kNN over gallery)
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import lightgbm as lgb

from src.config import PROJECT_ROOT, RunConfig
from src.data import SessionIndex, load_splits, session_pool, split_summary
from src.evaluate import format_open_set_report, open_set_metrics
from src.segment import segment_features_v3


def get_X(index, sids, cache_key):
    cache = PROJECT_ROOT / "runs" / "cache" / f"segv3_{cache_key}.npy"
    if cache.exists():
        X = np.load(cache)
        if X.shape[0] == len(sids):
            return X
    t0 = time.time()
    X = np.stack([segment_features_v3(index, s) for s in sids])
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache, X)
    print(f"[feats] {len(sids)} sessions in {time.time() - t0:.0f}s dim={X.shape[1]}")
    return X


def main(dataset="10s", protocol="legacy", seed=42):
    rcfg = RunConfig(dataset=dataset, protocol=protocol)
    rcfg.split.seed = seed
    index = SessionIndex(rcfg.dataset_dir, rcfg.bin_size_ms)
    splits = load_splits(rcfg.splits_path)
    print(split_summary(splits))
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
    X = get_X(index, all_sids, dataset)
    pos = {s: i for i, s in enumerate(all_sids)}
    A = lambda pool: X[[pos[s] for s in pool]]

    def y_of(pool):
        return np.array([vidx[splits["sessions"][s]["video"]] for s in pool], dtype=np.int64)

    params = dict(objective="multiclass", num_class=len(known), metric="multi_logloss",
                  learning_rate=0.05, num_leaves=63, min_data_in_leaf=20,
                  feature_fraction=0.7, bagging_fraction=0.8, bagging_freq=1,
                  lambda_l2=1.0, verbosity=-1, num_threads=8, seed=seed)
    t0 = time.time()
    model = lgb.train(params, lgb.Dataset(A(pools["train"]), y_of(pools["train"])),
                      num_boost_round=1500,
                      valid_sets=[lgb.Dataset(A(pools["val_probe"]), y_of(pools["val_probe"]))],
                      callbacks=[lgb.early_stopping(100, verbose=False)])
    print(f"trained in {time.time() - t0:.0f}s best_iter={model.best_iteration}")

    report = {"config": rcfg.to_dict(), "best_iteration": model.best_iteration,
              "n_features": int(X.shape[1]), "splits": {}}
    probs_cache = {}                        # strict protocol: feed run_combiner

    for split in ("val", "test"):
        gal = pools[f"{split}_gal"]
        probes = pools[f"{split}_probe"]
        imps = pools[f"{split}_imp"]
        y = y_of(probes)
        results = {}

        # open-set scorers built on the probability vector
        from src.evaluate import open_set_with_custom_score
        prob = model.predict(A(probes), num_iteration=model.best_iteration)
        prob_i = model.predict(A(imps), num_iteration=model.best_iteration)
        probs_cache[f"{split}_probe"] = prob
        probs_cache[f"{split}_imp"] = prob_i

        # 1) max probability
        results["prob"] = open_set_metrics(prob, y, prob_i.max(1), n_boot=1000)

        # 2) top1-top2 margin
        margin = np.partition(prob, -2, 1)[:, -1] - np.partition(prob, -2, 1)[:, -2]
        margin_i = np.partition(prob_i, -2, 1)[:, -1] - np.partition(prob_i, -2, 1)[:, -2]
        m = open_set_with_custom_score(prob, y, margin, margin_i)
        m["n_probes"], m["n_impostors"] = len(y), len(prob_i)
        m["auroc_detection"] = 0.0
        results["margin"] = m

        # 3) negative entropy (confidence)
        def nent(P):
            return -np.sum(np.log(np.clip(P, 1e-12, 1)) * P, axis=1)
        m = open_set_with_custom_score(prob, y, nent(prob), nent(prob_i))
        m["n_probes"], m["n_impostors"] = len(y), len(prob_i)
        m["auroc_detection"] = 0.0
        results["nent"] = m

        # 4) prob-profile agreement: cosine of probe prob vector to the video's
        #    mean gallery prob vector (class score), open-set score = max over classes
        gal_prob = model.predict(A(gal), num_iteration=model.best_iteration)
        gv = [splits["sessions"][s]["video"] for s in gal]
        prof = {}
        for j, v in enumerate(gv):
            prof.setdefault(vidx[v], []).append(j)

        def norm(M):
            return M / (np.linalg.norm(M, axis=1, keepdims=True) + 1e-12)

        mat = np.full((len(probes), len(known)), -1.0, dtype=np.float32)
        imp_sc = np.full(len(imps), -1.0, dtype=np.float32)
        Pn, In_ = norm(prob), norm(prob_i)
        for c, js in prof.items():
            T = norm(gal_prob[js].mean(0, keepdims=True))
            mat[:, c] = (Pn @ T.T)[:, 0]
            imp_sc = np.maximum(imp_sc, (In_ @ T.T)[:, 0])
        results["profile"] = open_set_metrics(mat, y, imp_sc, n_boot=1000)

        for mode, m in results.items():
            print(format_open_set_report(f"{split.upper()} [{mode}]", m))
        report["splits"][split] = results

    out_dir = PROJECT_ROOT / "runs" / (f"{dataset}_lgbmv3_strict"
                                       if protocol == "strict"
                                       else f"{dataset}_lgbmv3_s{seed}")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "report.json").write_text(json.dumps(report, indent=1, default=str),
                                         encoding="utf-8")
    if protocol == "strict":
        np.savez(PROJECT_ROOT / "runs" / "cache" / f"lgbm_probs_{dataset}_strict_v3.npz",
                 best_iter=model.best_iteration, **probs_cache)
        print(f"[wrote] {out_dir / 'report.json'} + probs cache (strict)")
    else:
        print(f"[wrote] {out_dir / 'report.json'}")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="10s")
    ap.add_argument("--protocol", default="legacy", choices=["legacy", "strict"])
    a = ap.parse_args()
    main(a.dataset, a.protocol)
