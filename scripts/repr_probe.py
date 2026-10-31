# -*- coding: utf-8 -*-
"""Representation probes: where does the video-identity signal live?

P1  kNN on scale-normalized session stats (batch structure + volumes)
P2  kNN on z-scored coarse (100ms) video_cdn down-byte series, zero-lag cosine
P3  same as P2 but with +-500ms best-lag search (FFT xcorr) on a probe subsample
All use the shared split protocol: gallery (3/video) vs test probes.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import RunConfig
from src.data import SessionIndex, load_bins, load_events, load_splits, session_pool
from src.evaluate import closed_set_metrics, open_set_metrics


def build_repr(index, sid, rep):
    bins = load_bins(index, sid)          # (5, T, 4) canonical
    events, chan, meta = load_events(index, sid)
    cdn = bins[0]                          # (T, 4)
    T = cdn.shape[0]
    b_ends = np.asarray(meta.get("batch_end_rel_ms", [T * 10]), dtype=float) / 1000.0
    b1 = b_ends[0] if len(b_ends) else T * 0.01
    b2 = b_ends[1] if len(b_ends) > 1 else b1
    tgrid = np.arange(T) * 0.01
    m1, m2 = tgrid <= b1, (tgrid > b1) & (tgrid <= b2)
    b1_bytes = float(cdn[m1, 0].sum())
    b2_bytes = float(cdn[m2, 0].sum())
    if rep == "stats":
        v = [b1, b2 - b1, T * 0.01, b1_bytes, b2_bytes,
             float(cdn[:, 0].sum()), float(cdn[:, 1].sum()),
             float((cdn[:, 0] > 0).sum()), float((cdn[:, 2] > 0).sum()),
             float(len(events)), float((events[:, 0] < 0).sum()),
             b2_bytes / (b1_bytes + 1e-9), float(cdn[:, 2].sum()) / (float(cdn[:, 3].sum()) + 1)]
        return np.log1p(np.abs(v)) * np.sign(v)
    if rep == "coarse":
        # 100ms bins of video_cdn down bytes, z-scored
        c = cdn[: T // 10 * 10, 0].reshape(-1, 10).sum(1)
        c = np.log1p(c)
        c = (c - c.mean()) / (c.std() + 1e-9)
        return c.astype(np.float32)
    raise ValueError(rep)


def knn_eval(G, Gv, P, Pv, I, gallery_videos, metric="cosine"):
    """G (C, D) class centroids, P (N, D) probes; returns closed + open metrics."""
    if metric == "cosine":
        Gn = G / (np.linalg.norm(G, axis=1, keepdims=True) + 1e-9)
        Pn = P / (np.linalg.norm(P, axis=1, keepdims=True) + 1e-9)
        In = I / (np.linalg.norm(I, axis=1, keepdims=True) + 1e-9)
        mat = Pn @ Gn.T
        imp = (In @ Gn.T).max(1)
    return open_set_metrics(mat.astype(np.float32), Pv, imp.astype(np.float32), n_boot=200)


def pad_stack(vecs):
    L = max(len(v) for v in vecs)
    out = np.zeros((len(vecs), L), dtype=np.float32)
    for i, v in enumerate(vecs):
        out[i, :len(v)] = v
    return out


def xcorr_best(sim_P, sim_G, max_lag_bins=5):
    """max over circular shifts of |xcorr| approximated by direct shifts (coarse bins)."""
    best = np.full((sim_P.shape[0], sim_G.shape[0]), -2.0, dtype=np.float32)
    for lag in range(-max_lag_bins, max_lag_bins + 1):
        P = np.roll(sim_P, lag, axis=1) if lag else sim_P
        best = np.maximum(best, P @ sim_G.T /
                          ((np.linalg.norm(sim_P, axis=1, keepdims=True) + 1e-9) @
                           (np.linalg.norm(sim_G, axis=1, keepdims=True).T + 1e-9)))
    return best


def main(dataset="10s"):
    rcfg = RunConfig(dataset=dataset)
    index = SessionIndex(rcfg.dataset_dir, rcfg.bin_size_ms)
    splits = load_splits(rcfg.splits_path)
    known = sorted(splits["known_videos"])
    vidx = {v: i for i, v in enumerate(known)}
    gal = session_pool(splits, "known", "test", gallery=True)
    probes = session_pool(splits, "known", "test", gallery=False)
    imps = session_pool(splits, "unknown", "test")
    print(f"gallery={len(gal)} probes={len(probes)} impostors={len(imps)}")

    for rep in ["stats", "coarse"]:
        t0 = time.time()
        G_v = [splits["sessions"][s]["video"] for s in gal]
        Pv = np.array([vidx[splits["sessions"][s]["video"]] for s in probes], dtype=np.int64)
        Xg_list = [build_repr(index, s, rep) for s in gal]
        Xp_list = [build_repr(index, s, rep) for s in probes]
        Xi_list = [build_repr(index, s, rep) for s in imps]
        if rep == "stats":
            Xg, Xp, Xi = (np.stack(Xg_list), np.stack(Xp_list), np.stack(Xi_list))
        else:
            width = max(max(len(v) for v in Xg_list), max(len(v) for v in Xp_list),
                        max(len(v) for v in Xi_list))
            Xg = pad_stack([np.pad(v, (0, width - len(v))) for v in Xg_list])
            Xp = pad_stack([np.pad(v, (0, width - len(v))) for v in Xp_list])
            Xi = pad_stack([np.pad(v, (0, width - len(v))) for v in Xi_list])
        # class centroids in feature space
        C = len(known)
        if rep == "stats":
            D = Xg.shape[1]
            cent = np.zeros((C, D), dtype=np.float64)
            cnt = np.zeros(C)
            for x, v in zip(Xg, G_v):
                cent[vidx[v]] += x
                cnt[vidx[v]] += 1
            cent /= np.maximum(cnt[:, None], 1)
            m = knn_eval(cent, None, Xp, Pv, Xi, known)
        else:
            # z-scored already; centroid = mean then renormalize
            D = Xg.shape[1]
            cent = np.zeros((C, D), dtype=np.float64)
            cnt = np.zeros(C)
            for x, v in zip(Xg, G_v):
                cent[vidx[v]] += x
                cnt[vidx[v]] += 1
            cent /= np.maximum(cnt[:, None], 1)
            m = knn_eval(cent, None, Xp, Pv, Xi, known)
        print(f"== P[{rep}] ({time.time() - t0:.0f}s) ==")
        print(f"  closed top1={m['closed']['top1']:.4f} top5={m['closed']['top5']:.4f} "
              f"TPR@1%FPR={m['fpr'][0.01]['tpr']:.4f} AUROC={m['auroc_detection']:.4f}")

    # P3: alignment search on subsample (reuse Xg/Xp from the 'coarse' loop iteration)
    t0 = time.time()
    rng = np.random.default_rng(0)
    sub = rng.choice(len(probes), 200, replace=False)
    mat = xcorr_best(Xp[sub], Xg, max_lag_bins=5)
    Pv = np.array([vidx[splits["sessions"][s]["video"]] for s in probes], dtype=np.int64)[sub]
    top1 = float(np.mean(mat.argmax(1) == Pv))
    print(f"== P[coarse+lag search ±500ms] top1={top1:.4f} ({time.time() - t0:.0f}s) ==")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "10s")
