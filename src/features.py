# -*- coding: utf-8 -*-
"""Hand-crafted statistical features per session + LightGBM baseline (pipeline C)."""
import time

import numpy as np

from .config import SEMANTIC_CHANNELS
from .data import SessionIndex, load_bins, load_events

PCT = (10, 25, 50, 75, 90, 99)


def _series_stats(x: np.ndarray, prefix: str, out: dict):
    if x.size == 0:
        for p in PCT:
            out[f"{prefix}_p{p}"] = 0.0
        out[f"{prefix}_mean"] = 0.0
        out[f"{prefix}_std"] = 0.0
        out[f"{prefix}_max"] = 0.0
        return
    out[f"{prefix}_mean"] = float(np.mean(x))
    out[f"{prefix}_std"] = float(np.std(x))
    out[f"{prefix}_max"] = float(np.max(x))
    for p in PCT:
        out[f"{prefix}_p{p}"] = float(np.percentile(x, p))


def _burst_stats(active: np.ndarray, bytes_per_bin: np.ndarray, prefix: str, out: dict):
    """Bursts = runs of active bins; length & byte mass stats."""
    if active.size == 0:
        out[f"{prefix}_n_bursts"] = 0
        for q in ("len", "bytes"):
            for s in ("max", "mean"):
                out[f"{prefix}_burst_{q}_{s}"] = 0.0
        return
    padded = np.concatenate([[False], active, [False]]).astype(np.int8)
    diff = np.diff(padded)
    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0]
    lens = ends - starts
    csum = np.concatenate([[0.0], np.cumsum(bytes_per_bin)])
    mass = csum[ends] - csum[starts]
    out[f"{prefix}_n_bursts"] = int(len(starts))
    out[f"{prefix}_burst_len_max"] = float(lens.max()) if len(lens) else 0.0
    out[f"{prefix}_burst_len_mean"] = float(lens.mean()) if len(lens) else 0.0
    out[f"{prefix}_burst_bytes_max"] = float(mass.max()) if len(mass) else 0.0
    out[f"{prefix}_burst_bytes_mean"] = float(mass.mean()) if len(mass) else 0.0


def session_features(index: SessionIndex, session_id: str) -> dict:
    feats = {}
    bins = load_bins(index, session_id)          # (5, T, 4) canonical channels
    events, chan, meta = load_events(index, session_id)

    total_down = np.zeros(bins.shape[1])
    for ci, cname in enumerate(SEMANTIC_CHANNELS):
        down_b, up_b = bins[ci, :, 0], bins[ci, :, 1]
        down_p, up_p = bins[ci, :, 2], bins[ci, :, 3]
        pre = f"ch_{cname}"
        feats[f"{pre}_down_bytes"] = float(down_b.sum())
        feats[f"{pre}_up_bytes"] = float(up_b.sum())
        feats[f"{pre}_down_pkts"] = float(down_p.sum())
        feats[f"{pre}_up_pkts"] = float(up_p.sum())
        feats[f"{pre}_nz_bins"] = float((down_b > 0).sum())
        _series_stats(down_b[down_b > 0] if (down_b > 0).any() else down_b[:0],
                      f"{pre}_down_binbytes", feats)
        _burst_stats(down_b > 0, down_b, pre, feats)
        if ci < 4:  # 'other' is empty everywhere; skip its contribution to total
            total_down += down_b
    _series_stats(total_down[total_down > 0] if (total_down > 0).any() else total_down[:0],
                  "total_down_binbytes", feats)
    _burst_stats(total_down > 0, total_down, "total", feats)

    # event-level stats
    size = events[:, 0]
    down, up = size > 0, size < 0
    feats["n_events"] = float(len(events))
    feats["n_down"], feats["n_up"] = float(down.sum()), float(up.sum())
    feats["down_bytes"] = float(size[down].sum())
    feats["up_bytes"] = float(-size[up].sum())
    feats["up_down_byte_ratio"] = float(feats["up_bytes"] / (feats["down_bytes"] + 1.0))
    if down.any():
        _series_stats(size[down], "down_size", feats)
        _series_stats(events[down, 1], "down_dt", feats)
    if up.any():
        _series_stats(-size[up], "up_size", feats)
        _series_stats(events[up, 1], "up_dt", feats)
    feats["duration_s"] = float(bins.shape[1] * 0.01)
    feats["events_per_s"] = feats["n_events"] / (feats["duration_s"] + 1e-6)

    # DASH batch timing structure (10s dataset only)
    for key in ("batch_end_rel_ms",):
        if key in meta:
            arr = np.asarray(meta[key], dtype=float)
            for j, v in enumerate(arr[:2]):
                feats[f"batch{j + 1}_end_ms"] = float(v)
            if len(arr) >= 2:
                feats["batch_gap_ms"] = float(arr[1] - arr[0])
    return feats


def build_feature_matrix(index: SessionIndex, session_ids: list[str, ...], verbose_every: int = 2000):
    """Returns (X float32, feature_names)."""
    dicts, names = [], None
    t0 = time.time()
    for i, sid in enumerate(session_ids):
        d = session_features(index, sid)
        dicts.append(d)
        if verbose_every and (i + 1) % verbose_every == 0:
            print(f"  features {i + 1}/{len(session_ids)} ({time.time() - t0:.0f}s)", flush=True)
    names = sorted(dicts[0].keys())
    X = np.zeros((len(dicts), len(names)), dtype=np.float32)
    for i, d in enumerate(dicts):
        for j, n in enumerate(names):
            X[i, j] = d.get(n, 0.0)
    return X, names
