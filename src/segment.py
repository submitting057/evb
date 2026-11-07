# -*- coding: utf-8 -*-
"""Segment-level fingerprint representation.

DASH segments are recovered from the video_cdn down-event stream by cutting at
inter-event gaps > SEG_GAP_S. Segment masses (bytes) are strongly video-specific
and stable across sessions of the same video; auto-quality rescales them
globally, so features are scale-normalized.
"""
import numpy as np

from .config import SEMANTIC_CHANNELS
from .data import SessionIndex, load_events

CDN = SEMANTIC_CHANNELS.index("video_cdn")
SEG_GAP_S = 0.045          # cut cdn down-stream at gaps > 45ms
MIN_SEG_BYTES = 2000       # ignore fragments below this mass
MAX_SEGS = 12              # keep first K segments
MAX_SEGS_V2 = 16           # v2 feature vector keeps more segments


def segment_features_v2(index: SessionIndex, session_id: str) -> np.ndarray:
    """Richer fixed-dim segment fingerprint for gradient-boosted models.

    Layout (MAX_SEGS_V2 * 7 + 8):
      per segment: log mass, log(mass/total), mass/first_mass, log onset,
                   log idle-before, log duration, log(rate = mass/dur)
      global: n_segs, log total, log(mass1/mass2), log median ratio,
              session duration, log n_events, log up bytes, max idle
    """
    times, masses, before, durs = segment_stream(index, session_id)
    n = min(len(masses), MAX_SEGS_V2)
    feats = np.zeros(MAX_SEGS_V2 * 7 + 8, dtype=np.float32)
    if n == 0:
        return feats
    total = masses.sum()
    first = masses[0]
    med = np.median(masses[:n])
    for i in range(n):
        b = i * 7
        dur = max(1e-3, durs[i])
        feats[b] = np.log(masses[i])
        feats[b + 1] = np.log(masses[i] / total)
        feats[b + 2] = masses[i] / first
        feats[b + 3] = np.log(times[i] + 1e-3)
        feats[b + 4] = np.log(before[i] + 1e-3)
        feats[b + 5] = np.log(dur)
        feats[b + 6] = np.log(masses[i] / dur)
    onsets = times[:n]
    feats[-8] = n
    feats[-7] = np.log(total)
    feats[-6] = np.log(masses[1] / first) if n > 1 else 0.0
    feats[-5] = np.log(med / first)
    feats[-4] = onsets[-1]
    feats[-3] = np.log(max(1e-3, before[:n].max()))
    feats[-2] = np.log(total / max(1e-3, onsets[-1]))   # avg rate over span
    feats[-1] = np.log(max(1e-3, durs[:n].max()))
    return feats


def segment_features_v3(index: SessionIndex, session_id: str) -> np.ndarray:
    """Packet-aware segment fingerprint (GRO re-split data only).

    The GRO re-split restores per-packet structure inside each DASH segment:
    packet counts, size dispersion (MTU plateau + remainder) and inter-packet
    timing.  v2 keeps only segment masses/durations, which are byte-conserved
    under re-splitting -- v3 adds the packet-level evidence v2 cannot see.

    Layout (MAX_SEGS_V2 * 12 + 12):
      per segment: v2's 7 (log mass, log share, mass/first, log onset,
                   log idle-before, log dur, log rate)
                   + log n_pkts, log mean pkt, log std pkt,
                     log median dt, log max gap
      global: v2's 8 + log total pkts, log mean pkt size, frac MTU-full,
              median inter-pkt dt
    """
    ev, chan, _ = load_events(index, session_id)
    m = (chan == CDN) & (ev[:, 0] > 0)
    feats = np.zeros(MAX_SEGS_V2 * 12 + 12, dtype=np.float32)
    if not m.any():
        return feats
    d = ev[m]
    t, sz = d[:, 3] / 1000.0, d[:, 0]
    gaps = np.diff(t, prepend=t[0])
    cuts = np.where(gaps > SEG_GAP_S)[0]
    starts = np.concatenate([[0], cuts])
    ends = np.concatenate([cuts, [len(t)]])
    segs = []
    for s, e in zip(starts, ends):
        mass = float(sz[s:e].sum())
        if mass >= MIN_SEG_BYTES:
            segs.append((s, e))
    if not segs:
        return feats
    total = sum(float(sz[s:e].sum()) for s, e in segs)
    first = float(sz[segs[0][0]:segs[0][1]].sum())
    n = min(len(segs), MAX_SEGS_V2)
    all_dt = []
    for i, (s, e) in enumerate(segs[:n]):
        b = i * 12
        p = sz[s:e]
        dt = np.diff(t[s:e])
        mass = float(p.sum())
        dur = max(1e-3, float(t[e - 1] - t[s]))
        feats[b] = np.log(mass)
        feats[b + 1] = np.log(mass / total)
        feats[b + 2] = mass / first
        feats[b + 3] = np.log(t[s] + 1e-3)
        feats[b + 4] = np.log(float(gaps[s]) + 1e-3) if s else 0.0
        feats[b + 5] = np.log(dur)
        feats[b + 6] = np.log(mass / dur)
        feats[b + 7] = np.log(len(p))
        feats[b + 8] = np.log(p.mean())
        feats[b + 9] = np.log(p.std() + 1.0)
        feats[b + 10] = np.log(np.median(dt) + 1e-4) if len(dt) else 0.0
        feats[b + 11] = np.log(dt.max() + 1e-4) if len(dt) else 0.0
        if len(dt):
            all_dt.append(float(np.median(dt)))
    onsets = [t[s] for s, _ in segs[:n]]
    masses = np.array([float(sz[s:e].sum()) for s, e in segs[:n]])
    g = MAX_SEGS_V2 * 12
    feats[g] = n
    feats[g + 1] = np.log(total)
    feats[g + 2] = np.log(masses[1] / first) if n > 1 else 0.0
    feats[g + 3] = np.log(np.median(masses) / first)
    feats[g + 4] = onsets[-1]
    feats[g + 5] = np.log(max(1e-3, float(gaps[segs[:n][0][0]:].max()) if n else 1e-3))
    feats[g + 6] = np.log(total / max(1e-3, onsets[-1]))
    feats[g + 7] = np.log(max(1e-3, float(max(
        t[e - 1] - t[s] for s, e in segs[:n]))))
    feats[g + 8] = np.log(len(sz))
    feats[g + 9] = np.log(sz.mean())
    feats[g + 10] = float(np.mean(sz >= 1200))
    feats[g + 11] = np.log(np.median(all_dt) + 1e-4) if all_dt else 0.0
    return feats


def segment_tokens(index: SessionIndex, session_id: str):
    """(n, 5) float32 tokens for the segs representation:
    [log mass, log(mass/total), log onset, log idle-before, log duration]."""
    ev, chan, _ = load_events(index, session_id)
    m = (chan == CDN) & (ev[:, 0] > 0)
    d = ev[m]
    if len(d) == 0:
        return np.zeros((0, 5), dtype=np.float32)
    t, sz = d[:, 3] / 1000.0, d[:, 0]
    gaps = np.diff(t, prepend=t[0])
    cuts = np.where(gaps > SEG_GAP_S)[0]
    starts = np.concatenate([[0], cuts])
    ends = np.concatenate([cuts, [len(t)]])
    rows, total = [], 0.0
    seg_list = []
    for s, e in zip(starts, ends):
        mass = float(sz[s:e].sum())
        if mass >= MIN_SEG_BYTES:
            seg_list.append((float(t[s]), mass, float(t[e - 1] - t[s]), float(gaps[s]) if s else 0.0))
    if not seg_list:
        return np.zeros((0, 5), dtype=np.float32)
    total = sum(x[1] for x in seg_list)
    for onset, mass, dur, before in seg_list[:MAX_SEGS]:
        rows.append([np.log(mass), np.log(mass / total), np.log(onset + 1e-3),
                     np.log(before + 1e-3), np.log(dur + 1e-3)])
    return np.asarray(rows, dtype=np.float32)


def segment_stream(index: SessionIndex, session_id: str):
    """Returns (times, masses, gaps_before, durations) of cdn down segments."""
    ev, chan, _ = load_events(index, session_id)
    m = (chan == CDN) & (ev[:, 0] > 0)
    d = ev[m]
    if len(d) == 0:
        return np.array([]), np.array([]), np.array([]), np.array([])
    t, sz = d[:, 3] / 1000.0, d[:, 0]
    gaps = np.diff(t, prepend=t[0])
    cuts = np.where(gaps > SEG_GAP_S)[0]
    starts = np.concatenate([[0], cuts])
    ends = np.concatenate([cuts, [len(t)]])
    times, masses, before, durs = [], [], [], []
    for s, e in zip(starts, ends):
        mass = float(sz[s:e].sum())
        if mass >= MIN_SEG_BYTES:
            times.append(float(t[s]))
            masses.append(mass)
            before.append(float(gaps[s]) if s > 0 else 0.0)
            durs.append(float(t[e - 1] - t[s]))
    return np.array(times), np.array(masses), np.array(before), np.array(durs)


def segment_features(index: SessionIndex, session_id: str) -> np.ndarray:
    """Fixed-dim scale-normalized segment fingerprint (for LGBM / kNN probes)."""
    times, masses, before, _ = segment_stream(index, session_id)
    n = min(len(masses), MAX_SEGS)
    feats = np.zeros(MAX_SEGS * 4 + 2, dtype=np.float32)
    if n == 0:
        return feats
    total = masses.sum()
    feats[:n] = np.log(masses[:n]) - np.log(total)      # scale-normalized log masses
    feats[MAX_SEGS:MAX_SEGS + n] = np.log1p(times[:n])   # onset times
    feats[2 * MAX_SEGS:2 * MAX_SEGS + n] = np.log1p(before[:n] * 1000)  # idle ms before seg
    feats[3 * MAX_SEGS:3 * MAX_SEGS + n] = np.log1p(np.diff(
        np.concatenate([[0], times[:n]])) * 1000)        # inter-segment onset spacing
    feats[-2] = n
    feats[-1] = np.log(total)
    return feats
