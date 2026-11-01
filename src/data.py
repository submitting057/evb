# -*- coding: utf-8 -*-
"""Session indexing, split generation and PyTorch datasets.

Data layout (per dataset):
  features/event_sequences/<vid>/<session_id>.npz
      events (N,4) f32 [signed_size, dt_global_s, dt_same_path_s, abs_time_s]
      path_ids (N,) i32  -> semantic channel via metadata.channel_labels
  features/time_bins_{10,20}ms/<vid>/<session_id>.npz
      bins (5, T, 4) f32 [down_B, up_B, down_pkts, up_pkts], channel order DISK_BIN_CHANNELS
"""
import json
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from .config import (BIN_CHANNEL_PERM, MAX_BINS, MAX_EVENTS, SEMANTIC_CHANNELS,
                     RunConfig, SplitConfig)

EPS = 1e-6


# --------------------------------------------------------------------------- index

class SessionIndex:
    """All sessions of one dataset, joined with the manifest."""

    def __init__(self, dataset_dir: Path, bin_size_ms: int = 10):
        self.dir = Path(dataset_dir)
        self.bin_subdir = f"time_bins_{bin_size_ms}ms"
        self.sessions = []  # list of dicts: session_id, video_id, rel_paths
        manifest = self.dir / "manifests" / "sessions.jsonl"
        for line in manifest.open(encoding="utf-8"):
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            sid = r["session_id"]
            self.sessions.append({
                "session_id": sid,
                "video_id": r["video_id"],
                "events": f"features/event_sequences/{r['video_id']}/{sid}.npz",
                "bins": f"features/{self.bin_subdir}/{r['video_id']}/{sid}.npz",
                "status": r.get("quality_status", "ok"),
            })
        self.by_video = defaultdict(list)
        for i, s in enumerate(self.sessions):
            self.by_video[s["video_id"]].append(i)
        self.videos = sorted(self.by_video.keys())
        self._by_id = {s["session_id"]: s for s in self.sessions}

    def __len__(self):
        return len(self.sessions)

    def sessions_by_id(self, session_id: str) -> dict:
        return self._by_id[session_id]


# --------------------------------------------------------------------------- splits

def _alloc_counts(n: int, fracs, mins):
    """Largest-remainder allocation of n items into len(fracs) buckets with minimums."""
    k = len(fracs)
    if n < sum(mins):
        # degenerate: give everything to the largest bucket, others 0
        counts = [0] * k
        counts[max(range(k), key=lambda i: fracs[i])] = n
        return counts
    counts = list(mins)
    remaining = n - sum(counts)
    exact = [f * n for f in fracs]
    for i in range(k):
        exact[i] -= mins[i]
    order = sorted(range(k), key=lambda i: exact[i] - int(exact[i]), reverse=True)
    for i in range(k):
        counts[i] += int(exact[i])
    left = remaining - sum(int(e) for e in exact)
    for i in order[:left]:
        counts[i] += 1
    return counts


def make_splits(index: SessionIndex, cfg: SplitConfig, out_path: Path) -> dict:
    rng = random.Random(cfg.seed)
    videos = list(index.videos)
    rng.shuffle(videos)
    n_known = round(len(videos) * cfg.known_video_frac)
    known_videos = sorted(videos[:n_known])
    unknown_videos = sorted(videos[n_known:])
    known_set = set(known_videos)

    splits = {"dataset_dir": str(index.dir), "seed": cfg.seed,
              "known_videos": known_videos, "unknown_videos": unknown_videos,
              "sessions": {}}
    for vid, sess_idx in index.by_video.items():
        sids = [index.sessions[i]["session_id"] for i in sess_idx]
        rng.shuffle(sids)
        if vid in known_set:
            mins = (2, 1, 1) if len(sids) >= 4 else (1, 1, 1)
            c_tr, c_va, c_te = _alloc_counts(len(sids), (cfg.train_frac, cfg.val_frac, cfg.test_frac), mins)
            parts = [sids[:c_tr], sids[c_tr:c_tr + c_va], sids[c_tr + c_va:]]
            role, part_names = "known", ["train", "val", "test"]
        else:
            c_va = round(len(sids) * cfg.unknown_val_frac)
            c_va = min(max(c_va, 1), len(sids) - 1) if len(sids) >= 2 else 0
            parts = [sids[:c_va], sids[c_va:]]
            role, part_names = "unknown", ["val", "test"]
        for part_name, part in zip(part_names, parts):
            gallery = []
            if role == "known" and part_name in ("val", "test"):
                gallery = sorted(rng.sample(part, min(cfg.gallery_k, len(part))))
            for sid in part:
                splits["sessions"][sid] = {
                    "video": vid, "role": role, "split": part_name,
                    "gallery": sid in gallery,
                }
    splits["gallery_k"] = cfg.gallery_k
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(splits, indent=1), encoding="utf-8")
    return splits


def make_splits_strict(index: SessionIndex, cfg: SplitConfig, out_path: Path) -> dict:
    """Strict open-set protocol: video-disjoint impostor pools (6:2:2).

    644 videos -> 386 known / 129 unknown->val impostors / 129 unknown->test
    impostors.  Unknown videos go WHOLE into their pool (no session split), so
    the calibration (val) and evaluation (test) impostor populations share no
    identity -- the deployment assumption the legacy protocol violates."""
    rng = random.Random(cfg.seed)
    videos = list(index.videos)
    rng.shuffle(videos)
    n_known = round(len(videos) * 0.6)
    n_val = round(len(videos) * 0.2)
    known_videos = sorted(videos[:n_known])
    val_unknown = sorted(videos[n_known:n_known + n_val])
    test_unknown = sorted(videos[n_known + n_val:])
    known_set, val_set = set(known_videos), set(val_unknown)

    splits = {"dataset_dir": str(index.dir), "seed": cfg.seed, "protocol": "strict",
              "known_videos": known_videos,
              "unknown_videos": sorted(val_unknown + test_unknown),
              "unknown_val_videos": val_unknown,
              "unknown_test_videos": test_unknown,
              "sessions": {}}
    for vid, sess_idx in index.by_video.items():
        sids = [index.sessions[i]["session_id"] for i in sess_idx]
        rng.shuffle(sids)
        if vid in known_set:
            mins = (2, 1, 1) if len(sids) >= 4 else (1, 1, 1)
            c_tr, c_va, c_te = _alloc_counts(len(sids), (cfg.train_frac, cfg.val_frac, cfg.test_frac), mins)
            parts = [sids[:c_tr], sids[c_tr:c_tr + c_va], sids[c_tr + c_va:]]
            for part_name, part in zip(["train", "val", "test"], parts):
                gallery = []
                if part_name in ("val", "test"):
                    gallery = sorted(rng.sample(part, min(cfg.gallery_k, len(part))))
                for sid in part:
                    splits["sessions"][sid] = {"video": vid, "role": "known",
                                               "split": part_name,
                                               "gallery": sid in gallery}
        else:
            part_name = "val" if vid in val_set else "test"
            for sid in sids:
                splits["sessions"][sid] = {"video": vid, "role": "unknown",
                                           "split": part_name, "gallery": False}
    splits["gallery_k"] = cfg.gallery_k
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(splits, indent=1), encoding="utf-8")
    return splits


def load_splits(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def split_summary(splits: dict) -> str:
    cnt = defaultdict(int)
    for rec in splits["sessions"].values():
        cnt[(rec["role"], rec["split"])] += 1
    per_vid_test = defaultdict(int)
    for rec in splits["sessions"].values():
        if rec["role"] == "known" and rec["split"] == "test":
            per_vid_test[rec["video"]] += 1
    gallery_vids = len({rec["video"] for rec in splits["sessions"].values()
                        if rec["role"] == "known" and rec["split"] == "test" and rec["gallery"]})
    te = [per_vid_test[v] for v in splits["known_videos"]]
    lines = [
        f"known videos={len(splits['known_videos'])} unknown videos={len(splits['unknown_videos'])}",
        f"sessions: " + ", ".join(f"{r}/{s}={cnt[(r, s)]}" for r, s in
                                  [("known", "train"), ("known", "val"), ("known", "test"),
                                   ("unknown", "val"), ("unknown", "test")]),
        f"known test sessions per video: min={min(te)} median={int(np.median(te))} max={max(te)}",
        f"videos with a test gallery: {gallery_vids}/{len(splits['known_videos'])}",
    ]
    return "\n".join(lines)


def session_pool(splits: dict, role: str, split: str, gallery: bool | None = None) -> list[str]:
    """Session ids for a pool; gallery=None -> all, True -> only gallery, False -> only probes."""
    return [sid for sid, rec in splits["sessions"].items()
            if rec["role"] == role and rec["split"] == split
            and (gallery is None or rec["gallery"] == gallery)]


# --------------------------------------------------------------------------- loading

def load_bins(index: SessionIndex, session_id: str) -> np.ndarray:
    """(5, T, 4) canonical channel order, float32."""
    rel = index.sessions_by_id(session_id)["bins"]
    z = np.load(index.dir / rel)
    return z["bins"][BIN_CHANNEL_PERM].astype(np.float32)


def load_events(index: SessionIndex, session_id: str) -> tuple[np.ndarray, np.ndarray, dict]:
    """events (N,4), channel ids in SEMANTIC_CHANNELS space (5), metadata dict."""
    rel = index.sessions_by_id(session_id)["events"]
    z = np.load(index.dir / rel)
    events = z["events"].astype(np.float32)
    path_ids = z["path_ids"].astype(np.int64)
    meta = json.loads(str(z["metadata"]))
    labels = {int(k): v[0]["channel_name"] for k, v in meta.get("channel_labels", {}).items()}
    name_to_id = {c: i for i, c in enumerate(SEMANTIC_CHANNELS)}
    chan = np.array([name_to_id.get(labels.get(int(p), "other"),
                                    name_to_id["other"]) for p in path_ids], dtype=np.int64)
    return events, chan, meta


def _subsample_mask(n: int, cap: int, rng: random.Random) -> np.ndarray:
    """Boolean mask keeping min(n, cap) items, uniform without replacement, order preserved."""
    if n <= cap:
        return np.ones(n, dtype=bool)
    keep = set(rng.sample(range(n), cap))
    return np.array([i in keep for i in range(n)], dtype=bool)


class PackedDataset(Dataset):
    """Pre-packed in-memory dataset over a session-id list, representation 'bins' or 'events'.

    For rep='segs', train-time augmentation (augment=True) simulates the nuisance
    factors identified in the data diagnosis: global quality rescale, adjacent
    segment merging (no-silence batches), trailing truncation, timing jitter.
    """

    def __init__(self, index: SessionIndex, session_ids: list[str], rep: str,
                 dataset_tag: str, seed: int = 0, augment: bool = False):
        self.augment = augment
        self._aug_rng = random.Random(seed + 777)
        self.rep = rep
        self.session_ids = list(session_ids)
        rng = random.Random(seed)
        if rep == "segs":
            from .segment import MAX_SEGS, segment_tokens
            S = MAX_SEGS
            x = np.zeros((len(self.session_ids), S, 5), dtype=np.float16)
            m = np.zeros((len(self.session_ids), S), dtype=bool)
            for i, sid in enumerate(self.session_ids):
                toks = segment_tokens(index, sid)
                n = min(len(toks), S)
                x[i, :n] = toks[:n].astype(np.float16)
                m[i, :n] = True
            self.x = torch.from_numpy(x)                      # (B, S, 5)
            self.mask = torch.from_numpy(m)                   # (B, S)
        elif rep == "bins":
            t_max = MAX_BINS[dataset_tag]
            feats, masks = [], []
            for sid in self.session_ids:
                b = load_bins(index, sid)                     # (5, T, 4)
                t = min(b.shape[1], t_max)
                if b.shape[1] > t_max:                        # truncate tail, keep head burst
                    b = b[:, :t_max]
                x = np.zeros((len(SEMANTIC_CHANNELS), t_max, 4), dtype=np.float16)
                m = np.zeros(t_max, dtype=bool)
                x[:, :t] = np.log1p(b).astype(np.float16)
                m[:t] = True
                feats.append(x.reshape(len(SEMANTIC_CHANNELS) * 4, t_max))
                masks.append(m)
            self.x = torch.from_numpy(np.stack(feats))        # (B, 20, T)
            self.mask = torch.from_numpy(np.stack(masks))     # (B, T)
        elif rep == "events":
            N = MAX_EVENTS
            x = np.zeros((len(self.session_ids), N, 4), dtype=np.float16)
            ch = np.zeros((len(self.session_ids), N), dtype=np.int16)
            m = np.zeros((len(self.session_ids), N), dtype=bool)
            for i, sid in enumerate(self.session_ids):
                ev, chan, _ = load_events(index, sid)
                keep = _subsample_mask(len(ev), N, rng)
                ev, chan = ev[keep], chan[keep]
                n = len(ev)
                size = ev[:, 0]
                feats = np.stack([
                    np.log1p(np.abs(size)),
                    np.sign(size),
                    np.log1p(ev[:, 1] + EPS),
                    np.log1p(ev[:, 2] + EPS),
                ], axis=1).astype(np.float16)
                x[i, :n] = feats
                ch[i, :n] = chan.astype(np.int16)
                m[i, :n] = True
            self.x = torch.from_numpy(x)                      # (B, N, 4)
            self.ch = torch.from_numpy(ch).long()             # (B, N)
            self.mask = torch.from_numpy(m)                   # (B, N)
        else:
            raise ValueError(rep)

    def __len__(self):
        return len(self.session_ids)

    def _augment_segs(self, x: torch.Tensor, mask: torch.Tensor):
        """x (S, 5) float tensor: [log mass, log(mass/total), log onset, log idle, log dur]."""
        rng = self._aug_rng
        n = int(mask.sum())
        t = x.numpy().astype(np.float32)   # x is float16; exp below overflows in fp16
        # 1) trailing truncation: keep first k segments (observation budget)
        if n > 3 and rng.random() < 0.3:
            n = rng.randint(3, n)
            t[n:] = 0.0
            t[:n, 1] = 0.0
        # 2) adjacent merges (no-silence batch downloads)
        keep = []
        i = 0
        while i < n:
            j = i
            while j + 1 < n and rng.random() < 0.18:
                j += 1
            keep.append((i, j))
            i = j + 1
        if len(keep) < n:
            new_t = np.zeros_like(t)
            masses = np.exp(t[:n, 0])
            onsets = np.exp(t[:n, 2])
            durs = np.exp(t[:n, 4])
            out_n = 0
            for (a, b) in keep:
                m = masses[a:b + 1].sum()
                new_t[out_n, 0] = np.log(m)
                new_t[out_n, 2] = t[a, 2]
                new_t[out_n, 3] = t[a, 3]
                end = onsets[b] + durs[b]
                new_t[out_n, 4] = np.log(max(1e-3, end - onsets[a]))
                out_n += 1
            n = out_n
            t = new_t
            t[n:] = 0.0
        # 3) global quality rescale on absolute log-mass (ratio feature invariant already)
        t[:n, 0] += rng.uniform(-0.8, 0.8)
        # 4) timing jitter
        t[:n, 2] += rng.uniform(-0.06, 0.06)
        t[:n, 3] += rng.uniform(-0.2, 0.2)
        t[:n, 4] += rng.uniform(-0.1, 0.1)
        # 5) light feature noise
        t[:n] += np.array([self._aug_rng.gauss(0, 0.03) for _ in range(n * 5)]).reshape(n, 5)
        # recompute log(mass/total) after merges/rescale
        tot = np.exp(t[:n, 0]).sum()
        t[:n, 1] = np.log(np.exp(t[:n, 0]) / tot)
        m = np.zeros(t.shape[0], dtype=bool)
        m[:n] = True
        return torch.from_numpy(t.astype(np.float32)), torch.from_numpy(m)

    def __getitem__(self, i):
        if self.rep == "segs":
            x, mask = self.x[i], self.mask[i]
            if self.augment:
                x, mask = self._augment_segs(x, mask)
            return x.float(), mask, i
        if self.rep == "bins":
            return self.x[i].float(), self.mask[i], i
        return self.x[i].float(), self.ch[i], self.mask[i], i


# --------------------------------------------------------------------------- samplers

class PKSampler(torch.utils.data.Sampler):
    """Yields batches of P classes x K samples using a video->row-indices map."""

    def __init__(self, labels: list[str], p: int, k: int, seed: int = 0):
        self.by_label = defaultdict(list)
        for i, lab in enumerate(labels):
            self.by_label[lab].append(i)
        self.labels = sorted(self.by_label)
        self.p, self.k = p, k
        self.epoch = 0

    def set_epoch(self, e: int):
        self.epoch = e

    def __iter__(self):
        rng = random.Random(1234 + self.epoch)
        labels = list(self.labels)
        rng.shuffle(labels)
        batches = []
        for c0 in range(0, len(labels), self.p):
            chunk = labels[c0:c0 + self.p]
            batch = []
            for lab in chunk:
                idxs = self.by_label[lab]
                take = min(self.k, len(idxs))
                batch.extend(rng.sample(idxs, take) if take < len(idxs) else list(idxs))
            if len(batch) > 1:
                batches.append(batch)
        rng.shuffle(batches)
        return iter(batches)

    def __len__(self):
        return (len(self.labels) + self.p - 1) // self.p
