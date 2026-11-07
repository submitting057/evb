# -*- coding: utf-8 -*-
"""DF-style 1D CNN over multi-channel packet-index signed-size series.

Representation
  Each semantic channel becomes its own packet-index series (position = packet
  ordinal inside the channel, value = signed normalized size).  The packet-index
  domain is invariant to the segment-merging nuisance (merging only changes
  timing, not order/mass) and partially robust to wall-clock jitter.

  value='logsize':  sign(s) * log1p(|s|) / log1p(65536)   (default)
  value='dir'    :  sign(s)                                (DF-pure +-1)
  dt_feat=True   :  adds a 2nd feature  log1p(dt_ms)/log1p(1000) per event

Fusion modes (multi-channel modeling exploration)
  merge  : selected channels' events merged by time into ONE series
           (the "simple temporal concatenation" naive baseline; channel
           identity is lost in the interleaving)
  stack  : per-channel series stacked as input channels, shared convs
  branch : per-channel DF trunk with own weights -> concat -> head

Side input: 120-d segment-v2 fingerprint (round-1 finding: segment mass
sequence carries the identity) through a small MLP, fused with the CNN flatten.
"""
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset

from .config import PROJECT_ROOT, SEMANTIC_CHANNELS
from .data import SessionIndex, load_events

LOG_K = float(np.log1p(65536))
CHANNEL_LEN = {"video_cdn": 4096, "api": 512, "thumbnail": 256,
               "channel_info": 128, "other": 64}
MERGE_LEN = 4096
CROP_MS = 500.0          # start-jitter amplitude (data diagnosis: +-500ms)


# ----------------------------------------------------------------- raw cache

def build_series_cache(index: SessionIndex, out_path: Path) -> None:
    """One pass over all event npz -> concatenated raw cache
    (signed size int32, time_ms int32, channel uint8, offsets)."""
    vals, times, chans, offs = [], [], [], [0]
    for rec in index.sessions:
        ev, chan, _ = load_events(index, rec["session_id"])
        vals.append(ev[:, 0].astype(np.int32))
        times.append(ev[:, 3].astype(np.int32))       # abs time, ms
        chans.append(chan.astype(np.uint8))
        offs.append(offs[-1] + len(ev))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_path,
             vals=np.concatenate(vals), times=np.concatenate(times),
             chans=np.concatenate(chans), offs=np.asarray(offs, dtype=np.int64))
    print(f"[cache] {len(offs) - 1} sessions -> {out_path}")


def load_series_cache(index: SessionIndex, dataset: str) -> dict:
    path = PROJECT_ROOT / "runs" / "cache" / f"df_series_{dataset}.npz"
    if not path.exists():
        build_series_cache(index, path)
    z = np.load(path)
    out = {k: z[k] for k in ("vals", "times", "chans", "offs")}
    out["ids"] = [rec["session_id"] for rec in index.sessions]
    return out


def _encode(size: np.ndarray, value: str) -> np.ndarray:
    if value == "dir":
        return np.sign(size).astype(np.float16)
    return (np.sign(size) * np.log1p(np.abs(size)) / LOG_K).astype(np.float16)


def _dts(t: np.ndarray) -> np.ndarray:
    if t.size == 0:
        return np.zeros(0, dtype=np.float16)
    d = np.diff(t, prepend=t[0]).astype(np.float32)
    return (np.log1p(np.clip(d, 0, None)) / float(np.log1p(1000))).astype(np.float16)


# ----------------------------------------------------------------- dataset

class DFData(Dataset):
    """In-memory packet-index series for a session-id list.

    Flat modes return (x, y, i); branch mode returns ([(F, L_c) per channel], y, i).
    Series are left-aligned and right-padded with zeros ("no more packets").

    chan_emb (merge only): appends per-event channel one-hot rows, so the
    merged stream keeps BOTH temporal interleaving AND channel identity.
    """

    def __init__(self, cache: dict, session_ids: list[str], labels: dict,
                 channels: list[str], fusion: str, value: str = "logsize",
                 dt_feat: bool = False, chan_emb: bool = False,
                 augment: bool = False, seed: int = 0,
                 lens: dict | None = None):
        self.channels = [c for c in SEMANTIC_CHANNELS if c in channels]
        self.fusion, self.value, self.dt_feat = fusion, value, dt_feat
        self.chan_emb = chan_emb and fusion == "merge"
        self.augment = augment
        self.rng = random.Random(seed + 31)
        self.ids = list(session_ids)
        self.labels = np.array([labels[s] for s in self.ids], dtype=np.int64)
        pos = {s: i for i, s in enumerate(cache["ids"])}
        self.F = 2 if dt_feat else 1

        V, T, D = cache["vals"], cache["times"], cache["chans"]
        offs = cache["offs"]
        self.data = []                # per session: [(enc, aux|None, dt|None, t)]
        for sid in self.ids:
            a, b = offs[pos[sid]], offs[pos[sid] + 1]
            v, t, c = V[a:b], T[a:b], D[a:b]
            if fusion == "merge":
                sel_ids = [SEMANTIC_CHANNELS.index(ch) for ch in self.channels]
                sel = np.isin(c, sel_ids)
                v, t, c = v[sel], t[sel], c[sel]
                o = np.argsort(t, kind="stable")
                v, t, c = v[o][:MERGE_LEN], t[o][:MERGE_LEN], c[o][:MERGE_LEN]
                aux = np.eye(len(SEMANTIC_CHANNELS), dtype=np.float16)[c].T \
                    if self.chan_emb else None
                self.data.append([(_encode(v, self.value), aux,
                                   _dts(t) if dt_feat else None, t)])
            else:
                per = []
                for ch in self.channels:
                    m = c == SEMANTIC_CHANNELS.index(ch)
                    vch, tch = v[m], t[m]
                    per.append((_encode(vch, self.value), None,
                                _dts(tch) if dt_feat else None, tch))
                self.data.append(per)
        clen = dict(CHANNEL_LEN)
        clen.update(lens or {})
        if fusion == "branch":
            self.lens = [clen[ch] for ch in self.channels]
        else:
            self.lens = [MERGE_LEN if fusion == "merge"
                         else max(clen[ch] for ch in self.channels)]
        self.n_series = 1 if fusion == "merge" else len(self.channels)

    def __len__(self):
        return len(self.ids)

    def _augment(self, per):
        """Nuisance factors diagnosed in round 1, applied in time domain:
        start-point jitter (+-500ms), tail truncation (observation budget),
        auto-quality global rescale (video 2-10x -> log-size shift),
        channel dropout (event removal in the merged stream), value noise."""
        rng = self.rng
        t_last = max((t[-1] for (_, _, _, t) in per if t.size), default=0)
        t0 = rng.uniform(0, CROP_MS) if rng.random() < 0.5 else 0.0
        t1 = t_last * rng.uniform(0.65, 1.0) if (rng.random() < 0.25 and t_last) else np.inf
        out = []
        for j, (enc, aux, dt, t) in enumerate(per):
            if enc.size and (t0 > 0 or t1 < np.inf):
                keep = (t >= t0) & (t <= t1)
                enc, aux, dt, t = (enc[keep], aux[:, keep] if aux is not None else None,
                                   dt[keep] if dt is not None else None, t[keep])
            if enc.size and aux is not None:      # merged: drop a whole channel's events
                drop = np.zeros(len(enc), dtype=bool)
                for cid in range(1, len(SEMANTIC_CHANNELS)):   # never drop video
                    if aux[cid].sum() > 0 and rng.random() < 0.15:
                        drop |= aux[cid] > 0.5
                if drop.any():
                    keep2 = ~drop
                    enc = enc[keep2]
                    aux = aux[:, keep2]
                    dt = dt[keep2] if dt is not None else None
                    t = t[keep2]
            if enc.size and self.value != "dir":
                v = enc.astype(np.float32)
                # quality rescale: size *= a  <=>  v += ln(a)/LOG_K
                if aux is not None:
                    is_video = aux[0] > 0.5
                    pos = v > 0
                    v[pos & is_video] += rng.gauss(0, 0.4) / LOG_K
                    v[pos & ~is_video] += rng.gauss(0, 0.15) / LOG_K
                elif self.channels[j] == "video_cdn":
                    v[v > 0] += rng.gauss(0, 0.4) / LOG_K
                else:
                    v[v > 0] += rng.gauss(0, 0.15) / LOG_K
                v += rng.gauss(0, 0.02)
                enc = v.astype(np.float16)
            if aux is None and j > 0 and rng.random() < 0.15:   # per-channel dropout
                enc = enc[:0]
                dt = dt[:0] if dt is not None else dt
            out.append((enc, aux, dt))
        return out

    def __getitem__(self, i):
        per = self.data[i]
        items = self._augment(per) if self.augment else \
            [(enc, aux, dt) for enc, aux, dt, _ in per]
        if self.fusion == "branch":
            blocks = []
            for j, (enc, aux, dt) in enumerate(items):
                lj = self.lens[j]
                x = np.zeros((self.F, lj), dtype=np.float32)
                n = min(len(enc), lj)
                if n:
                    x[0, :n] = enc[:n]
                    if self.dt_feat:
                        x[1, :n] = dt[:n]
                blocks.append(x)
            return blocks, int(self.labels[i]), i
        L = self.lens[0]
        n_oh = len(SEMANTIC_CHANNELS) if self.chan_emb else 0
        x = np.zeros((self.n_series * self.F + n_oh, L), dtype=np.float32)
        for j, (enc, aux, dt) in enumerate(items):
            n = min(len(enc), L)
            if n:
                x[j * self.F, :n] = enc[:n]
                if self.dt_feat:
                    x[j * self.F + 1, :n] = dt[:n]
                if aux is not None:
                    x[self.n_series * self.F:, :n] = aux[:, :n]
        return torch.from_numpy(x), int(self.labels[i]), i


def df_collate(batch):
    """Handles both flat (x, y, i) and branch [(x_c...), y, i] samples."""
    if isinstance(batch[0][0], list):
        n_ch = len(batch[0][0])
        cols = [[] for _ in range(n_ch)]
        ys, idx = [], []
        for blocks, y, i in batch:
            for c in range(n_ch):
                cols[c].append(blocks[c])
            ys.append(y)
            idx.append(i)
        return tuple(torch.from_numpy(np.stack(col)) for col in cols), \
            torch.tensor(ys, dtype=torch.long), torch.tensor(idx)
    xs, ys, idx = zip(*batch)
    return torch.stack(xs), torch.tensor(ys, dtype=torch.long), torch.tensor(idx)


# ----------------------------------------------------------------- model

class DFBlock(nn.Module):
    """DF conv block: 2x(Conv1d k=8 'same' + BN) -> MaxPool(4,4,ceil) -> Drop.
    Block 1 uses ELU, later blocks ReLU (faithful to src/Model_NoDef.py)."""

    def __init__(self, cin: int, cout: int, act: str = "relu"):
        super().__init__()
        self.c1 = nn.Conv1d(cin, cout, 8, padding=4)   # pad 4: length l -> l+1
        self.b1 = nn.BatchNorm1d(cout)
        self.c2 = nn.Conv1d(cout, cout, 8, padding=4)
        self.b2 = nn.BatchNorm1d(cout)
        self.act = nn.ELU() if act == "elu" else nn.ReLU()
        self.pool = nn.MaxPool1d(4, 4, ceil_mode=True)
        self.drop = nn.Dropout(0.1)

    def forward(self, x):
        x = self.act(self.b1(self.c1(x)))
        x = self.act(self.b2(self.c2(x)))
        return self.drop(self.pool(x))


class DFTrunk(nn.Module):
    def __init__(self, cin: int, widths=(32, 64, 128, 256)):
        super().__init__()
        acts = ("elu", "relu", "relu", "relu")
        chans = (cin,) + tuple(widths)
        self.blocks = nn.ModuleList(
            DFBlock(chans[k], chans[k + 1], acts[k]) for k in range(len(widths)))

    def out_len(self, lin: int) -> int:
        l = lin
        for _ in range(4):
            l = -(-(l + 2) // 4)                # 2x conv k8 pad4 (+1 each), pool /4 ceil
        return max(l, 1)

    def forward(self, x):
        for b in self.blocks:
            x = b(x)
        return x.flatten(1)


class DFNet(nn.Module):
    """Modified DF CNN with three fusion modes + optional side MLP."""

    def __init__(self, channels: list[str], fusion: str, n_class: int,
                 F: int = 1, side_dim: int = 0, chan_emb: bool = False,
                 widths=(32, 64, 128, 256), lens: dict | None = None):
        super().__init__()
        self.fusion = fusion
        clen = dict(CHANNEL_LEN)
        clen.update(lens or {})
        if fusion in ("merge", "stack"):
            cin = (F + (len(SEMANTIC_CHANNELS) if chan_emb else 0)) \
                if fusion == "merge" else len(channels) * F
            lin = MERGE_LEN if fusion == "merge" else max(clen[c] for c in channels)
            self.trunk = DFTrunk(cin, widths)
            self._flat = 256 * self.trunk.out_len(lin)
        else:                                    # branch
            self.trunk = None
            self.trunk_lens = [clen[c] for c in channels]
            flat = 0
            for l in self.trunk_lens:
                t = DFTrunk(F, widths)
                flat += 256 * t.out_len(l)
            self._flat = flat
            self.trunks = nn.ModuleList(DFTrunk(F, widths) for _ in channels)
        self.side = nn.Sequential(
            nn.LayerNorm(side_dim), nn.Linear(side_dim, 256), nn.ReLU(),
            nn.Dropout(0.3), nn.Linear(256, 256)) if side_dim else None
        d_in = self._flat + (256 if self.side else 0)
        self.head = nn.Sequential(
            nn.Linear(d_in, 512), nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(0.7),
            nn.Linear(512, 512), nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(512, n_class))

    def forward(self, *xs, side=None):
        if self.fusion == "branch":
            f = torch.cat([t(x) for t, x in zip(self.trunks, xs)], 1)
        else:
            f = self.trunk(xs[0])
        if self.side is not None and side is not None:
            f = torch.cat([f, self.side(side)], 1)
        return self.head(f)
