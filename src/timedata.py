# -*- coding: utf-8 -*-
"""Time-domain (10ms bins) DF member -- representation axis.

All existing members live in the packet-index domain; the absolute timing of
segment bursts/idles (visible only on the wall-clock axis) is orthogonal
evidence.  This dataset feeds the video_cdn channel's 10ms bins (log down/up
bytes) to the proven DF trunk.  Nuisance augmentation mirrors DFData but in
bin space (start crop 500ms, tail truncation, quality rescale, value noise).
"""
import random

import numpy as np
from torch.utils.data import Dataset

from .config import MAX_BINS
from .data import SessionIndex, load_bins

LOG_B = float(np.log1p(4e6))        # ~ a full 10s burst in one bin (safety cap)


class BinsData(Dataset):
    """(F=2, MAX_BINS) time-bin series for the video_cdn channel.

    F rows: [log1p(down_bytes), log1p(up_bytes)] / LOG_B, left-aligned,
    right-padded with zeros.
    """

    def __init__(self, index: SessionIndex, session_ids: list[str],
                 labels: dict, augment: bool = False, seed: int = 0,
                 bin_size_ms: int = 10, max_bins: int | None = None):
        self.ids = list(session_ids)
        self.labels = np.array([labels[s] for s in self.ids], dtype=np.int64)
        self.augment = augment
        self.rng = random.Random(seed + 31)
        self.max_bins = max_bins or MAX_BINS.get("10s_nogro", 1500)
        # video_cdn is DISK channel 0; keep [down_B, up_B] quantities
        self.data = []
        for sid in self.ids:
            b = load_bins(index, sid)[0]           # (T, 4)
            x = np.stack([np.log1p(np.clip(b[:, 0], 0, None)) / LOG_B,
                          np.log1p(np.clip(b[:, 1], 0, None)) / LOG_B])
            x = x.astype(np.float16)
            self.data.append((x, b.shape[0]))      # T for time-domain augment

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        x, T = self.data[i]
        x = x.astype(np.float32)
        if self.augment:
            rng = self.rng
            t0 = rng.uniform(0, 500.0) if rng.random() < 0.5 else 0.0   # ms
            b0 = int(t0 / 10)
            t1 = T * rng.uniform(0.65, 1.0) if (rng.random() < 0.25 and T) else 1e9
            b1 = min(int(t1 / 10), x.shape[1])
            if b0 > 0 or b1 < x.shape[1]:
                x = np.zeros_like(x)
                if b1 > b0:
                    seg = self.data[i][0][:, b0:b1].astype(np.float32)
                    x[:, : seg.shape[1]] = seg
            # quality rescale: bytes *= a  <=>  log += ln(a)/LOG_B (down row only)
            x[0] = x[0] + rng.gauss(0, 0.4) / LOG_B
            x[1] = x[1] + rng.gauss(0, 0.15) / LOG_B
            x = x + rng.gauss(0, 0.02)
        out = np.zeros((2, self.max_bins), dtype=np.float32)
        n = min(x.shape[1], self.max_bins)
        out[:, :n] = x[:, :n]
        return out, int(self.labels[i]), i


def bins_collate(batch):
    xs, ys, idx = zip(*batch)
    return np.stack(xs), np.array(ys), np.array(idx)
