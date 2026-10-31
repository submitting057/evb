# -*- coding: utf-8 -*-
"""Per-channel event-count stats to fix sequence lengths for the DF pipeline."""
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import PROJECT_ROOT, SEMANTIC_CHANNELS, DATASET_DIRS
from src.data import SessionIndex, load_splits, load_events


def main(dataset="10s"):
    index = SessionIndex(DATASET_DIRS[dataset], 10)
    splits = load_splits(PROJECT_ROOT / "splits" / f"splits_{dataset}_s42.json")
    rng = random.Random(0)
    sids = sorted(splits["sessions"])
    sample = rng.sample(sids, 800)
    counts = defaultdict(list)
    sizes = defaultdict(list)
    tot = []
    for sid in sample:
        ev, chan, _ = load_events(index, sid)
        tot.append(len(ev))
        for c in range(len(SEMANTIC_CHANNELS)):
            m = chan == c
            counts[c].append(int(m.sum()))
            if m.any():
                sizes[c].append(float(np.abs(ev[m, 0]).mean()))
    q = lambda a: np.percentile(a, [50, 95, 99, 100]).round(0).astype(int) if a else [0] * 4
    print(f"dataset={dataset}  sampled={len(sample)} sessions")
    print(f"total events: median={int(np.median(tot))} p95={q(tot)[1]} p99={q(tot)[2]} max={q(tot)[3]}")
    for c, name in enumerate(SEMANTIC_CHANNELS):
        a = counts[c]
        present = np.mean(np.array(a) > 0)
        print(f"  {name:12s} n: median={int(np.median(a)):5d} p95={q(a)[1]:5d} p99={q(a)[2]:5d} "
              f"max={q(a)[3]:5d}  present={present:.2f}  mean|size|={np.mean(sizes[c]) if sizes[c] else 0:.0f}B")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "10s")
