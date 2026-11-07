# -*- coding: utf-8 -*-
"""Generate the video-level known/unknown + per-video session splits.

--protocol legacy : original split (80/20 known/unknown; unknown sessions
                    halved into val/test impostor pools -- identities overlap).
--protocol strict : 6:2:2 video-disjoint (386 known / 129 unknown->val impostors /
                    129 unknown->test impostors; 10s only per project decision).
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import RunConfig
from src.data import (SessionIndex, load_splits, make_splits, make_splits_strict,
                      split_summary)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--protocol", default="legacy", choices=["legacy", "strict"])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--dataset", default=None,
                    help="restrict to one dataset key (e.g. 10s_nogro); "
                         "default keeps the historical per-protocol list")
    args = ap.parse_args()
    if args.dataset:
        datasets = [args.dataset]
    else:
        datasets = ["10s", "5s"] if args.protocol == "legacy" else ["10s"]
    for dataset in datasets:
        cfg = RunConfig(dataset=dataset, protocol=args.protocol)
        cfg.split.seed = args.seed
        if cfg.splits_path.exists():
            print(f"[skip] {cfg.splits_path} exists")
            splits = load_splits(cfg.splits_path)
        else:
            index = SessionIndex(cfg.dataset_dir, cfg.bin_size_ms)
            gen = make_splits_strict if args.protocol == "strict" else make_splits
            splits = gen(index, cfg.split, cfg.splits_path)
            print(f"[wrote] {cfg.splits_path}")
        print(f"== {dataset} protocol={args.protocol} seed={args.seed} ==")
        print(split_summary(splits))
        print()


if __name__ == "__main__":
    main()
