# -*- coding: utf-8 -*-
"""Generate the video-level known/unknown + per-video session splits."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import RunConfig
from src.data import SessionIndex, load_splits, make_splits, split_summary


def main():
    for dataset in ["10s", "5s"]:
        for seed in [42]:
            cfg = RunConfig(dataset=dataset)
            cfg.split.seed = seed
            if cfg.splits_path.exists():
                print(f"[skip] {cfg.splits_path} exists")
                splits = load_splits(cfg.splits_path)
            else:
                index = SessionIndex(cfg.dataset_dir, cfg.bin_size_ms)
                splits = make_splits(index, cfg.split, cfg.splits_path)
                print(f"[wrote] {cfg.splits_path}")
            print(f"== {dataset} seed={seed} ==")
            print(split_summary(splits))
            print()


if __name__ == "__main__":
    main()
