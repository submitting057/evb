# -*- coding: utf-8 -*-
"""CLI: python scripts/run_train.py <10s|5s> <bins|events> [seed]"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import RunConfig
from src.train import train


def main():
    dataset = sys.argv[1] if len(sys.argv) > 1 else "10s"
    rep = sys.argv[2] if len(sys.argv) > 2 else "bins"
    seed = int(sys.argv[3]) if len(sys.argv) > 3 else 42
    rcfg = RunConfig(dataset=dataset, rep=rep)
    rcfg.split.seed = seed
    train(rcfg)


if __name__ == "__main__":
    main()
