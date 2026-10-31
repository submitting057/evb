# -*- coding: utf-8 -*-
"""Final evaluation of a trained embedding model: closed-set + open-set on val & test,
both 'template' (mean of gallery embeddings) and 'max' (max over gallery sessions) scoring."""
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import RunConfig
from src.data import (PackedDataset, SessionIndex, load_splits, session_pool,
                      split_summary)
from src.evaluate import format_open_set_report, open_set_metrics
from src.models import build_model
from src.train import Embedder


def eval_split(emb, splits, known_videos, vidx, split: str) -> dict:
    gal = session_pool(splits, "known", split, gallery=True)
    probes = session_pool(splits, "known", split, gallery=False)
    imps = session_pool(splits, "unknown", split)
    video_of = splits["sessions"]
    y = np.array([vidx[video_of[s]["video"]] for s in probes], dtype=np.int64)

    E = emb(gal + probes + imps)
    n_g, n_p = len(gal), len(probes)
    E_g, E_p, E_i = E[:n_g], E[n_g:n_g + n_p], E[n_g + n_p:]
    gal_rows_by_video = {}
    for j, s in enumerate(gal):
        gal_rows_by_video.setdefault(video_of[s]["video"], []).append(j)
    vids = [v for v in known_videos if v in gal_rows_by_video]
    col = {v: i for i, v in enumerate(vids)}

    results = {}
    # mode 1: template = normalized mean of gallery embeddings
    T = np.stack([E_g[gal_rows_by_video[v]].mean(0) for v in vids])
    T = T / np.linalg.norm(T, axis=1, keepdims=True)
    full = np.full((n_p, len(known_videos)), -1.0, dtype=np.float32)
    full[:, [vidx[v] for v in vids]] = E_p @ T.T
    imp_scores = (E_i @ T.T).max(axis=1)
    results["template"] = open_set_metrics(full, y, imp_scores)

    # mode 2: max over individual gallery sessions
    sim_pg = E_p @ E_g.T                                   # (P, G)
    sim_ig = E_i @ E_g.T                                   # (I, G)
    full2 = np.full((n_p, len(known_videos)), -1.0, dtype=np.float32)
    imp2 = np.full(len(E_i), -1.0, dtype=np.float32)
    for v in vids:
        rows = gal_rows_by_video[v]
        full2[:, vidx[v]] = sim_pg[:, rows].max(axis=1)
        imp2 = np.maximum(imp2, sim_ig[:, rows].max(axis=1))
    results["max"] = open_set_metrics(full2, y, imp2)
    return results


def main(dataset="10s", rep="bins", seed=42):
    rcfg = RunConfig(dataset=dataset, rep=rep)
    rcfg.split.seed = seed
    ckpt_path = rcfg.run_dir / "best.pt"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    index = SessionIndex(rcfg.dataset_dir, rcfg.bin_size_ms)
    splits = load_splits(rcfg.splits_path)
    print(split_summary(splits))

    model = build_model(rcfg, dropout=rcfg.train.dropout).to(device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model"])
    print(f"loaded {ckpt_path} (epoch {ckpt['epoch']})")

    pools = dict(
        val_gal=session_pool(splits, "known", "val", gallery=True),
        val_probe=session_pool(splits, "known", "val", gallery=False),
        val_imp=session_pool(splits, "unknown", "val"),
        test_gal=session_pool(splits, "known", "test", gallery=True),
        test_probe=session_pool(splits, "known", "test", gallery=False),
        test_imp=session_pool(splits, "unknown", "test"),
    )
    all_sids = sorted({s for v in pools.values() for s in v})
    t0 = time.time()
    ds = PackedDataset(index, all_sids, rcfg.rep, rcfg.dataset, seed=0)
    print(f"packed {len(all_sids)} sessions in {time.time() - t0:.0f}s")
    emb = Embedder(model, ds, all_sids, device)

    known_videos = sorted(splits["known_videos"])
    vidx = {v: i for i, v in enumerate(known_videos)}
    report = {"checkpoint": str(ckpt_path), "best_epoch": ckpt["epoch"], "splits": {}}
    for split in ("val", "test"):
        res = eval_split(emb, splits, known_videos, vidx, split)
        report["splits"][split] = res
        for mode, m in res.items():
            print(format_open_set_report(f"{split.upper()} [{mode}]", m))
    # str() keys for JSON
    report["splits"] = {s: {mode: m for mode, m in res.items()}
                        for s, res in report["splits"].items()}
    out = rcfg.run_dir / "report.json"
    out.write_text(json.dumps(report, indent=1, default=str), encoding="utf-8")
    print(f"[wrote] {out}")


if __name__ == "__main__":
    ds = sys.argv[1] if len(sys.argv) > 1 else "10s"
    rep = sys.argv[2] if len(sys.argv) > 2 else "bins"
    main(ds, rep)
