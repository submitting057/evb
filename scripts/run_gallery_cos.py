# -*- coding: utf-8 -*-
"""Gallery-cosine member dump (A3): DF penultimate embeddings vs registered
gallery templates -- a non-softmax geometry member for the open-set combiner.

Motivation: every member so far scores a session against the TRAINED class
weights (softmax).  The protocol also registers 3 gallery sessions per known
video which softmax never uses.  Here the 512-d pre-logit embedding
(head[:-1], eval mode) of every probe/impostor/gallery session is compared by
cosine to per-class gallery templates (renormalized mean of L2-normalized
gallery embeddings, built PER SPLIT from that split's gallery pool) ->
runs/{tag}_cos/probs_{split}.npz (probe, probe_y, imp), directly loadable by
run_combiner.py --cos-tags.  Real session exemplars vs learned hyperplanes:
a deliberately different error geometry for the soft-AND ensemble.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from run_tta_dump import unpack
from src.config import PROJECT_ROOT, RunConfig
from src.data import SessionIndex, load_splits, session_pool
from src.dfnet import DFData, DFNet, df_collate, load_series_cache
from src.evaluate import open_set_metrics


@torch.no_grad()
def embed(model, ds, device, batch=256):
    model.eval()
    ld = DataLoader(ds, batch_size=batch, shuffle=False, collate_fn=df_collate)
    out = []
    for b in ld:
        xs, _, _ = unpack(b)
        xs = [x.to(device) for x in xs]
        if model.fusion == "branch":
            f = torch.cat([t(x) for t, x in zip(model.trunks, xs)], 1)
        else:
            f = model.trunk(xs[0])
        out.append(model.head[:-1](f).float().cpu().numpy())
    return np.concatenate(out)


def _norm(E):
    return E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-8)


def templates(E, y, n_class):
    En = _norm(E)
    T = np.zeros((n_class, En.shape[1]), dtype=np.float32)
    for c in range(n_class):
        m = y == c
        if m.any():
            T[c] = En[m].mean(0)
    return _norm(T).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="10s")
    ap.add_argument("--tags", required=True, help="comma list of DF run tags")
    args = ap.parse_args()
    tags = [t for t in args.tags.split(",") if t]
    device = "cuda" if torch.cuda.is_available() else "cpu"

    rcfg = RunConfig(dataset=args.dataset)
    index = SessionIndex(rcfg.dataset_dir, rcfg.bin_size_ms)
    splits = load_splits(rcfg.splits_path)
    known = sorted(splits["known_videos"])
    vidx = {v: i for i, v in enumerate(known)}
    labels = {s: vidx[r["video"]] if r["role"] == "known" else -1
              for s, r in splits["sessions"].items()}
    pools = {}
    for split in ("val", "test"):
        pools[f"{split}_probe"] = session_pool(splits, "known", split, gallery=False)
        pools[f"{split}_imp"] = session_pool(splits, "unknown", split)
        pools[f"{split}_gal"] = session_pool(splits, "known", split, gallery=True)
    cache = load_series_cache(index, args.dataset)

    for tag in tags:
        saved = torch.load(PROJECT_ROOT / "runs" / tag / "model.pt",
                           map_location=device, weights_only=False)
        margs = saved["args"]
        channels = saved.get("channels") or \
            [c for c in margs["channels"].split(",") if c]
        model = DFNet(channels, margs["fusion"], n_class=len(known),
                      F=2 if margs["dt"] else 1,
                      side_dim=120 if margs["side"] == "seg" else 0,
                      chan_emb=margs["chan_emb"]).to(device)
        model.load_state_dict(saved["state"])
        print(f"\n=== {tag} -> {tag}_cos ===")

        out_dir = PROJECT_ROOT / "runs" / f"{tag}_cos"
        out_dir.mkdir(parents=True, exist_ok=True)
        report = {"tag": tag, "splits": {}}
        for split in ("val", "test"):
            E, ys = {}, {}
            for name in ("probe", "imp", "gal"):
                ds = DFData(cache, pools[f"{split}_{name}"], labels, channels,
                            margs["fusion"], margs["value"], margs["dt"],
                            chan_emb=margs["chan_emb"], augment=False)
                E[name] = embed(model, ds, device)
                ys[name] = ds.labels
            T = templates(E["gal"], ys["gal"], len(known))
            Cp = _norm(E["probe"]) @ T.T
            Ci = _norm(E["imp"]) @ T.T
            m = open_set_metrics(Cp.astype(np.float64), ys["probe"],
                                 Ci.max(1), n_boot=200)
            print(f"{split:4s} cos-member top1={m['closed']['top1']:.3f} "
                  f"@.5%={m['fpr'][0.005]['tpr']:.3f} @1%={m['fpr'][0.01]['tpr']:.3f} "
                  f"@5%={m['fpr'][0.05]['tpr']:.3f} AUROC={m['auroc_detection']:.3f} "
                  f"(gallery n={len(ys['gal'])})")
            report["splits"][split] = m
            np.savez(out_dir / f"probs_{split}.npz",
                     probe=Cp, probe_y=ys["probe"], imp=Ci)
        (out_dir / "report.json").write_text(
            json.dumps(report, indent=1, default=str), encoding="utf-8")
        print(f"[wrote] {out_dir}")


if __name__ == "__main__":
    main()
