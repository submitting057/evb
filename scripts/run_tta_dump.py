# -*- coding: utf-8 -*-
"""TTA view dump (A1) + raw logit dump (A2) for trained DF members -- no training.

A1: score each probe/impostor session under J deterministic time-domain views
    (head crop x tail truncation, the diagnosed nuisance factors the members
    were trained to be invariant to) and average the per-view probabilities.
    A true probe stays stable across views; an impostor's lucky spike is
    view-fragile -> within-member averaging damps the upper tail.
    Saved as runs/{tag}_tta/probs_{split}.npz so run_combiner.py can use TTA
    members directly (as replacements or as extra ensemble members).

A2: dump raw logits (runs/{tag}/logits_{split}.npz) so post-hoc score
    families (max-logit / energy) can be evaluated in run_combiner.py.

Views: t0 in {0,250,500} ms x tail in {1.0,0.85}; (0,1.0) is the identity view
and is checked against the member's saved probs for exactness.
LGBM has no TTA here (its segv2 features would need recomputation per view).
"""
import argparse
import copy
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import PROJECT_ROOT, RunConfig
from src.data import SessionIndex, load_splits, session_pool
from src.dfnet import DFData, DFNet, df_collate, load_series_cache
from src.evaluate import open_set_metrics

VIEWS = [(0.0, 1.0), (0.0, 0.85), (250.0, 1.0), (250.0, 0.85),
         (500.0, 1.0), (500.0, 0.85)]


def unpack(batch):
    *raw, y, idx = batch
    xs = list(raw[0]) if len(raw) == 1 and isinstance(raw[0], (tuple, list)) else raw
    return xs, y, idx


def view_dataset(ds: DFData, t0: float, tail: float) -> DFData:
    """Deterministic cross-channel time-domain view of a DFData (no RNG)."""
    d2 = copy.copy(ds)                      # shallow; .data rebuilt below
    new = []
    for per in ds.data:
        t_last = max((t[-1] for (_, _, _, t) in per if t.size), default=0)
        t1 = t_last * tail if (tail < 1.0 and t_last) else np.inf
        row = []
        for enc, aux, dt, t in per:
            if enc.size and (t0 > 0 or t1 < np.inf):
                keep = (t >= t0) & (t <= t1)
                enc = enc[keep]
                aux = aux[:, keep] if aux is not None else None
                dt = dt[keep] if dt is not None else None
                t = t[keep]
            row.append((enc, aux, dt, t))
        new.append(row)
    d2.data = new
    return d2


@torch.no_grad()
def predict(model, ds, device, batch=256):
    model.eval()
    ld = DataLoader(ds, batch_size=batch, shuffle=False, collate_fn=df_collate)
    probs, logits = [], []
    for b in ld:
        xs, _, _ = unpack(b)
        xs = [x.to(device) for x in xs]
        lg = model(*xs)
        logits.append(lg.float().cpu().numpy())
        probs.append(torch.softmax(lg, 1).float().cpu().numpy())
    return np.concatenate(probs), np.concatenate(logits)


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
    pools = dict(
        val_probe=session_pool(splits, "known", "val", gallery=False),
        val_imp=session_pool(splits, "unknown", "val"),
        test_probe=session_pool(splits, "known", "test", gallery=False),
        test_imp=session_pool(splits, "unknown", "test"),
    )
    cache = load_series_cache(index, args.dataset)

    for tag in tags:
        tag_dir = PROJECT_ROOT / "runs" / tag
        saved = torch.load(tag_dir / "model.pt", map_location=device,
                           weights_only=False)
        margs = saved["args"]
        # run_df saves the channel LIST at top level; args.channels is the raw string
        channels = saved.get("channels") or \
            [c for c in margs["channels"].split(",") if c]
        model = DFNet(channels, margs["fusion"], n_class=len(known),
                      F=2 if margs["dt"] else 1,
                      side_dim=120 if margs["side"] == "seg" else 0,
                      chan_emb=margs["chan_emb"]).to(device)
        model.load_state_dict(saved["state"])
        print(f"\n=== {tag} (fusion={margs['fusion']}, dt={margs['dt']}) ===")

        tta_dir = PROJECT_ROOT / "runs" / f"{tag}_tta"
        tta_dir.mkdir(parents=True, exist_ok=True)
        report = {"tag": tag, "views": [list(v) for v in VIEWS], "splits": {}}

        for split in ("val", "test"):
            base = {k: DFData(cache, pools[f"{split}_{k}"], labels, channels,
                              margs["fusion"], margs["value"], margs["dt"],
                              chan_emb=margs["chan_emb"], augment=False)
                    for k in ("probe", "imp")}
            Pv, Lv = {k: [] for k in base}, {}
            for (t0, tail) in VIEWS:
                for k, ds in base.items():
                    pv, lv = predict(model, view_dataset(ds, t0, tail), device)
                    Pv[k].append(pv)
                    if (t0, tail) == (0.0, 1.0):
                        Lv[k] = lv
            # identity-view exactness check against the member's saved probs
            ref = np.load(tag_dir / f"probs_{split}.npz")
            for k in base:
                dmax = float(np.abs(Pv[k][0] - ref[k]).max())
                assert dmax < 1e-3, f"identity view mismatch {tag} {split} {k}: {dmax}"
            y = base["probe"].labels
            assert (ref["probe_y"] == y).all(), "pool order mismatch"

            P = {k: np.mean(Pv[k], axis=0) for k in base}
            for name, PP in (("orig", ref["probe"]), ("tta", P["probe"])):
                m = open_set_metrics(PP, y,
                                     (ref["imp"] if name == "orig" else P["imp"]).max(1),
                                     n_boot=0)
                print(f"{split:4s} {name:4s} top1={m['closed']['top1']:.3f} "
                      f"@.5%={m['fpr'][0.005]['tpr']:.3f} "
                      f"@1%={m['fpr'][0.01]['tpr']:.3f} "
                      f"@5%={m['fpr'][0.05]['tpr']:.3f} "
                      f"AUROC={m['auroc_detection']:.3f}")
                report["splits"].setdefault(split, {})[name] = m
            np.savez(tta_dir / f"probs_{split}.npz",
                     probe=P["probe"], probe_y=y, imp=P["imp"])
            np.savez(tag_dir / f"logits_{split}.npz",
                     probe=Lv["probe"], probe_y=y, imp=Lv["imp"])

        (tta_dir / "report.json").write_text(json.dumps(report, indent=1, default=str),
                                             encoding="utf-8")
        print(f"[wrote] {tta_dir / 'report.json'}, {tag_dir}/logits_*.npz")


if __name__ == "__main__":
    main()
