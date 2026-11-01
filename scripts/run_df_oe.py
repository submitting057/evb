# -*- coding: utf-8 -*-
"""T1b: true-unknown outlier exposure (OE) fine-tune for a DF member.

The T1a audit closed the synthetic line (F1 borderline-below gate, F2/F3 pass
trivially with probe-overlapping bulk), so the OE material is the REAL val
unknown sessions -- the only family guaranteed to cover the "coordinated
collinear" magnets.  Discipline:

  * 5-fold leave-unknown-video-out (fold split identical to run_combiner
    --lovo: sorted unknown val videos, default_rng(0) permutation, strided);
    the model of fold f NEVER sees fold f's videos in its OE set, so pooled
    LOVO judging stays honest.
  * per (lam, fold): fine-tune CE(train) + lam * CE(uniform, OE-sessions)
    from the frozen checkpoint; score val probes + fold-f impostors.
  * one "full" model per lam (OE on all val unknowns) -> the deployment
    candidate; it is judged ONLY on test (val is in-sample for it), and its
    deployment threshold must come from pooled LOVO scores, never from val.

Gates (pilot, pre-registered): val-probe top1 >= .963; fold-out magnet
videos' scores drop vs the frozen member; pooled-LOVO ensemble (LGBM + other
DF + OE-DF, fixed geo weights) >= frozen baseline .696.
"""
import argparse
import copy
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import PROJECT_ROOT, RunConfig
from src.data import SessionIndex, load_splits, session_pool
from src.dfnet import DFData, DFNet, df_collate, load_series_cache
from src.evaluate import open_set_metrics

FOLDS = 5


def unpack(batch):
    *raw, y, idx = batch
    xs = list(raw[0]) if len(raw) == 1 and isinstance(raw[0], (tuple, list)) else raw
    return xs, y, idx


@torch.no_grad()
def predict(model, ds, device, batch=256):
    model.eval()
    ld = DataLoader(ds, batch_size=batch, shuffle=False, collate_fn=df_collate)
    out = []
    for b in ld:
        xs, _, _ = unpack(b)
        out.append(torch.softmax(model(*[x.to(device) for x in xs]), 1)
                   .float().cpu().numpy())
    return np.concatenate(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="10s")
    ap.add_argument("--tag", default="dfs_branch_all_dt",
                    help="frozen DF member to fine-tune")
    ap.add_argument("--lams", default="0.05,0.2")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--folds-to-run", default="all",
                    help="'all', or comma fold indices for a pilot (e.g. '0')")
    ap.add_argument("--skip-full", action="store_true")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    lams = [float(x) for x in args.lams.split(",")]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out_root = PROJECT_ROOT / "runs" / (args.out or f"{args.tag}_oe")

    rcfg = RunConfig(dataset=args.dataset, protocol="strict")
    index = SessionIndex(rcfg.dataset_dir, rcfg.bin_size_ms)
    splits = load_splits(rcfg.splits_path)
    known = sorted(splits["known_videos"])
    vidx = {v: i for i, v in enumerate(known)}
    labels = {s: vidx[r["video"]] if r["role"] == "known" else -1
              for s, r in splits["sessions"].items()}
    pools = dict(
        train=session_pool(splits, "known", "train"),
        val_probe=session_pool(splits, "known", "val", gallery=False),
        val_imp=session_pool(splits, "unknown", "val"),
        test_probe=session_pool(splits, "known", "test", gallery=False),
        test_imp=session_pool(splits, "unknown", "test"),
    )
    vid_row = np.array([splits["sessions"][s]["video"] for s in pools["val_imp"]])
    uvid = np.array(sorted(set(vid_row.tolist())))
    perm = np.random.default_rng(0).permutation(len(uvid))     # == run_combiner
    fold_sets = [set(uvid[perm[f::FOLDS]].tolist()) for f in range(FOLDS)]
    v2f = {v: f for f, S in enumerate(fold_sets) for v in S}
    row_fold = np.array([v2f[v] for v in vid_row])

    saved = torch.load(PROJECT_ROOT / "runs" / args.tag / "model.pt",
                       map_location=device, weights_only=False)
    margs = saved["args"]
    cache = load_series_cache(index, args.dataset)

    def mk(sids, augment):
        return DFData(cache, sids, labels, saved["channels"], margs["fusion"],
                      margs["value"], margs["dt"], chan_emb=margs["chan_emb"],
                      augment=augment, seed=42)

    def fresh_model():
        m = DFNet(saved["channels"], margs["fusion"], n_class=len(known),
                  F=2 if margs["dt"] else 1, side_dim=0,
                  chan_emb=margs["chan_emb"]).to(device)
        m.load_state_dict(saved["state"])
        return m

    base_probe = predict(fresh_model(), mk(pools["val_probe"], False), device)
    base_imp = predict(fresh_model(), mk(pools["val_imp"], False), device)
    y_val = mk(pools["val_probe"], False).labels
    print(f"[frozen] val probe top1={float((base_probe.argmax(1) == y_val).mean()):.4f}"
          f"  imp maxprob q99={np.quantile(base_imp.max(1), .99):.3f}")

    summary = {"tag": args.tag, "lams": lams, "epochs": args.epochs,
               "frozen_top1": float((base_probe.argmax(1) == y_val).mean())}

    folds = (list(range(FOLDS)) if args.folds_to_run == "all"
             else [int(x) for x in args.folds_to_run.split(",")])
    for lam in lams:
        lam_dir = out_root / f"lam{lam}"
        lam_dir.mkdir(parents=True, exist_ok=True)
        jobs = folds + ([None] if not args.skip_full else [])
        for g in jobs:
            full = g is None
            keep = np.ones(len(row_fold), bool) if full else (row_fold != g)
            oe_sids = [s for s, k in zip(pools["val_imp"], keep) if k]
            fold_sids = pools["test_imp"] if full else \
                [s for s, k in zip(pools["val_imp"], ~keep) if k]

            model = fresh_model()
            opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
            scaler = torch.amp.GradScaler(enabled=device == "cuda")
            crit = torch.nn.CrossEntropyLoss(label_smoothing=0.05)
            tr = DataLoader(mk(pools["train"], True), batch_size=64, shuffle=True,
                            collate_fn=df_collate)
            oe = DataLoader(mk(oe_sids, False), batch_size=64, shuffle=True,
                            collate_fn=df_collate, drop_last=False)
            t0 = time.time()
            for ep in range(args.epochs):
                model.train()
                it = iter(oe)
                for b in tr:
                    xs, yb, _ = unpack(b)
                    xs = [x.to(device, non_blocking=True) for x in xs]
                    yb = yb.to(device)
                    try:
                        ox, _, _ = unpack(next(it))
                    except StopIteration:
                        it = iter(oe)
                        ox, _, _ = unpack(next(it))
                    ox = [x.to(device, non_blocking=True) for x in ox]
                    with torch.autocast(device_type="cuda", dtype=torch.float16,
                                        enabled=device == "cuda"):
                        ce = crit(model(*xs), yb)
                        po = torch.softmax(model(*ox), 1)
                        oel = -torch.log(po.clamp_min(1e-8)).mean()
                        loss = ce + lam * oel
                    opt.zero_grad()
                    scaler.scale(loss).backward()
                    scaler.step(opt)
                    scaler.update()
                if ep == args.epochs - 1 or ep % 2 == 1:
                    pv = predict(model, mk(pools["val_probe"], False), device)
                    print(f"lam={lam} {'full' if full else f'fold{g}'} "
                          f"ep{ep + 1} top1={float((pv.argmax(1) == y_val).mean()):.4f} "
                          f"({time.time() - t0:.0f}s)", flush=True)

            pr = predict(model, mk(pools["val_probe"], False), device)
            pi = predict(model, mk(fold_sids, False), device)
            if not full:
                np.savez(lam_dir / f"fold{g}.npz", probe=pr, probe_y=y_val, imp=pi,
                         imp_fold=np.array([v2f[v] for v in
                                            [splits["sessions"][s]["video"]
                                             for s in fold_sids]]))
                b_i = base_imp[~keep]
                top = np.argsort(b_i.max(1))[::-1][:max(1, int(0.005 * len(b_i)))]
                print(f"  fold{g}: held-out imp q99 maxprob "
                      f"{np.quantile(pi.max(1), .99):.3f} (frozen same rows "
                      f"{np.quantile(b_i.max(1), .99):.3f}); "
                      f"top0.5% mean {pi.max(1)[top].mean():.3f} vs "
                      f"{b_i.max(1)[top].mean():.3f}")
            else:
                tp = predict(model, mk(pools["test_probe"], False), device)
                ti = predict(model, mk(pools["test_imp"], False), device)
                np.savez(lam_dir / "full.npz", probe=pr, probe_y=y_val,
                         val_imp=pi, test_probe=tp,
                         test_probe_y=mk(pools["test_probe"], False).labels,
                         test_imp=ti)
                m = open_set_metrics(pr, y_val, pi.max(1), n_boot=0)
                print(f"  full(in-sample val): top1={m['closed']['top1']:.4f} "
                      f"@.5%={m['fpr'][0.005]['tpr']:.3f} "
                      f"@1%={m['fpr'][0.01]['tpr']:.3f}")

    (out_root / "summary.json").write_text(json.dumps(summary, indent=1),
                                           encoding="utf-8")
    print(f"[wrote] {out_root}")


if __name__ == "__main__":
    main()
