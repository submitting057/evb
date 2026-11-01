# -*- coding: utf-8 -*-
"""T2-A2: DF member trained with an ArcFace head (metric-learning member).

Plain-language summary:

  All current DF members are trained with plain softmax cross-entropy, so
  they can share the same blind spots (M2: the two DFs agree on 52% of
  impostor sessions).  ArcFace trains a NORMALIZED cosine classifier with
  an angular margin instead: every class gets a unit-length proxy vector,
  every session a unit-length embedding, and the model is pushed to keep a
  margin between the true class angle and all others.  Metric heads change
  the error structure (they fight class "hubness": proxies all compete on
  equal footing), so the hope is DIFFERENT mistakes on impostors -- the
  same logic that made the aux member work.

  Discipline (pre-registered):
  * trained FROM SCRATCH with the same recipe as dfs_branch_all_e100
    (same channels/arch/aug/epochs); the ONLY change is the loss head.
    Fine-tuning from e100 would inherit its errors and fail the
    independence gate by construction.
  * training head: s=32, m=0.5 (standard ArcFace).
  * export scale s_eval is chosen so the VAL PROBE max-prob MEDIAN matches
    e100's -- probes are known-video sessions, no unknown-video info is
    used (this is a scale alignment for entering the geometric product,
    not the post-hoc temperature recalibration that is in the graveyard).
  * admission (checked separately after this script): argmax agreement
    with existing DF members on VAL impostors <= 10%, joint-overconfidence
    rate drop >= 30%, then pooled-LOVO combiner + paired cluster CI.
"""
import argparse
import copy
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as Fn
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import PROJECT_ROOT, RunConfig
from src.data import SessionIndex, load_splits, session_pool, split_summary
from src.dfnet import DFData, DFNet, df_collate, load_series_cache
from src.evaluate import format_open_set_report, open_set_metrics


def unpack(batch):
    *raw, y, idx = batch
    xs = list(raw[0]) if len(raw) == 1 and isinstance(raw[0], (tuple, list)) else raw
    return xs, y, idx


class ArcFaceHead(nn.Module):
    def __init__(self, dim, n_class, s=32.0, m=0.5):
        super().__init__()
        self.W = nn.Parameter(torch.empty(n_class, dim))
        nn.init.xavier_uniform_(self.W)
        self.s, self.m = s, m
        self.cos_m = float(np.cos(m))
        self.sin_m = float(np.sin(m))

    def forward(self, feat, y=None):
        # cosine head in strict fp32 (autocast off): the margin sqrt is
        # numerically sensitive and fp16 matmul precision is not needed here
        with torch.autocast(device_type="cuda", enabled=False):
            f = Fn.normalize(feat.float(), dim=1)
            w = Fn.normalize(self.W.float(), dim=1)
            cos = f @ w.T                               # (B, C)
            if y is None or not self.training:
                return cos
            sin = torch.sqrt(torch.clamp(1.0 - cos * cos, min=1e-9))
            cos_m = cos * self.cos_m - sin * self.sin_m
            onehot = Fn.one_hot(y, cos.shape[1]).bool()
            out = torch.where(onehot, cos_m, cos)
            return self.s * out


class ArcDF(nn.Module):
    """DFNet backbone (softmax head stripped) + ArcFace head."""

    def __init__(self, channels, fusion, n_class, s=32.0, m=0.5):
        super().__init__()
        self.net = DFNet(channels, fusion, n_class=n_class)
        self.net.head[-1] = nn.Identity()               # penultimate 512-d
        self.arc = ArcFaceHead(512, n_class, s=s, m=m)

    def feats(self, *xs):
        return Fn.normalize(self.net(*xs), dim=1)

    def forward(self, *xs, y=None):
        return self.arc(self.feats(*xs), y)


@torch.no_grad()
def cosmat(model, loader, device):
    model.eval()
    out = []
    for batch in loader:
        xs, _, _ = unpack(batch)
        xs = [x.to(device) for x in xs]
        out.append(model.arc(model.feats(*xs)).float().cpu().numpy())
    return np.concatenate(out)                          # (N, C) cosines


def softmax_np(S):
    S = S - S.max(1, keepdims=True)
    E = np.exp(S)
    return E / E.sum(1, keepdims=True)


def fit_s_eval(cos_val_probe, target_median):
    """Pick s so that median(max softmax(s*cos)) == target_median."""
    lo, hi = 1.0, 200.0
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        med = np.median(softmax_np(mid * cos_val_probe).max(1))
        if med > target_median:                         # too sharp -> lower s
            hi = mid
        else:
            lo = mid
    return 0.5 * (lo + hi)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="10s")
    ap.add_argument("--protocol", default="strict")
    ap.add_argument("--channels",
                    default="video_cdn,api,channel_info,thumbnail,other")
    ap.add_argument("--fusion", default="branch")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--min-epochs", type=int, default=15)
    ap.add_argument("--patience", type=int, default=12)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--s-train", type=float, default=32.0)
    ap.add_argument("--m", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--tag", default="dfs_branch_arc")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    rcfg = RunConfig(dataset=args.dataset, protocol=args.protocol)
    index = SessionIndex(rcfg.dataset_dir, rcfg.bin_size_ms)
    splits = load_splits(rcfg.splits_path)
    print(split_summary(splits))
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
    cache = load_series_cache(index, args.dataset)
    channels = [c for c in args.channels.split(",") if c]

    def mk(pool, augment):
        return DFData(cache, pool, labels, channels, args.fusion, "logsize",
                      False, augment=augment, seed=args.seed)

    ds = {name: mk(pool, name == "train") for name, pool in pools.items()}
    ld = {name: DataLoader(d, batch_size=args.batch if name == "train" else 256,
                           shuffle=name == "train", collate_fn=df_collate)
          for name, d in ds.items()}

    model = ArcDF(channels, args.fusion, len(known),
                  s=args.s_train, m=args.m).to(device)
    n_par = sum(p.numel() for p in model.parameters())
    print(f"[model] ArcDF branch channels={channels} s={args.s_train} "
          f"m={args.m} params={n_par / 1e6:.2f}M")

    out_dir = PROJECT_ROOT / "runs" / args.tag
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt = out_dir / "model.pt"

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    steps = len(ld["train"]) * args.epochs
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr,
                                                total_steps=steps, pct_start=0.08)
    scaler = torch.amp.GradScaler(enabled=device == "cuda")
    crit = torch.nn.CrossEntropyLoss()

    best = {"top1": -1.0, "state": None, "epoch": -1}
    t0, hist = time.time(), []
    for ep in range(args.epochs):
        model.train()
        for batch in ld["train"]:
            xs, y, _ = unpack(batch)
            xs = [x.to(device, non_blocking=True) for x in xs]
            y = y.to(device)
            with torch.autocast(device_type="cuda", dtype=torch.float16,
                                enabled=device == "cuda"):
                loss = crit(model(*xs, y=y), y)
            opt.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
        cv = cosmat(model, ld["val_probe"], device)
        top1 = float((cv.argmax(1) == ds["val_probe"].labels).mean())
        hist.append({"epoch": ep, "val_top1": top1})
        mark = ""
        if top1 > best["top1"]:
            best.update(top1=top1, state=copy.deepcopy(model.state_dict()),
                        epoch=ep)
            mark = " *"
        print(f"ep{ep:03d} val_top1={top1:.4f}{mark}  ({time.time() - t0:.0f}s)",
              flush=True)
        if ep >= args.min_epochs and ep - best["epoch"] >= args.patience:
            print(f"early stop at {ep} (best ep{best['epoch']})")
            break
    model.load_state_dict(best["state"])

    # ---- s_eval calibration: match e100's val-probe maxprob median --------
    ref = np.load(PROJECT_ROOT / "runs" / "dfs_branch_all_e100"
                  / "probs_val.npz")["probe"]
    ref_med = float(np.median(ref.max(1)))
    cv = cosmat(model, ld["val_probe"], device)
    s_eval = fit_s_eval(cv, ref_med)
    print(f"[s_eval] matched to e100 probe-maxprob median {ref_med:.3f} "
          f"-> s={s_eval:.1f}")

    torch.save({"state": best["state"], "args": vars(args),
                "arc": {"s_train": args.s_train, "m": args.m,
                        "s_eval": s_eval},
                "best_epoch": best["epoch"], "hist": hist,
                "channels": channels, "fusion": args.fusion}, ckpt)
    print(f"[saved] {ckpt}")

    report = {"args": vars(args), "best_epoch": best["epoch"], "hist": hist,
              "s_eval": s_eval, "splits": {}}
    for split in ("val", "test"):
        cp = cosmat(model, ld[f"{split}_probe"], device)
        ci = cosmat(model, ld[f"{split}_imp"], device)
        pv = softmax_np(s_eval * cp)
        pi = softmax_np(s_eval * ci)
        yv = ds[f"{split}_probe"].labels
        m = open_set_metrics(pv, yv, pi.max(1), n_boot=1000)
        print(format_open_set_report(f"{split.upper()} [prob]", m))
        report["splits"][split] = {"prob": m}
        np.savez(out_dir / f"probs_{split}.npz", probe=pv, probe_y=yv, imp=pi)
    (out_dir / "report.json").write_text(json.dumps(report, indent=1,
                                                    default=str),
                                         encoding="utf-8")
    print(f"[wrote] {out_dir / 'report.json'}")


if __name__ == "__main__":
    main()
