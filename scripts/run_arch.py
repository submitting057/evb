# -*- coding: utf-8 -*-
"""Train + evaluate an alternative-architecture expert (src/model_zoo.py) on
the same DFData packet-index series as scripts/run_df.py.

Protocol, augmentation, hyperparameters and output layout (runs/{tag}/
{model.pt, probs_val.npz, probs_test.npz, report.json}) are kept identical to
run_df.py so members are drop-in for run_combiner.py and run_paired_boot.py.

Usage:
  python scripts/run_arch.py --dataset 10s_nogro --protocol strict \
      --arch resnet --channels video_cdn --chan-len video_cdn:8192 \
      --epochs 100 --tag ngL_res_vid
"""
import argparse
import copy
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as Fn
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import PROJECT_ROOT, RunConfig
from src.data import SessionIndex, load_splits, session_pool, split_summary
from src.dfnet import DFData, df_collate, load_series_cache
from src.evaluate import format_open_set_report, open_set_metrics, \
    open_set_with_custom_score
from src.model_zoo import BranchNet


def unpack(batch):
    *raw, y, idx = batch
    xs = list(raw[0]) if len(raw) == 1 and isinstance(raw[0], (tuple, list)) else raw
    return xs, y, idx


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    out = []
    for batch in loader:
        xs, y, idx = unpack(batch)
        xs = [x.to(device) for x in xs]
        logits = model(*xs)
        out.append(Fn.softmax(logits, 1).float().cpu().numpy())
    return np.concatenate(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="10s_nogro")
    ap.add_argument("--protocol", default="strict", choices=["legacy", "strict"])
    ap.add_argument("--arch", required=True, choices=["resnet", "tcn", "vit"])
    ap.add_argument("--channels", default="video_cdn")
    ap.add_argument("--dt", action="store_true")
    ap.add_argument("--chan-len", default="",
                    help="per-channel length overrides, 'video_cdn:8192,...'")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--min-epochs", type=int, default=15)
    ap.add_argument("--patience", type=int, default=12)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--head-drop", default="",
                    help="override head dropout 'd1,d2' (vit uses 0.3,0.2 by "
                         "default -- 0.7/0.5 was sized for DF's 8k-dim flatten)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--model-seed", type=int, default=None)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--load", action="store_true",
                    help="skip training, evaluate runs/{tag}/model.pt")
    args = ap.parse_args()
    mseed = args.model_seed if args.model_seed is not None else args.seed

    torch.manual_seed(mseed)
    np.random.seed(mseed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    rcfg = RunConfig(dataset=args.dataset, protocol=args.protocol)
    rcfg.split.seed = args.seed
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
    chan_len = {}
    for part in filter(None, args.chan_len.split(",")):
        name, _, val = part.partition(":")
        chan_len[name.strip()] = int(val)

    def mk(pool, augment):
        return DFData(cache, pool, labels, channels, "branch", "logsize",
                      args.dt, augment=augment, seed=mseed, lens=chan_len)

    ds = {name: mk(pool, name == "train") for name, pool in pools.items()}
    ld = {name: DataLoader(d, batch_size=args.batch if name == "train" else 256,
                           shuffle=name == "train", collate_fn=df_collate)
          for name, d in ds.items()}

    from src.dfnet import CHANNEL_LEN
    clen = dict(CHANNEL_LEN)
    clen.update(chan_len)
    lens = [clen[c] for c in channels]
    hd = (0.3, 0.2) if (args.arch == "vit" and not args.head_drop) else \
        tuple(float(v) for v in args.head_drop.split(",")) if args.head_drop else (0.7, 0.5)
    model = BranchNet(args.arch, channels, lens, n_class=len(known),
                      F=2 if args.dt else 1, head_drop=hd).to(device)
    n_par = sum(p.numel() for p in model.parameters())
    print(f"[model] arch={args.arch} channels={channels} dt={args.dt} "
          f"chan_len={chan_len or 'default'} params={n_par / 1e6:.2f}M "
          f"flat={model._flat}")

    out_dir = PROJECT_ROOT / "runs" / args.tag
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt = out_dir / "model.pt"

    if args.load:
        saved = torch.load(ckpt, map_location=device, weights_only=False)
        model.load_state_dict(saved["state"])
        best = {"top1": float("nan"), "state": saved["state"],
                "epoch": saved.get("best_epoch", -1)}
        hist = saved.get("hist", [])
        print(f"[load] {ckpt} (best_epoch={best['epoch']})")
    else:
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
        steps = len(ld["train"]) * args.epochs
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr,
                                                    total_steps=steps,
                                                    pct_start=0.08)
        scaler = torch.amp.GradScaler(enabled=device == "cuda")
        crit = torch.nn.CrossEntropyLoss(label_smoothing=0.05)

        best = {"top1": -1.0, "state": None, "epoch": -1}
        t0 = time.time()
        hist = []
        for ep in range(args.epochs):
            model.train()
            for batch in ld["train"]:
                xs, y, idx = unpack(batch)
                xs = [x.to(device, non_blocking=True) for x in xs]
                y = y.to(device)
                with torch.autocast(device_type="cuda", dtype=torch.float16,
                                    enabled=device == "cuda"):
                    loss = crit(model(*xs), y)
                opt.zero_grad()
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
                sched.step()
            pv = predict(model, ld["val_probe"], device)
            top1 = float((pv.argmax(1) == ds["val_probe"].labels).mean())
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
        torch.save({"state": best["state"], "args": vars(args),
                    "best_epoch": best["epoch"], "hist": hist,
                    "flat": model._flat, "channels": channels,
                    "chan_len": chan_len, "arch": args.arch}, ckpt)
        print(f"[saved] {ckpt}")

    report = {"args": vars(args), "best_epoch": best["epoch"], "hist": hist,
              "params_m": n_par / 1e6, "splits": {}}
    for split in ("val", "test"):
        pv = predict(model, ld[f"{split}_probe"], device)
        pi = predict(model, ld[f"{split}_imp"], device)
        y = ds[f"{split}_probe"].labels
        results = {"prob": open_set_metrics(pv, y, pi.max(1), n_boot=1000)}
        margin = np.partition(pv, -2, 1)[:, -1] - np.partition(pv, -2, 1)[:, -2]
        margin_i = np.partition(pi, -2, 1)[:, -1] - np.partition(pi, -2, 1)[:, -2]
        m = open_set_with_custom_score(pv, y, margin, margin_i)
        m["n_probes"], m["n_impostors"] = len(y), len(pi)
        m["auroc_detection"] = 0.0
        results["margin"] = m
        for mode, mm in results.items():
            print(format_open_set_report(f"{split.upper()} [{mode}]", mm))
        report["splits"][split] = results
        np.savez(out_dir / f"probs_{split}.npz", probe=pv, probe_y=y, imp=pi)
    (out_dir / "report.json").write_text(json.dumps(report, indent=1, default=str),
                                         encoding="utf-8")
    print(f"[wrote] {out_dir / 'report.json'}")


if __name__ == "__main__":
    main()
