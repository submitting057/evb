# -*- coding: utf-8 -*-
"""Train + evaluate the modified DF CNN on multi-channel signed-size series.

Usage:
  python scripts/run_df.py --dataset 10s --fusion branch \
      --channels video_cdn,thumbnail,channel_info --side seg --tag df_b_vtc
Evaluation protocol identical to run_lgbm_v2 (same splits/pools/scorers).
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

from src.config import DATASET_DIRS, PROJECT_ROOT, RunConfig
from src.data import SessionIndex, load_splits, session_pool, split_summary
from src.dfnet import DFData, DFNet, df_collate, load_series_cache
from src.evaluate import (format_open_set_report, open_set_metrics,
                          open_set_with_custom_score)
from src.segment import segment_features_v2


def get_seg_feats(index, sids, dataset):
    cache = PROJECT_ROOT / "runs" / "cache" / f"segv2_{dataset}.npy"
    if not (cache.exists() and np.load(cache, mmap_mode="r").shape[0] == len(sids)):
        X = np.stack([segment_features_v2(index, s) for s in sids])
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.save(cache, X)
    return np.load(cache).astype(np.float32)


def unpack(batch):
    """-> (list_of_input_tensors, y, idx); branch collate wraps inputs in a tuple."""
    *raw, y, idx = batch
    xs = list(raw[0]) if len(raw) == 1 and isinstance(raw[0], (tuple, list)) else raw
    return xs, y, idx


@torch.no_grad()
def predict(model, loader, device, seg_std, use_side):
    model.eval()
    out = []
    for batch in loader:
        xs, y, idx = unpack(batch)
        xs = [x.to(device) for x in xs]
        side = seg_std[idx].to(device) if use_side else None
        logits = model(*xs, side=side)
        out.append(Fn.softmax(logits, 1).float().cpu().numpy())
    return np.concatenate(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="10s")
    ap.add_argument("--protocol", default="legacy", choices=["legacy", "strict"])
    ap.add_argument("--channels", default="video_cdn,api,channel_info,thumbnail,other")
    ap.add_argument("--fusion", default="stack", choices=["merge", "stack", "branch"])
    ap.add_argument("--value", default="logsize", choices=["logsize", "dir"])
    ap.add_argument("--dt", action="store_true")
    ap.add_argument("--chan-emb", action="store_true")
    ap.add_argument("--chan-len", default="",
                    help="per-channel length overrides, e.g. "
                         "'video_cdn:8192,api:768' (default = CHANNEL_LEN; "
                         "pass the same value again on --load)")
    ap.add_argument("--side", default="none", choices=["none", "seg"])
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--min-epochs", type=int, default=15)
    ap.add_argument("--patience", type=int, default=12)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--model-seed", type=int, default=None,
                    help="override ONLY the model/augmentation randomness "
                         "(torch/np/DFData); the split keeps --seed. Used by "
                         "the multi-seed robustness runs so every seed "
                         "shares the same s42 strict split.")
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
    all_sids = sorted(set().union(*pools.values()))
    cache = load_series_cache(index, args.dataset)
    channels = [c for c in args.channels.split(",") if c]
    chan_len = {}
    for part in filter(None, args.chan_len.split(",")):
        name, _, val = part.partition(":")
        chan_len[name.strip()] = int(val)
    segX = get_seg_feats(index, all_sids, args.dataset)
    seg_pos = {s: i for i, s in enumerate(all_sids)}

    def mk(pool, augment):
        return DFData(cache, pool, labels, channels, args.fusion, args.value,
                      args.dt, chan_emb=args.chan_emb, augment=augment,
                      seed=mseed, lens=chan_len)

    ds = {name: mk(pool, name == "train") for name, pool in pools.items()}
    ld = {name: DataLoader(d, batch_size=args.batch if name == "train" else 256,
                           shuffle=name == "train", collate_fn=df_collate)
          for name, d in ds.items()}

    seg_rows = {name: np.array([seg_pos[s] for s in pools[name]]) for name in pools}
    use_side = args.side == "seg"
    seg_std = torch.from_numpy(segX)                     # standardized below
    if use_side:
        tr = seg_rows["train"]
        mu, sd = segX[tr].mean(0, keepdims=True), segX[tr].std(0, keepdims=True) + 1e-6
        seg_std = torch.from_numpy((segX - mu) / sd)

    model = DFNet(channels, args.fusion, n_class=len(known),
                  F=2 if args.dt else 1, side_dim=120 if use_side else 0,
                  chan_emb=args.chan_emb, lens=chan_len).to(device)
    n_par = sum(p.numel() for p in model.parameters())
    print(f"[model] fusion={args.fusion} channels={channels} value={args.value} "
          f"dt={args.dt} side={args.side} chan_emb={args.chan_emb} "
          f"chan_len={chan_len or 'default'} "
          f"params={n_par / 1e6:.2f}M flat={model._flat}")

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
                                                    total_steps=steps, pct_start=0.08)
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
                side = seg_std[idx.long()].to(device) if use_side else None
                with torch.autocast(device_type="cuda", dtype=torch.float16,
                                    enabled=device == "cuda"):
                    loss = crit(model(*xs, side=side), y)
                opt.zero_grad()
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
                sched.step()
            pv = predict(model, ld["val_probe"], device, seg_std, use_side)
            top1 = float((pv.argmax(1) == ds["val_probe"].labels).mean())
            hist.append({"epoch": ep, "val_top1": top1})
            mark = ""
            if top1 > best["top1"]:
                best.update(top1=top1, state=copy.deepcopy(model.state_dict()), epoch=ep)
                mark = " *"
            print(f"ep{ep:03d} val_top1={top1:.4f}{mark}  ({time.time() - t0:.0f}s)")
            if ep >= args.min_epochs and ep - best["epoch"] >= args.patience:
                print(f"early stop at {ep} (best ep{best['epoch']})")
                break
        model.load_state_dict(best["state"])
        torch.save({"state": best["state"], "args": vars(args),
                    "best_epoch": best["epoch"], "hist": hist,
                    "flat": model._flat, "channels": channels,
                    "chan_len": chan_len, "fusion": args.fusion}, ckpt)
        print(f"[saved] {ckpt}")

    report = {"args": vars(args), "best_epoch": best["epoch"], "hist": hist,
              "params_m": n_par / 1e6, "splits": {}}
    for split in ("val", "test"):
        pv = predict(model, ld[f"{split}_probe"], device, seg_std, use_side)
        pi = predict(model, ld[f"{split}_imp"], device, seg_std, use_side)
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
