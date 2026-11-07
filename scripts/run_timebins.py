# -*- coding: utf-8 -*-
"""Train + evaluate the time-domain (10ms bins) DF member (src/timedata.py).

Same protocol/augmentation budget/output layout as run_arch.py; the trunk is
the proven DF architecture on a 1500-bin single-channel series (F=2 rows:
log down/up bytes of the video_cdn channel).
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
from src.dfnet import DFNet
from src.evaluate import format_open_set_report, open_set_metrics, \
    open_set_with_custom_score
from src.timedata import BinsData


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    out = []
    for x, y, idx in loader:
        x = x.to(device)
        out.append(Fn.softmax(model(x), 1).float().cpu().numpy())
    return np.concatenate(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="10s_nogro")
    ap.add_argument("--protocol", default="strict")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--min-epochs", type=int, default=15)
    ap.add_argument("--patience", type=int, default=12)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--tag", required=True)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
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
    ds = {name: BinsData(index, pool, labels, augment=name == "train",
                         seed=args.seed)
          for name, pool in pools.items()}
    ld = {name: DataLoader(d, batch_size=args.batch if name == "train" else 256,
                           shuffle=name == "train")
          for name, d in ds.items()}

    # single-channel DF trunk over the 1500-bin series, F=2
    model = DFNet(["video_cdn"], "stack", n_class=len(known), F=2,
                  lens={"video_cdn": 1500}).to(device)
    n_par = sum(p.numel() for p in model.parameters())
    print(f"[model] timebins DF stack params={n_par / 1e6:.2f}M")

    out_dir = PROJECT_ROOT / "runs" / args.tag
    out_dir.mkdir(parents=True, exist_ok=True)
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
        for x, y, idx in ld["train"]:
            x, y = x.to(device), y.to(device)
            with torch.autocast(device_type="cuda", dtype=torch.float16,
                                enabled=device == "cuda"):
                loss = crit(model(x), y)
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
                "flat": model._flat}, out_dir / "model.pt")

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
