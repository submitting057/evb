# -*- coding: utf-8 -*-
"""Anti-correlation joint training of member B against frozen member A
(complementarity_training.md, the only untried cell of the correlation
genealogy: actively decorrelated co-training vs passively independent members).

Setup (proposal 3.1/3.2/3.4):
  A = frozen ngL_dfs_branch_vid_dt (the strongest DF seat, w=.30 in geo4)
  B = DF-vid same backbone/channels (video_cdn:8192, no dt), trained with
      L = CE + lam * E_pseudo[ s_A(x) * p_B(x, c_A(x)) ]            (design 1)
      or, on the deployed geometric product q ~ p_A^wa * p_B^wb,
      L = CE(y, q) + lam * E_pseudo[ max_c q(c) ]                    (design 2)
  pseudo-impostors = TRAIN-split sessions of K known videos statically held
  out from B's CE (their classes are blind to B during the main phase; the
  K videos' val/test probe rows stay in the standard pools so every artifact
  stays row-aligned with the combiner).

Repair phase (optional, addresses the blind-class probe cost): plain CE
fine-tune on the FULL 386-video train pool at low lr, no anti term.  Outputs
to runs/<tag>_rep/.  Gate-1 (val-impostor argmax agreement vs A) is applied
separately to both variants; selection never touches test.

Protocol: val-impostor agreement/co-peak are used as TRAINING MONITORS and
selection signals only (gradients never flow into any unknown-video session);
test is dumped in the standard prob format but never used for selection.
"""
import argparse
import copy
import itertools
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
from src.dfnet import DFData, DFNet, df_collate, load_series_cache
from src.evaluate import (format_open_set_report, open_set_metrics,
                          open_set_with_custom_score)

A_TAG = "ngL_dfs_branch_vid_dt"          # frozen anchor member
VID_TAG = "ngL_dfs_branch_vid"           # incumbent of the slot B competes for
CHANNELS = ["video_cdn"]
CHAN_LEN = {"video_cdn": 8192}


def unpack(batch):
    *raw, y, idx = batch
    xs = list(raw[0]) if len(raw) == 1 and isinstance(raw[0], (tuple, list)) else raw
    return xs, y, idx


def b_view(xs):
    """B's input = feature row 0 of the dt-featured blocks (identical to the
    dt=False dataset sample for the same seed: _augment makes no dt-dependent
    rng calls), so A (F=2) and B (F=1) see the SAME augmented view."""
    return [b[:, :1, :] for b in xs]


@torch.no_grad()
def predict(model, loader, device, full_view=False):
    model.eval()
    out = []
    for batch in loader:
        xs, _, _ = unpack(batch)
        xs = [x.to(device) for x in (xs if full_view else b_view(xs))]
        out.append(Fn.softmax(model(*xs), 1).float().cpu().numpy())
    return np.concatenate(out)


def agreement_stats(pb_imp, pa_imp, name_a):
    """argmax agreement + co-peak diagnostics of B vs A on impostor rows."""
    ca = pa_imp.argmax(1)
    cb = pb_imp.argmax(1)
    sa = pa_imp.max(1)
    copeak = pb_imp[np.arange(len(ca)), ca]
    top21 = np.argsort(-sa)[:21]
    return {
        f"agree_vs_{name_a}": float((ca == cb).mean()),
        f"copeak_mean_vs_{name_a}": float(copeak.mean()),
        f"copeak_top21_vs_{name_a}": float(copeak[top21].mean()),
        f"agree_top21_vs_{name_a}": float((ca[top21] == cb[top21]).mean()),
    }


def evaluate_and_save(model, dirs, loaders, ds, device, extra, tag):
    """Standard prob dump + open-set report for one variant directory."""
    out = PROJECT_ROOT / "runs" / tag
    out.mkdir(parents=True, exist_ok=True)
    report = {"splits": {}, **extra}
    for split in ("val", "test"):
        pv = predict(model, loaders[f"{split}_probe"], device)
        pi = predict(model, loaders[f"{split}_imp"], device)
        y = ds[f"{split}_probe"].labels
        results = {"prob": open_set_metrics(pv, y, pi.max(1), n_boot=1000)}
        margin = np.partition(pv, -2, 1)[:, -1] - np.partition(pv, -2, 1)[:, -2]
        margin_i = np.partition(pi, -2, 1)[:, -1] - np.partition(pi, -2, 1)[:, -2]
        m = open_set_with_custom_score(pv, y, margin, margin_i)
        m["n_probes"], m["n_impostors"] = len(y), len(pi)
        m["auroc_detection"] = 0.0
        results["margin"] = m
        report["splits"][split] = results
        np.savez(out / f"probs_{split}.npz", probe=pv, probe_y=y, imp=pi)
        for mode, mm in results.items():
            print(format_open_set_report(f"{tag} {split.upper()} [{mode}]", mm))
    # blind-class probe damage (main phase only; repaired model saw all classes)
    pv = np.load(out / "probs_val.npz")
    yv = pv["probe_y"]
    blind = np.isin(yv, dirs["blind_classes"])
    if blind.any():
        cor = pv["probe"].argmax(1) == yv
        report["probe_damage"] = {
            "n_blind_rows": int(blind.sum()),
            "top1_blind": float(cor[blind].mean()),
            "top1_nonblind": float(cor[~blind].mean()),
            "top1_overall": float(cor.mean())}
        print(f"[blind-class rows] top1 blind={cor[blind].mean():.3f} "
              f"non-blind={cor[~blind].mean():.3f} overall={cor.mean():.3f}")
    (out / "report.json").write_text(json.dumps(report, indent=1, default=str),
                                     encoding="utf-8")
    print(f"[wrote] {out / 'report.json'}")
    return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="10s_nogro")
    ap.add_argument("--design", type=int, choices=(1, 2), required=True)
    ap.add_argument("--lam", type=float, required=True,
                    help="anti-correlation weight (proposal sweep .05/.2/.5/1)")
    ap.add_argument("--k-holdout", type=int, default=16)
    ap.add_argument("--holdout-seed", type=int, default=123)
    ap.add_argument("--wa", type=float, default=2 / 3,
                    help="design 2: weight of A in the product (B gets 1-wa)")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--min-epochs", type=int, default=15)
    ap.add_argument("--patience", type=int, default=12)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--repair-epochs", type=int, default=0)
    ap.add_argument("--repair-lr", type=float, default=1e-4)
    ap.add_argument("--repair-pick", default="top1",
                    choices=("top1", "agree30", "agree25"),
                    help="best-epoch criterion during repair: plain top1, or "
                         "top1 subject to val-imp agreement vs A staying "
                         "below .30 / .25 (knee selection, val-only)")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--init-from", default="",
                    help="skip the main phase, load runs/<tag>/model.pt "
                         "(repair-only iteration)")
    args = ap.parse_args()
    mseed = args.seed
    torch.manual_seed(mseed)
    np.random.seed(mseed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    rcfg = RunConfig(dataset=args.dataset, protocol="strict")
    rcfg.split.seed = 42                     # split seed is fixed s42 (protocol)
    index = SessionIndex(rcfg.dataset_dir, rcfg.bin_size_ms)
    splits = load_splits(rcfg.splits_path)
    print(split_summary(splits))
    known = sorted(splits["known_videos"])
    vidx = {v: i for i, v in enumerate(known)}
    labels = {s: vidx[r["video"]] if r["role"] == "known" else -1
              for s, r in splits["sessions"].items()}

    # ---- pools: CE excludes the K held-out videos entirely ---------------
    holdout = sorted(np.random.default_rng(args.holdout_seed).choice(
        len(known), size=args.k_holdout, replace=False))
    hold_videos = [known[i] for i in holdout]
    blind_classes = sorted(vidx[v] for v in hold_videos)
    hset = set(hold_videos)
    pools = dict(
        train=session_pool(splits, "known", "train"),
        val_probe=session_pool(splits, "known", "val", gallery=False),
        val_imp=session_pool(splits, "unknown", "val"),
        test_probe=session_pool(splits, "known", "test", gallery=False),
        test_imp=session_pool(splits, "unknown", "test"),
    )
    ce_train = [s for s in pools["train"]
                if splits["sessions"][s]["video"] not in hset]
    pseudo = [s for s in pools["train"]
              if splits["sessions"][s]["video"] in hset]
    print(f"[holdout] K={args.k_holdout} videos, {len(pseudo)} pseudo-impostor "
          f"TRAIN sessions, CE pool {len(ce_train)} (full {len(pools['train'])})")
    print(f"[holdout] videos: {hold_videos}")

    cache = load_series_cache(index, args.dataset)

    def mk(pool, augment, dt, seed):
        return DFData(cache, pool, labels, CHANNELS, "branch", "logsize",
                      dt_feat=dt, augment=augment, seed=seed, lens=CHAN_LEN)

    # all datasets carry the dt row (A needs it); B consumes row 0 only
    ds = {name: mk(pool, name == "train", True, mseed)
          for name, pool in pools.items()}
    ds["ce_train"] = mk(ce_train, True, True, mseed)
    ds["pseudo"] = mk(pseudo, True, True, mseed + 7)
    ld = {name: DataLoader(d, batch_size=args.batch if name in ("train", "ce_train")
                           else 256, shuffle=name in ("train", "ce_train", "pseudo"),
                           collate_fn=df_collate)
          for name, d in ds.items()}

    # ---- frozen A --------------------------------------------------------
    A = DFNet(CHANNELS, "branch", n_class=len(known), F=2, lens=CHAN_LEN).to(device)
    saved = torch.load(PROJECT_ROOT / "runs" / A_TAG / "model.pt",
                       map_location=device, weights_only=False)
    A.load_state_dict(saved["state"])
    A.eval()
    for p in A.parameters():
        p.requires_grad_(False)

    # A's targets on the pseudo pool (raw, no augmentation; c_A stable under
    # the mild augment, and this keeps A out of the inner loop for design 1)
    dsA = mk(pseudo, False, True, mseed)
    with torch.no_grad():
        pa = predict(A, DataLoader(dsA, batch_size=256, collate_fn=df_collate),
                     device, full_view=True)
    s_A = torch.from_numpy(pa.max(1).astype(np.float32)).to(device)
    c_A = torch.from_numpy(pa.argmax(1).astype(np.int64)).to(device)
    print(f"[A-on-pseudo] s_A mean={s_A.mean():.3f} q10={s_A.quantile(.1):.3f} "
          f"q50={s_A.quantile(.5):.3f}; argmax==true class "
          f"{float((pa.argmax(1) == dsA.labels).mean()):.3f}")

    # reference members for monitoring (argmax only)
    imp_ref = {}
    for ref in (A_TAG, VID_TAG):
        d = np.load(PROJECT_ROOT / "runs" / ref / "probs_val.npz")
        imp_ref[ref] = d["imp"] / d["imp"].sum(1, keepdims=True)
    A_imp_arg = imp_ref[A_TAG].argmax(1)
    VID_imp_arg = imp_ref[VID_TAG].argmax(1)

    model = DFNet(CHANNELS, "branch", n_class=len(known), F=1,
                  lens=CHAN_LEN).to(device)
    print(f"[model] B params={sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")

    dirs = {"blind_classes": np.array(blind_classes)}
    common = {"args": vars(args), "holdout_videos": hold_videos,
              "blind_classes": blind_classes, "A_tag": A_TAG}

    def monitor():
        """val probe top1 + val-impostor agreement/co-peak vs A and vid."""
        pv = predict(model, ld["val_probe"], device)
        pi = predict(model, ld["val_imp"], device)
        top1 = float((pv.argmax(1) == ds["val_probe"].labels).mean())
        cb = pi.argmax(1)
        ca = A_imp_arg
        sa = imp_ref[A_TAG].max(1)
        cop = pi[np.arange(len(ca)), ca]
        t21 = np.argsort(-sa)[:21]
        mon = {"top1": top1,
               "agree_A": float((cb == ca).mean()),
               "copeak_A": float(cop.mean()),
               "copeak_top21_A": float(cop[t21].mean()),
               "agree_vid": float((cb == VID_imp_arg).mean())}
        return mon

    # ---- main phase -------------------------------------------------------
    hist = []
    best = {"top1": -1.0, "state": None, "epoch": -1}
    ckpt = PROJECT_ROOT / "runs" / args.tag / "model.pt"
    ckpt.parent.mkdir(parents=True, exist_ok=True)
    if args.init_from:
        saved = torch.load(PROJECT_ROOT / "runs" / args.init_from / "model.pt",
                           map_location=device, weights_only=False)
        model.load_state_dict(saved["state"])
        print(f"[init] loaded runs/{args.init_from}/model.pt "
              f"(best_epoch={saved.get('best_epoch')})")
    else:
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
        steps = len(ld["ce_train"]) * args.epochs
        sched = torch.optim.lr_scheduler.OneCycleLR(
            opt, max_lr=args.lr, total_steps=steps, pct_start=0.08)
        scaler = torch.amp.GradScaler(enabled=device == "cuda")
        crit = torch.nn.CrossEntropyLoss(label_smoothing=0.05)
        t0 = time.time()
        for ep in range(args.epochs):
            model.train()
            # fresh pseudo iterator each epoch; re-iterate (with fresh
            # augmentation draws) whenever the small pseudo pool exhausts --
            # itertools.cycle would replay the SAME cached augmented batches
            pseudo_it = None
            def next_pseudo():
                nonlocal pseudo_it
                try:
                    return next(pseudo_it)
                except (TypeError, StopIteration):
                    pseudo_it = iter(ld["pseudo"])
                    return next(pseudo_it)
            for batch in ld["ce_train"]:
                xs, y, _ = unpack(batch)
                xs = [x.to(device, non_blocking=True) for x in xs]
                y = y.to(device)
                xb = [x.to(device, non_blocking=True) for x in b_view(xs)]
                xp, _, ip = unpack(next_pseudo())
                xp = [x.to(device, non_blocking=True) for x in xp]
                xpb = [x.to(device, non_blocking=True) for x in b_view(xp)]
                ip = ip.to(device)
                with torch.autocast(device_type="cuda", dtype=torch.float16,
                                    enabled=device == "cuda"):
                    lb = model(*xb)
                    lpb = model(*xpb)
                if args.design == 1:
                    # direct decorrelation: penalize B's prob at A's argmax,
                    # weighted by A's confidence s_A
                    l_ce = crit(lb, y)
                    p_at = Fn.softmax(lpb.float(), 1)[
                        torch.arange(len(ip), device=device), c_A[ip]]
                    l_anti = (s_A[ip] * p_at).mean()
                    loss = l_ce + args.lam * l_anti
                else:
                    # design 2: optimize the deployed product
                    # q(c) ∝ p_A(c)^wa · p_B(c)^(1-wa)
                    # CE batch: product must answer y (built-in rescue gradients)
                    with torch.no_grad():
                        la2 = Fn.log_softmax(A(*xs).float(), 1)
                    lpa2 = Fn.log_softmax(lb.float(), 1)
                    z2 = args.wa * la2 + (1 - args.wa) * lpa2
                    logq2 = z2 - torch.logsumexp(z2, 1, keepdim=True)
                    l_ce2 = Fn.cross_entropy(logq2, y, label_smoothing=0.05)
                    # pseudo batch: penalize the product's peak
                    with torch.no_grad():
                        la = Fn.log_softmax(A(*xp).float(), 1)
                    lpa = Fn.log_softmax(lpb.float(), 1)
                    z = args.wa * la + (1 - args.wa) * lpa
                    logq = z - torch.logsumexp(z, 1, keepdim=True)
                    l_anti2 = torch.exp(logq).max(1).values.mean()
                    loss = l_ce2 + args.lam * l_anti2
                opt.zero_grad()
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
                sched.step()
            mon = monitor()
            hist.append({"epoch": ep, **mon})
            mark = ""
            if mon["top1"] > best["top1"]:
                best.update(top1=mon["top1"],
                            state=copy.deepcopy(model.state_dict()), epoch=ep)
                mark = " *"
            print(f"ep{ep:03d} top1={mon['top1']:.4f} agreeA={mon['agree_A']:.3f} "
                  f"copk={mon['copeak_A']:.3f} agreeVid={mon['agree_vid']:.3f}"
                  f"{mark}  ({time.time() - t0:.0f}s)")
            if ep >= args.min_epochs and ep - best["epoch"] >= args.patience:
                print(f"early stop at {ep} (best ep{best['epoch']})")
                break
        model.load_state_dict(best["state"])
        torch.save({"state": best["state"], "args": vars(args),
                    "best_epoch": best["epoch"], "hist": hist,
                    "channels": CHANNELS, "chan_len": CHAN_LEN,
                    "fusion": "branch"}, ckpt)
        print(f"[saved] {ckpt}")

    rep = evaluate_and_save(
        model, dirs, ld, ds, device,
        {**common, "best_epoch": best["epoch"], "hist": hist,
         "design": args.design, "lam": args.lam, "phase": "main"},
        args.tag)
    # gate-1 metrics (val only) into the report
    pv = np.load(PROJECT_ROOT / "runs" / args.tag / "probs_val.npz")
    pi = pv["imp"] / pv["imp"].sum(1, keepdims=True)
    g1 = {}
    g1.update(agreement_stats(pi, imp_ref[A_TAG], "A"))
    g1.update(agreement_stats(pi, imp_ref[VID_TAG], "vid"))
    rep["gate1_val"] = g1
    print("[gate1-val] " + "  ".join(f"{k}={v:.3f}" for k, v in g1.items()))
    (PROJECT_ROOT / "runs" / args.tag / "report.json").write_text(
        json.dumps(rep, indent=1, default=str), encoding="utf-8")

    # ---- repair phase: plain CE on the FULL train pool, low lr -----------
    if args.repair_epochs > 0:
        tag_r = f"{args.tag}_rep"
        ds_full = mk(pools["train"], True, False, mseed)
        ld_full = DataLoader(ds_full, batch_size=args.batch, shuffle=True,
                             collate_fn=df_collate)
        opt = torch.optim.AdamW(model.parameters(), lr=args.repair_lr,
                                weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.OneCycleLR(
            opt, max_lr=args.repair_lr,
            total_steps=len(ld_full) * args.repair_epochs, pct_start=0.1)
        scaler = torch.amp.GradScaler(enabled=device == "cuda")
        crit = torch.nn.CrossEntropyLoss(label_smoothing=0.05)
        best_r = {"top1": -1.0, "state": None, "epoch": -1}
        cap = {"top1": None, "agree30": .30, "agree25": .25}[args.repair_pick]
        hist_r = []
        t0 = time.time()
        for ep in range(args.repair_epochs):
            model.train()
            for batch in ld_full:
                xs, y, _ = unpack(batch)
                xb = [x.to(device, non_blocking=True) for x in b_view(xs)]
                y = y.to(device)
                with torch.autocast(device_type="cuda", dtype=torch.float16,
                                    enabled=device == "cuda"):
                    loss = crit(model(*xb), y)
                opt.zero_grad()
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
                sched.step()
            mon = monitor()
            hist_r.append({"epoch": ep, **mon})
            mark = ""
            ok = (cap is None or mon["agree_A"] <= cap)
            if ok and mon["top1"] > best_r["top1"]:
                best_r.update(top1=mon["top1"],
                              state=copy.deepcopy(model.state_dict()), epoch=ep)
                mark = " *"
            print(f"[rep] ep{ep:03d} top1={mon['top1']:.4f} "
                  f"agreeA={mon['agree_A']:.3f} copk={mon['copeak_A']:.3f}"
                  f"{mark}  ({time.time() - t0:.0f}s)")
        if best_r["state"] is None:            # no epoch met the agree cap
            print(f"[repair] WARNING: no epoch with agree_A <= {cap}; "
                  f"keeping the LAST state (curve is in hist_r)")
            best_r = {"top1": hist_r[-1]["top1"],
                      "state": copy.deepcopy(model.state_dict()),
                      "epoch": hist_r[-1]["epoch"]}
        model.load_state_dict(best_r["state"])
        rep_dir = PROJECT_ROOT / "runs" / tag_r
        rep_dir.mkdir(parents=True, exist_ok=True)
        torch.save({"state": best_r["state"], "args": vars(args),
                    "best_epoch": best_r["epoch"], "hist": hist_r,
                    "channels": CHANNELS, "chan_len": CHAN_LEN,
                    "fusion": "branch", "init_from": args.tag or args.init_from},
                   rep_dir / "model.pt")
        rep_r = evaluate_and_save(
            model, dirs, ld, ds, device,
            {**common, "best_epoch": best_r["epoch"], "hist": hist_r,
             "design": args.design, "lam": args.lam, "phase": "repair"},
            tag_r)
        pvr = np.load(PROJECT_ROOT / "runs" / tag_r / "probs_val.npz")
        pir = pvr["imp"] / pvr["imp"].sum(1, keepdims=True)
        g1r = {}
        g1r.update(agreement_stats(pir, imp_ref[A_TAG], "A"))
        g1r.update(agreement_stats(pir, imp_ref[VID_TAG], "vid"))
        rep_r["gate1_val"] = g1r
        print("[gate1-val rep] " + "  ".join(f"{k}={v:.3f}" for k, v in g1r.items()))
        (PROJECT_ROOT / "runs" / tag_r / "report.json").write_text(
            json.dumps(rep_r, indent=1, default=str), encoding="utf-8")


if __name__ == "__main__":
    main()
