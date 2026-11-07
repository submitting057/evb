# -*- coding: utf-8 -*-
"""Diagnose whether errors/misses concentrate on truncated sessions.

For the GRO re-split dataset (10s_nogro): video_cdn events > 4096 are tail-
truncated by the default CHANNEL_LEN.  This script buckets test probes and
impostors by video_cdn event count and reports per-bucket closed-set error
and open-set miss (probe below the FPR-calibrated threshold) for each member
and for a geo combination, plus where the val magnet events live.

Usage:
  python scripts/run_ng_trunc_diag.py --dataset 10s_nogro \
      --tags a,b,c --w 0.35,0.10,0.20,0.35 --buckets 3000,4096,6000,8192
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import PROJECT_ROOT, RunConfig
from src.data import SessionIndex, load_splits, session_pool
from src.evaluate import _threshold_for_fpr


def norm(P):
    return P / (P.sum(1, keepdims=True) + 1e-12)


def geo(ws, mats):
    pairs = [(P, w) for P, w in zip(mats, ws) if w > 0]
    s = sum(w for _, w in pairs)
    L = sum(w / s * np.log(np.clip(P, 1e-12, 1)) for P, w in pairs)
    L -= L.max(1, keepdims=True)
    E = np.exp(L)
    return E / E.sum(1, keepdims=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="10s_nogro")
    ap.add_argument("--tags", required=True, help="3 DF tags (e100,dt,aux)")
    ap.add_argument("--w", required=True, help="4 geo weights lgbm,e100,dt,aux")
    ap.add_argument("--buckets", default="3000,4096,6000,8192")
    ap.add_argument("--out", default="ng_trunc_diag")
    args = ap.parse_args()
    tags = [t for t in args.tags.split(",") if t]
    w = [float(x) for x in args.w.split(",")]

    rcfg = RunConfig(dataset=args.dataset, protocol="strict")
    splits = load_splits(rcfg.splits_path)
    index = SessionIndex(rcfg.dataset_dir, rcfg.bin_size_ms)
    pools = dict(
        val_probe=session_pool(splits, "known", "val", gallery=False),
        val_imp=session_pool(splits, "unknown", "val"),
        test_probe=session_pool(splits, "known", "test", gallery=False),
        test_imp=session_pool(splits, "unknown", "test"),
    )

    # video_cdn event counts per session from the series cache
    # (cache row order == index.sessions order used at build time)
    z = np.load(PROJECT_ROOT / "runs" / "cache" / f"df_series_{args.dataset}.npz")
    ids = [rec["session_id"] for rec in index.sessions]
    pos = {s: i for i, s in enumerate(ids)}
    C, offs = z["chans"], z["offs"]
    nvc = {}
    for sid, i in pos.items():
        a, b = offs[i], offs[i + 1]
        nvc[sid] = int((C[a:b] == 0).sum())      # 0 = video_cdn

    members = ["lgbm"] + tags
    mem = {}
    lz = np.load(PROJECT_ROOT / "runs" / "cache"
                 / f"lgbm_probs_{args.dataset}_strict.npz")
    mem["lgbm"] = {s: (norm(lz[f"{s}_probe"]), norm(lz[f"{s}_imp"]))
                   for s in ("val", "test")}
    for t in tags:
        mem[t] = {}
        for s in ("val", "test"):
            d = np.load(PROJECT_ROOT / "runs" / t / f"probs_{s}.npz")
            mem[t][s] = (norm(d["probe"]), norm(d["imp"]))
    y = {"test": np.load(PROJECT_ROOT / "runs" / tags[0]
                         / "probs_test.npz")["probe_y"]}

    P = geo(w, [mem[n]["test"][0] for n in members])
    I = geo(w, [mem[n]["test"][1] for n in members])
    SP, CO = P.max(1), P.argmax(1) == y["test"]
    Iv = geo(w, [mem[n]["val"][1] for n in members]).max(1)

    edges = [int(x) for x in args.buckets.split(",")] + [10 ** 9]
    rep = {"config": vars(args), "buckets": edges}

    def bucket(v):
        for i, e in enumerate(edges):
            if v <= e:
                return i
        return len(edges)

    print(f"{'bucket(video_cdn events)':28s} {'n':>5s} {'top1':>6s} "
          f"{'TPR@0.5%':>8s} {'imp_med_conf':>12s}")
    rep["probe_buckets"] = []
    nb = len(edges)
    cnt = [0] * nb
    cor = [0] * nb
    tpr = [0.0] * nb
    imp_med = [0.0] * nb
    imp_cnt = [0] * nb
    for i, sid in enumerate(pools["test_probe"]):
        b = bucket(nvc[sid])
        cnt[b] += 1
        cor[b] += int(CO[i])
    for i, sid in enumerate(pools["test_imp"]):
        b = bucket(nvc[sid])
        imp_cnt[b] += 1
    for fr in (0.005,):
        th = _threshold_for_fpr(Iv, fr)
        for i, sid in enumerate(pools["test_probe"]):
            b = bucket(nvc[sid])
            tpr[b] += int(CO[i] & (SP[i] >= th))
    si = I.max(1)
    for i, sid in enumerate(pools["test_imp"]):
        b = bucket(nvc[sid])
        imp_med[b] += si[i]
    for b in range(nb):
        if cnt[b] == 0:
            continue
        row = {"bucket": f"<={edges[b]}", "n": cnt[b],
               "top1": cor[b] / cnt[b],
               "tpr@0.5%": tpr[b] / cnt[b],
               "imp_n": imp_cnt[b],
               "imp_med_conf": (imp_med[b] / imp_cnt[b]) if imp_cnt[b] else None}
        rep["probe_buckets"].append(row)
        print(f"<={edges[b]:>8d}                 {cnt[b]:5d} "
              f"{row['top1']:6.3f} {row['tpr@0.5%']:8.3f} "
              f"{(row['imp_med_conf'] or 0):12.3f}")

    # val magnets: where do top-0.5% val impostor events live?
    th = _threshold_for_fpr(Iv, 0.005)
    mag = np.where(Iv >= th)[0]
    mag_nvc = [nvc[s] for s in [pools["val_imp"][i] for i in mag]]
    rep["magnets"] = {
        "n": int(len(mag)),
        "median_nvc": float(np.median(mag_nvc)),
        "frac_gt4096": float(np.mean(np.array(mag_nvc) > 4096)),
        "imp_all_frac_gt4096": float(np.mean(
            [nvc[s] > 4096 for s in pools["val_imp"]])),
    }
    print(f"\nval magnets: n={len(mag)}, median video_cdn events "
          f"{rep['magnets']['median_nvc']:.0f}, frac>4096 "
          f"{rep['magnets']['frac_gt4096']:.3f} "
          f"(all val impostors frac>4096 {rep['magnets']['imp_all_frac_gt4096']:.3f})")

    out = PROJECT_ROOT / "runs" / args.out
    out.mkdir(parents=True, exist_ok=True)
    (out / "report.json").write_text(json.dumps(rep, indent=1, default=str),
                                     encoding="utf-8")
    print(f"[wrote] {out / 'report.json'}")


if __name__ == "__main__":
    main()
