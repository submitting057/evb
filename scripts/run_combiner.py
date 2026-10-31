# -*- coding: utf-8 -*-
"""Open-set combiner search over saved member probabilities (no retraining).

Members: LGBM (runs/cache/lgbm_probs_{ds}.npz) + DF variants (runs/{tag}/probs_*.npz).
Rules:
  arith : weighted arithmetic mean of probs
  geo   : weighted geometric mean  p_c ∝ prod_i p_i[c]^w_i   (soft AND gate)
  min   : elementwise min over members (hard AND gate, equal weights)
  + temperature: per-member T fitted on val probes (NLL) before combining.

Selection: weights/temperature/rule chosen by VAL TPR@0.5%FPR (tie-break @1%).
Test is only reported, never used for selection.
"""
import argparse
import itertools
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import PROJECT_ROOT
from src.evaluate import _threshold_for_fpr, open_set_metrics

GRID = (0.005, 0.01, 0.05)


def norm(P):
    return P / (P.sum(1, keepdims=True) + 1e-12)


def combine(rule, ws, mats):
    pairs = [(m, w) for m, w in zip(mats, ws) if w > 0]
    s = sum(w for _, w in pairs)
    pairs = [(m, w / s) for m, w in pairs]
    if not pairs:
        raise ValueError("all weights zero")
    if len(pairs) == 1:
        return pairs[0][0]
    if rule == "arith":
        return sum(w * P for P, w in pairs)
    if rule == "geo":
        L = sum(w * np.log(np.clip(P, 1e-12, 1)) for P, w in pairs)
        L -= L.max(1, keepdims=True)
        E = np.exp(L)
        return E / E.sum(1, keepdims=True)
    if rule == "min":
        return np.minimum.reduce([P for P, _ in pairs])
    raise ValueError(rule)


def quick(P, y, I):
    best, correct = P.max(1), P.argmax(1) == y
    out = {"top1": float(np.mean(correct))}
    for f in GRID:
        th = _threshold_for_fpr(I.max(1), f)
        out[f] = float(np.mean(correct & (best >= th)))
    return out


def fit_temp(P, y, grid=(0.25, 0.5, 1, 1.5, 2, 3, 4, 6, 8)):
    L = np.log(np.clip(P, 1e-12, 1))
    best = (1, np.inf)
    for T in grid:
        Lt = L / T
        Lt -= Lt.max(1, keepdims=True)
        E = np.exp(Lt)
        E /= E.sum(1, keepdims=True)
        nll = -np.mean(np.log(np.clip(E[np.arange(len(y)), y], 1e-12, 1)))
        if nll < best[1]:
            best = (T, nll)
    return best[0]


def simplex(step, k):
    vals = [round(i * step, 4) for i in range(int(round(1 / step)) + 1)]
    for w in itertools.product(vals, repeat=k):
        if abs(sum(w) - 1.0) < 1e-9:
            yield w


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="10s")
    ap.add_argument("--df-tags", required=True)
    ap.add_argument("--step", type=float, default=0.05)
    args = ap.parse_args()
    tags = [t for t in args.df_tags.split(",") if t]

    z = np.load(PROJECT_ROOT / "runs" / "cache" / f"lgbm_probs_{args.dataset}.npz")
    names = ["lgbm"] + tags
    y = {}
    raw = {}                                     # (name, split) -> (probe, imp)
    for split in ("val", "test"):
        d0 = np.load(PROJECT_ROOT / "runs" / tags[0] / f"probs_{split}.npz")
        y[split] = d0["probe_y"]
        raw[("lgbm", split)] = (norm(z[f"{split}_probe"]), norm(z[f"{split}_imp"]))
        for t in tags:
            d = np.load(PROJECT_ROOT / "runs" / t / f"probs_{split}.npz")
            assert (d["probe_y"] == y[split]).all(), "row order mismatch"
            raw[(t, split)] = (norm(d["probe"]), norm(d["imp"]))

    temps = {n: fit_temp(raw[(n, "val")][0], y["val"]) for n in names}
    print("members+temperature:", {n: temps[n] for n in names})

    def scaled(n, split, use_T):
        Pv, I = raw[(n, split)]
        if not use_T or temps[n] == 1:
            return Pv, I
        def f(M):
            L = np.log(np.clip(M, 1e-12, 1)) / temps[n]
            L -= L.max(1, keepdims=True)
            E = np.exp(L)
            return E / E.sum(1, keepdims=True)
        return f(Pv), f(I)

    variants = []
    for use_T in (False, True):
        for rule in ("arith", "geo", "min"):
            w0 = tuple(round(1.0 / len(names), 4) for _ in names)
            cands = [("equal", w0)] if rule == "min" else \
                [("equal", w0)] + [(f"w={w}", w) for w in simplex(args.step, len(names))]
            for wname, w in cands:
                Pv = combine(rule, list(w), [scaled(n, "val", use_T)[0] for n in names])
                Iv = combine(rule, list(w), [scaled(n, "val", use_T)[1] for n in names])
                variants.append({"rule": rule, "temp": use_T, "w": list(w),
                                 "wname": wname, "val": quick(Pv, y["val"], Iv)})
    variants.sort(key=lambda v: (-v["val"][0.005], -v["val"][0.01], -v["val"]["top1"]))
    print(f"\n{'rule':6s} {'T':5s} {'weights':30s} | val @.5%  @1%   @5%   top1")
    for v in variants[:12]:
        print(f"{v['rule']:6s} {str(v['temp']):5s} {v['wname']:30s} | "
              f"{v['val'][0.005]:.3f}  {v['val'][0.01]:.3f}  {v['val'][0.05]:.3f}  "
              f"{v['val']['top1']:.3f}")

    # test report: top-3 val configs + equal-weight arith baseline
    print("\n--- TEST ---")
    keys, finals = [], []
    baselines = [v for v in variants if not v["temp"] and v["rule"] == "arith"
                 and v["wname"] == "equal"]
    ordered = variants[:3] + baselines
    for v in ordered:
        key = (v["rule"], v["temp"], v["wname"])
        if key in keys:
            continue
        keys.append(key)
        Pv = combine(v["rule"], v["w"], [scaled(n, "test", v["temp"])[0] for n in names])
        Iv = combine(v["rule"], v["w"], [scaled(n, "test", v["temp"])[1] for n in names])
        m = open_set_metrics(Pv, y["test"], Iv.max(1), n_boot=1000)
        m["fpr"] = {str(k): v for k, v in m["fpr"].items()}
        tag = f"{v['rule']}|T={v['temp']}|{v['wname']}"
        print(f"{tag:40s} top1={m['closed']['top1']:.3f} "
              f"@.5%={m['fpr']['0.005']['tpr']:.3f} "
              f"@1%={m['fpr']['0.01']['tpr']:.3f} @5%={m['fpr']['0.05']['tpr']:.3f} "
              f"AUROC={m['auroc_detection']:.3f}")
        finals.append({"config": tag, "val": v["val"], "test": m})

    out = PROJECT_ROOT / "runs" / f"combo_{args.dataset}"
    out.mkdir(parents=True, exist_ok=True)
    (out / "report.json").write_text(
        json.dumps({"members": names, "temps": temps, "top_variants": variants[:12],
                    "finals": finals}, indent=1, default=str), encoding="utf-8")
    print(f"[wrote] {out / 'report.json'}")


if __name__ == "__main__":
    main()
