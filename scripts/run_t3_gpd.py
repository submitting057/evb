# -*- coding: utf-8 -*-
"""T3 third item: GPD tail-calibrated thresholds, dual-caliber deliverable.

Plain-language summary:

  In deployment the threshold must be set from the VAL impostor scores.
  The plain empirical quantile at 0.5% rests on ~19 events, and the FPR it
  produces on TEST drifts upward (realized 1.12% for geo4 vs nominal 0.5%)
  -- the val score tail is slightly lighter than the test tail.  Extreme
  value theory offers a parametric alternative: fit a Generalized Pareto
  Distribution (GPD) to the excesses over a high threshold u (peaks over
  threshold), then EXTRAPOLATE the p-quantile from the fitted tail:

      P(S > s) = (n_u/N) * (1 + xi*(s-u)/sigma)^(-1/xi)
      s_p      = u + (sigma/xi) * [ ((n_u/N)/p)^xi - 1 ]

  This script reports, for the frozen geo4 system, the DEPLOYMENT caliber
  (val threshold -> test realization) at nominal 0.5%/1%/5%:
    * empirical quantile threshold  (current practice, reference row)
    * GPD threshold at u = Q90 and u = Q95 of val impostor scores,
      on the raw max-prob score and on its logit transform.
  Deliverable table: threshold, test realized FPR, test TPR.  A method
  "works" if its realized FPR lands closer to nominal than the empirical
  one -- judged descriptively (no TPR-improvement claim is being made).
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import genpareto

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import PROJECT_ROOT, RunConfig
from src.data import load_splits, session_pool

GRID = (0.005, 0.01, 0.05)
MEMBERS = ["lgbm", "dfs_branch_all_e100", "dfs_branch_all_dt", "dfs_branch_aux"]
W4 = [0.35, 0.10, 0.20, 0.35]


def norm(P):
    return P / (P.sum(1, keepdims=True) + 1e-12)


def geo(ws, mats):
    w = np.asarray(ws, float)
    w = w / w.sum()
    L = sum(wi * np.log(np.clip(P, 1e-12, 1)) for wi, P in zip(w, mats))
    L -= L.max(1, keepdims=True)
    E = np.exp(L)
    return E / E.sum(1, keepdims=True)


def logit1(p):
    p = np.clip(p, 1e-9, 1 - 1e-9)
    return np.log(p / (1 - p))


def gpd_threshold(scores, p, u_q):
    """POT quantile: fit GPD to excesses over u=quantile(scores, u_q)."""
    N = len(scores)
    u = np.quantile(scores, u_q)
    exc = scores[scores > u] - u
    n_u = len(exc)
    xi, _, sig = genpareto.fit(exc, floc=0.0)
    if abs(xi) < 1e-6:
        s = u + sig * np.log((n_u / N) / p)
    else:
        arg = ((n_u / N) / p) ** xi
        s = u + (sig / xi) * (arg - 1.0)
    return float(s), float(xi), float(sig), float(u), int(n_u)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="10s")
    ap.add_argument("--out", default="t3_gpd")
    args = ap.parse_args()

    rcfg = RunConfig(dataset=args.dataset, protocol="strict")
    splits = load_splits(rcfg.splits_path)
    pools = dict(
        val_probe=session_pool(splits, "known", "val", gallery=False),
        val_imp=session_pool(splits, "unknown", "val"),
        test_probe=session_pool(splits, "known", "test", gallery=False),
        test_imp=session_pool(splits, "unknown", "test"),
    )
    vid = {k: np.array([splits["sessions"][s]["video"] for s in v])
           for k, v in pools.items()}

    mem = {}
    z = np.load(PROJECT_ROOT / "runs" / "cache"
                / f"lgbm_probs_{args.dataset}_strict.npz")
    mem["lgbm"] = {s: (norm(z[f"{s}_probe"]), norm(z[f"{s}_imp"]))
                   for s in ("val", "test")}
    for t in MEMBERS[1:]:
        mem[t] = {}
        for s in ("val", "test"):
            d = np.load(PROJECT_ROOT / "runs" / t / f"probs_{s}.npz")
            mem[t][s] = (norm(d["probe"]), norm(d["imp"]))
    y = np.load(PROJECT_ROOT / "runs" / MEMBERS[1]
                / "probs_test.npz")["probe_y"]
    G = geo(W4, [mem[n]["test"][0] for n in MEMBERS])
    GI_val = geo(W4, [mem[n]["val"][1] for n in MEMBERS]).max(1)
    GI_test = geo(W4, [mem[n]["test"][1] for n in MEMBERS]).max(1)
    sp, correct = G.max(1), G.argmax(1) == y

    # cluster bootstrap CI for realized FPR of each method (robustness of
    # the deliverable number itself): resample test impostor videos
    iv = sorted(set(vid["test_imp"].tolist()))
    ig = [np.where(vid["test_imp"] == v)[0] for v in iv]
    rng = np.random.default_rng(0)

    transforms = {"raw": (lambda x: x, lambda x: x),
                  "logit": (logit1, lambda t: 1 / (1 + np.exp(-t)))}
    methods = {}

    rep = {"grid": list(GRID), "methods": {}}
    print("=== T3-GPD: deployment caliber (val threshold -> test) ===")
    print(f"geo4 val imp scores: q50={np.quantile(GI_val,.5):.3f} "
          f"q99={np.quantile(GI_val,.99):.3f} max={GI_val.max():.3f}")

    for tname, (fwd, inv) in transforms.items():
        sval, stest = fwd(GI_val), fwd(GI_test)
        for u_q in (0.90, 0.95):
            mname = f"gpd_{tname}_u{int(u_q*100)}"
            methods[mname] = ("gpd", tname, u_q)
        methods[f"empirical_{tname}"] = ("emp", tname, None)

    for mname, spec in methods.items():
        kind, tname, u_q = spec
        fwd, inv = transforms[tname]
        sval, stest = fwd(GI_val), fwd(GI_test)
        rows = {}
        for p in GRID:
            if kind == "emp":
                th = float(np.quantile(sval, 1 - p))
                fit = {}
            else:
                th, xi, sig, u, n_u = gpd_threshold(sval, p, u_q)
                fit = {"xi": xi, "sigma": sig, "u": u, "n_u": n_u}
            rfpr = float(np.mean(stest >= th))
            tpr = float(np.mean(correct & (sp >= inv(np.array([th]))[0])))
            # bootstrap CI of realized FPR (impostor video clusters)
            boots = []
            idx = np.arange(len(stest))
            for _ in range(1000):
                ci = np.concatenate([ig[j] for j in
                                     rng.integers(0, len(iv), len(iv))])
                boots.append(float(np.mean(stest[ci] >= th)))
            lo, hi = np.percentile(boots, [2.5, 97.5])
            rows[str(p)] = {"threshold": float(th), "realized_fpr": rfpr,
                            "rfpr_ci": [float(lo), float(hi)], "tpr": tpr,
                            **fit}
        rep["methods"][mname] = rows
        print(f"\n{mname}:")
        for p in GRID:
            r = rows[str(p)]
            extra = (f" xi={r.get('xi', 0):+.2f}" if "xi" in r else "")
            print(f"  @{p:g}%: th={r['threshold']:.4f} -> test FPR_real="
                  f"{r['realized_fpr']:.4f} [{r['rfpr_ci'][0]:.4f},"
                  f"{r['rfpr_ci'][1]:.4f}] TPR={r['tpr']:.3f}{extra}")

    # best-method summary: |realized - nominal| distance
    print("\n=== |realized FPR - nominal| summary @0.5% ===")
    for mname in methods:
        r = rep["methods"][mname]["0.005"]
        print(f"  {mname:22s} |{r['realized_fpr'] - 0.005:.4f}|")

    out = PROJECT_ROOT / "runs" / args.out
    out.mkdir(parents=True, exist_ok=True)
    (out / "report.json").write_text(json.dumps(rep, indent=1, default=str),
                                     encoding="utf-8")
    print(f"[wrote] {out / 'report.json'}")


if __name__ == "__main__":
    main()
