# -*- coding: utf-8 -*-
"""One-sided improvement probability: chsplit geo4 vs frozen nogro baseline.

10000-rep video-cluster paired bootstrap, same protocol as run_paired_boot.py,
reporting P(dTPR>0) / median / q05 / q95 per FPR tier.  Written for the
2026-11-07 confirmation ruling (see findings.md 3.6c): the paired CI contains
0 at every tier, so the ruling relied on point estimates plus this one-sided
probability (.84/.79/.79) instead of the CI-excludes-0 gate.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import PROJECT_ROOT, RunConfig
from src.data import load_splits, session_pool
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


rcfg = RunConfig(dataset="10s_nogro", protocol="strict")
spl = load_splits(rcfg.splits_path)
pools = dict(tp=session_pool(spl, "known", "test", gallery=False),
             ti=session_pool(spl, "unknown", "test"))
vid = {k: np.array([spl["sessions"][s]["video"] for s in v]) for k, v in pools.items()}


def load(tags, w):
    z = np.load(PROJECT_ROOT / "runs" / "cache" / "lgbm_probs_10s_nogro_strict.npz")
    mp = [norm(z["test_probe"])]
    mi = [norm(z["test_imp"])]
    for t in tags:
        d = np.load(PROJECT_ROOT / "runs" / t / "probs_test.npz")
        mp.append(norm(d["probe"]))
        mi.append(norm(d["imp"]))
    y = np.load(PROJECT_ROOT / "runs" / tags[0] / "probs_test.npz")["probe_y"]
    P = geo(w, mp)
    return P.max(1), P.argmax(1) == y, geo(w, mi).max(1)


SPA, COA, SIA = load(["ngL_dfs_branch_vid", "ngL_dfs_branch_vid_dt",
                      "ngL_dfs_branch_aux3"], [.15, .15, .3, .4])
SPB, COB, SIB = load(["ngL_dfs_branch_all_e100", "ngL_dfs_branch_all_dt",
                      "ngL_dfs_branch_aux"], [.15, .55, 0, .3])

pv = sorted(set(vid["tp"].tolist()))
iv = sorted(set(vid["ti"].tolist()))
pg = [np.where(vid["tp"] == v)[0] for v in pv]
ig = [np.where(vid["ti"] == v)[0] for v in iv]
rng = np.random.default_rng(0)
NB = 10000
d = {f: [] for f in (.005, .01, .05)}
for _ in range(NB):
    ci = np.concatenate([ig[j] for j in rng.integers(0, len(iv), len(iv))])
    cpi = np.concatenate([pg[j] for j in rng.integers(0, len(pv), len(pv))])
    for f in d:
        tpr = {}
        for nm, (SP, CO, SI) in (("A", (SPA, COA, SIA)), ("B", (SPB, COB, SIB))):
            th = _threshold_for_fpr(SI[ci], f)
            tpr[nm] = float(np.mean(CO[cpi] & (SP[cpi] >= th)))
        d[f].append(tpr["A"] - tpr["B"])
print(f"{NB} boots, dTPR(S1 chsplit - baseline):")
for f, v in d.items():
    v = np.array(v)
    print(f"  @{f:g}%  median {np.median(v):+.4f}  mean {v.mean():+.4f}  "
          f"P(d>0)={np.mean(v > 0):.3f}  q05 {np.quantile(v, .05):+.4f}  "
          f"q95 {np.quantile(v, .95):+.4f}")
