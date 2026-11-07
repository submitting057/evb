# -*- coding: utf-8 -*-
"""Paired video-cluster bootstrap comparison of two geo-combination rules.

Generalizes the acceptance step of run_t2_aux_eval.py: rule A and rule B may
use DIFFERENT member sets (e.g. baseline tags vs channel-length-adapted tags),
each with its own weight vector.  Both rules resample the same probe/impostor
video clusters and calibrate on their own resampled impostor scores;
acceptance = dTPR(A-B) CI excluding 0 at any FPR tier.

Usage:
  python scripts/run_paired_boot.py --dataset 10s_nogro \
      --a-tags ng_dfs_branch_all_e100,ng_dfs_branch_all_dt,ng_dfs_branch_aux \
      --a-w 0.35,0.10,0.20,0.35 \
      --b-tags ngL_dfs_branch_all_e100,ngL_dfs_branch_all_dt,ngL_dfs_branch_aux \
      --b-w 0.35,0.10,0.20,0.35 --out ng_paired_boot
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import PROJECT_ROOT, RunConfig
from src.data import load_splits, session_pool
from src.evaluate import _threshold_for_fpr

GRID = (0.005, 0.01, 0.05)


def norm(P):
    return P / (P.sum(1, keepdims=True) + 1e-12)


def geo(ws, mats):
    pairs = [(P, w) for P, w in zip(mats, ws) if w > 0]
    s = sum(w for _, w in pairs)
    L = sum(w / s * np.log(np.clip(P, 1e-12, 1)) for P, w in pairs)
    L -= L.max(1, keepdims=True)
    E = np.exp(L)
    return E / E.sum(1, keepdims=True)


def load_members(dataset, tags, lgbm_suffix=""):
    """tags: DF run tags (no lgbm). Returns {"lgbm": ..., tag: ...} prob pairs."""
    mem = {}
    z = np.load(PROJECT_ROOT / "runs" / "cache"
                / f"lgbm_probs_{dataset}_strict{lgbm_suffix}.npz")
    mem["lgbm"] = {s: (norm(z[f"{s}_probe"]), norm(z[f"{s}_imp"]))
                   for s in ("val", "test")}
    for t in tags:
        mem[t] = {}
        for s in ("val", "test"):
            d = np.load(PROJECT_ROOT / "runs" / t / f"probs_{s}.npz")
            mem[t][s] = (norm(d["probe"]), norm(d["imp"]))
    return mem


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="10s")
    ap.add_argument("--a-tags", required=True, help="3 DF tags of rule A")
    ap.add_argument("--a-w", required=True,
                    help="4 weights lgbm,<a-tags...> of rule A")
    ap.add_argument("--b-tags", required=True, help="3 DF tags of rule B")
    ap.add_argument("--b-w", required=True, help="4 weights of rule B")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--out", default="paired_boot")
    ap.add_argument("--a-lgbm-suffix", default="",
                    help="rule A lgbm probs cache suffix (e.g. _fast200)")
    ap.add_argument("--b-lgbm-suffix", default="",
                    help="rule B lgbm probs cache suffix")
    args = ap.parse_args()
    rules = {}
    for rn, tstr, wstr, lsuf in (("A", args.a_tags, args.a_w, args.a_lgbm_suffix),
                                 ("B", args.b_tags, args.b_w, args.b_lgbm_suffix)):
        tags = [t for t in tstr.split(",") if t]
        w = [float(x) for x in wstr.split(",")]
        assert len(w) == len(tags) + 1, f"--{rn.lower()}-w must have len(tags)+1 entries"
        rules[rn] = {"members": ["lgbm"] + tags, "w": w, "lgbm_suffix": lsuf}

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

    all_tags = sorted({t for r in rules.values() for t in r["members"][1:]})
    mem = {}
    for lsuf in {r["lgbm_suffix"] for r in rules.values()}:
        sub = load_members(args.dataset, all_tags, lsuf)
        key = f"lgbm{lsuf}" if lsuf else "lgbm"
        mem[key] = sub["lgbm"]
        for t in all_tags:
            mem.setdefault(t, sub[t])
    def mname(n, lsuf):
        if n == "lgbm":
            return f"lgbm{lsuf}" if lsuf else "lgbm"
        return n
    for rn, r in rules.items():
        r["members"] = [mname(n, r["lgbm_suffix"]) for n in r["members"]]
    ref_tag = rules["A"]["members"][1]
    y = {"val": np.load(PROJECT_ROOT / "runs" / ref_tag / "probs_val.npz")["probe_y"],
         "test": np.load(PROJECT_ROOT / "runs" / ref_tag / "probs_test.npz")["probe_y"]}
    for rn, r in rules.items():
        t = r["members"][1]
        assert np.array_equal(y["val"], np.load(
            PROJECT_ROOT / "runs" / t / "probs_val.npz")["probe_y"]), \
            f"rule {rn} member {t} has a different val probe order"

    rep = {"config": {"dataset": args.dataset,
                      "A": {"members": rules["A"]["members"], "w": rules["A"]["w"]},
                      "B": {"members": rules["B"]["members"], "w": rules["B"]["w"]},
                      "n_boot": args.n_boot}}

    P, SP, CO, SI = {}, {}, {}, {}
    for rn, r in rules.items():
        P[rn] = geo(r["w"], [mem[n]["test"][0] for n in r["members"]])
        SP[rn] = P[rn].max(1)
        CO[rn] = P[rn].argmax(1) == y["test"]
        SI[rn] = geo(r["w"], [mem[n]["test"][1] for n in r["members"]]).max(1)

    det = {}
    for rn in rules:
        row = {"top1": float(CO[rn].mean()), "fpr": {}}
        for fr in GRID:
            t = _threshold_for_fpr(SI[rn], fr)
            row["fpr"][str(fr)] = float(np.mean(CO[rn] & (SP[rn] >= t)))
        det[rn] = row
        print(f"  {rn} top1={row['top1']:.3f} " + " ".join(
            f"@{fr:g}%={row['fpr'][str(fr)]:.3f}" for fr in GRID))
    rep["test_oracle"] = det

    dep = {}
    for rn, r in rules.items():
        Iv = geo(r["w"], [mem[n]["val"][1] for n in r["members"]]).max(1)
        row = {}
        for fr in GRID:
            th = _threshold_for_fpr(Iv, fr)
            row[str(fr)] = {"realized_fpr": float(np.mean(SI[rn] >= th)),
                            "tpr": float(np.mean(CO[rn] & (SP[rn] >= th)))}
        dep[rn] = row
    rep["test_deployment"] = dep
    for rn in rules:
        print(f"  {rn} deploy: " + " ".join(
            f"@{fr:g}% FPR_real={dep[rn][str(fr)]['realized_fpr']:.4f}"
            f" TPR={dep[rn][str(fr)]['tpr']:.3f}" for fr in GRID))

    pv = sorted(set(vid["test_probe"].tolist()))
    iv = sorted(set(vid["test_imp"].tolist()))
    pg = [np.where(vid["test_probe"] == v)[0] for v in pv]
    ig = [np.where(vid["test_imp"] == v)[0] for v in iv]
    rng = np.random.default_rng(0)
    boot = {rn: {str(fr): [] for fr in GRID} for rn in rules}
    dtp = {str(fr): [] for fr in GRID}
    for _ in range(args.n_boot):
        ci = np.concatenate([ig[j] for j in rng.integers(0, len(iv), len(iv))])
        cpi = np.concatenate([pg[j] for j in rng.integers(0, len(pv), len(pv))])
        for fr in GRID:
            tpr = {}
            for rn in rules:
                th = _threshold_for_fpr(SI[rn][ci], fr)
                tpr[rn] = float(np.mean(CO[rn][cpi] & (SP[rn][cpi] >= th)))
                boot[rn][str(fr)].append(tpr[rn])
            dtp[str(fr)].append(tpr["A"] - tpr["B"])
    rep["bootstrap_cluster"] = {
        rn: {str(fr): [float(np.percentile(boot[rn][str(fr)], 2.5)),
                       float(np.percentile(boot[rn][str(fr)], 97.5))]
             for fr in GRID} for rn in rules}
    rep["dTPR_cluster"] = {str(fr): [float(np.percentile(dtp[str(fr)], 2.5)),
                                     float(np.percentile(dtp[str(fr)], 97.5))]
                           for fr in GRID}
    verdict = {}
    for fr in GRID:
        lo, hi = rep["dTPR_cluster"][str(fr)]
        exc = bool(lo > 0 or hi < 0)
        verdict[str(fr)] = {"ci": [lo, hi], "excludes_0": exc}
        print(f"  paired dTPR(A-B) @{fr:g}% CI [{lo:+.3f},{hi:+.3f}] "
              f"{'EXCLUDES 0' if exc else 'includes 0'}")
    rep["verdict"] = verdict

    out = PROJECT_ROOT / "runs" / args.out
    out.mkdir(parents=True, exist_ok=True)
    (out / "report.json").write_text(json.dumps(rep, indent=1, default=str),
                                     encoding="utf-8")
    print(f"[wrote] {out / 'report.json'}")


if __name__ == "__main__":
    main()
