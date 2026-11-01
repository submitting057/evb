# -*- coding: utf-8 -*-
"""Multi-seed robustness of the T2-A1 aux result (pre-registered in T2).

Plain-language summary:

  The accepted claim is: "adding the aux (auxiliary-channels-only) member
  to the 3-member geo ensemble improves the open-set TPR" -- but all
  evidence so far comes from models trained with seed 42 on one fixed
  split.  To check it is not a lucky seed, the three DF members are
  retrained with model seeds 43/44/45 on the SAME strict split (lgbm and
  the split stay fixed; only DF training randomness varies), and for each
  seed the same question is asked fresh:

    * 4-member combiner (lgbm,e100,dt,aux) pooled LOVO -> best geo weights
    * 3-member combiner (lgbm,e100,dt)     pooled LOVO -> best geo weights
    * paired video-cluster bootstrap dTPR(geo4 - geo3) on TEST

  The per-seed baseline is that seed's OWN best 3-member geo (not the
  frozen seed-42 weights), so the comparison is never handicapped.
  Robustness verdict: the direction should be positive in every seed and
  the CI should exclude 0 in most seeds.

Usage:
  python scripts/run_ms_eval.py --seed 43        # after that seed's
      two combiner runs exist (combo_10s_strict_ms43 and ..._ms43_3m)
  python scripts/run_ms_eval.py --agg            # aggregate seeds 42-45
      (seed 42 numbers come from runs/t2_aux_eval/report.json)
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
    w = np.asarray(ws, float)
    w = w / w.sum()
    L = sum(wi * np.log(np.clip(P, 1e-12, 1)) for wi, P in zip(w, mats))
    L -= L.max(1, keepdims=True)
    E = np.exp(L)
    return E / E.sum(1, keepdims=True)


def load_members(seed, dataset="10s"):
    """lgbm (fixed cache) + per-seed DF dirs; returns (names, mem, y)."""
    names = ["lgbm", f"dfs{seed}_branch_all_e100", f"dfs{seed}_branch_all_dt",
             f"dfs{seed}_branch_aux"]
    mem = {}
    z = np.load(PROJECT_ROOT / "runs" / "cache"
                / f"lgbm_probs_{dataset}_strict.npz")
    mem["lgbm"] = {s: (norm(z[f"{s}_probe"]), norm(z[f"{s}_imp"]))
                   for s in ("val", "test")}
    for t in names[1:]:
        mem[t] = {}
        for s in ("val", "test"):
            d = np.load(PROJECT_ROOT / "runs" / t / f"probs_{s}.npz")
            mem[t][s] = (norm(d["probe"]), norm(d["imp"]))
    y = {"val": np.load(PROJECT_ROOT / "runs" / names[1]
                        / "probs_val.npz")["probe_y"],
         "test": np.load(PROJECT_ROOT / "runs" / names[1]
                         / "probs_test.npz")["probe_y"]}
    return names, mem, y


def best_rule_weights(rep_path):
    """Top LOVO row among plain prob rules (geo/arith, temp=False).

    The per-seed baseline is that seed's OWN best plain rule -- on seed 45
    the 3-member LOVO actually prefers arith over geo, and comparing
    against the true best alternative is the honest test.
    """
    rep = json.load(open(PROJECT_ROOT / rep_path, encoding="utf-8"))
    rows = [r for r in rep["lovo"]["top15"]
            if r["rule"] in ("geo", "arith") and not r["temp"]]
    r = rows[0]
    return r["rule"], [float(x) for x in r["w"]], r["lovo"], rows


def comb(rule, ws, mats):
    w = np.asarray(ws, float)
    w = w / w.sum()
    if rule == "arith":
        return sum(wi * P for wi, P in zip(w, mats))
    return geo(w, mats)


def eval_seed(seed, dataset="10s", n_boot=2000):
    rcfg = RunConfig(dataset=dataset, protocol="strict")
    splits = load_splits(rcfg.splits_path)
    pools = dict(
        test_probe=session_pool(splits, "known", "test", gallery=False),
        test_imp=session_pool(splits, "unknown", "test"),
    )
    vid = {k: np.array([splits["sessions"][s]["video"] for s in v])
           for k, v in pools.items()}
    names, mem, y = load_members(seed, dataset)

    rule4, w4, lovo4, rows4 = best_rule_weights(
        f"runs/combo_10s_strict_ms{seed}/report.json")
    rule3, w3_3, lovo3, rows3 = best_rule_weights(
        f"runs/combo_10s_strict_ms{seed}_3m/report.json")
    # embed the 3-member weights into the 4-member space (aux weight 0)
    w3 = w3_3[:2] + [w3_3[2], 0.0]

    G4 = comb(rule4, w4, [mem[n]["test"][0] for n in names])
    I4 = comb(rule4, w4, [mem[n]["test"][1] for n in names])
    G3 = comb(rule3, w3, [mem[n]["test"][0] for n in names])
    I3 = comb(rule3, w3, [mem[n]["test"][1] for n in names])
    SP = {"geo4": G4.max(1), "geo3": G3.max(1)}
    CO = {"geo4": G4.argmax(1) == y["test"], "geo3": G3.argmax(1) == y["test"]}
    SI = {"geo4": I4.max(1), "geo3": I3.max(1)}

    det = {}
    for rn in ("geo4", "geo3"):
        row = {"top1": float(CO[rn].mean())}
        for fr in GRID:
            th = _threshold_for_fpr(SI[rn], fr)
            row[fr] = float(np.mean(CO[rn] & (SP[rn] >= th)))
        det[rn] = row

    # member top1 per seed (train quality record)
    top1s = {}
    for n, key in zip(names, ["lgbm", "e100", "dt", "aux"]):
        top1s[key] = float((mem[n]["test"][0].argmax(1) == y["test"]).mean())

    pv = sorted(set(vid["test_probe"].tolist()))
    iv = sorted(set(vid["test_imp"].tolist()))
    pg = [np.where(vid["test_probe"] == v)[0] for v in pv]
    ig = [np.where(vid["test_imp"] == v)[0] for v in iv]
    rng = np.random.default_rng(0)
    dtp = {str(fr): [] for fr in GRID}
    for _ in range(n_boot):
        ci = np.concatenate([ig[j] for j in rng.integers(0, len(iv), len(iv))])
        cpi = np.concatenate([pg[j] for j in rng.integers(0, len(pv), len(pv))])
        for fr in GRID:
            th4 = _threshold_for_fpr(SI["geo4"][ci], fr)
            th3 = _threshold_for_fpr(SI["geo3"][ci], fr)
            dtp[str(fr)].append(float(
                np.mean(CO["geo4"][cpi] & (SP["geo4"][cpi] >= th4))
                - np.mean(CO["geo3"][cpi] & (SP["geo3"][cpi] >= th3))))
    ci_out = {}
    for fr in GRID:
        lo, hi = np.percentile(dtp[str(fr)], [2.5, 97.5])
        ci_out[str(fr)] = {"ci": [float(lo), float(hi)],
                           "excludes_0": bool(lo > 0 or hi < 0)}

    out = {"seed": seed, "rule4": rule4, "w4": w4, "rule3": rule3, "w3": w3,
           "lovo4": lovo4, "lovo3": lovo3,
           "member_test_top1": top1s, "test": det, "dTPR_cluster": ci_out}
    print(f"seed {seed}: 4m {rule4}{w4} | 3m {rule3}{w3}")
    print(f"  LOVO 4m {lovo4['0.005']:.3f} vs 3m {lovo3['0.005']:.3f}")
    for rn in ("geo4", "geo3"):
        print(f"  {rn} test " + " ".join(f"@{fr:g}%={det[rn][fr]:.3f}"
                                         for fr in GRID)
              + f" top1={det[rn]['top1']:.3f}")
    for fr in GRID:
        c = ci_out[str(fr)]
        print(f"  dTPR @{fr:g}% CI [{c['ci'][0]:+.3f},{c['ci'][1]:+.3f}] "
              f"{'EXCLUDES 0' if c['excludes_0'] else 'includes 0'}")
    d = PROJECT_ROOT / "runs" / f"ms{seed}_eval"
    d.mkdir(parents=True, exist_ok=True)
    (d / "report.json").write_text(json.dumps(out, indent=1, default=str),
                                   encoding="utf-8")
    print(f"[wrote] {d / 'report.json'}")
    return out


def aggregate():
    """Seed 42 from the accepted T2-A1 files; 43-45 from ms eval reports."""
    rows = []
    t2 = json.load(open(PROJECT_ROOT / "runs" / "t2_aux_eval" / "report.json",
                        encoding="utf-8"))
    def _tpr42(v):
        out = {"top1": v["top1"]}
        for fr in GRID:
            x = v["fpr"][str(fr)]
            out[str(fr)] = x["tpr"] if isinstance(x, dict) else x
        return out

    rows.append({
        "seed": 42, "w4": t2["config"]["w4"], "w3": t2["config"]["w3"],
        "lovo4": {"0.005": 0.773}, "lovo3": {"0.005": 0.696},
        "test": {k: _tpr42(v) for k, v in t2["test_oracle"].items()},
        "dTPR_cluster": t2["verdict"], "member_test_top1": {
            k: v["top1_test"] for k, v in t2["members"].items()}})
    for seed in (43, 44, 45):
        p = PROJECT_ROOT / "runs" / f"ms{seed}_eval" / "report.json"
        if p.exists():
            rows.append(json.load(open(p, encoding="utf-8")))

    print("\n=== multi-seed aggregate (geo4 vs that seed's best geo3) ===")
    print(f"{'seed':>5} {'LOVO4':>6} {'LOVO3':>6} | "
          f"{'t4@.5%':>7} {'t3@.5%':>7} | dTPR@.5% CI | @1% CI | @5% CI")
    for r in rows:
        l4 = r.get("lovo4", {}).get("0.005", float("nan"))
        l3 = r.get("lovo3", {}).get("0.005", float("nan"))
        d = r["dTPR_cluster"]
        c5 = d.get("0.005", d.get(0.005))
        c1 = d.get("0.01", d.get(0.01))
        c05 = d.get("0.05", d.get(0.05))
        t4 = r["test"]["geo4"].get("0.005", r["test"]["geo4"].get(0.005))
        t3 = r["test"]["geo3"].get("0.005", r["test"]["geo3"].get(0.005))
        print(f"{r['seed']:>5} {l4:6.3f} {l3:6.3f} | {t4:7.3f} {t3:7.3f} | "
              f"[{c5['ci'][0]:+.3f},{c5['ci'][1]:+.3f}]"
              f"{'*' if c5['excludes_0'] else ' '} "
              f"[{c1['ci'][0]:+.3f},{c1['ci'][1]:+.3f}]"
              f"{'*' if c1['excludes_0'] else ' '} "
              f"[{c05['ci'][0]:+.3f},{c05['ci'][1]:+.3f}]"
              f"{'*' if c05['excludes_0'] else ' '}")
        w4 = r.get("w4")
        if w4:
            print(f"      w4={w4}")
    n = len(rows)
    ex5 = sum(1 for r in rows if r["dTPR_cluster"]
              .get("0.005", r["dTPR_cluster"].get(0.005, {}))["excludes_0"])
    ex1 = sum(1 for r in rows if r["dTPR_cluster"]
              .get("0.01", r["dTPR_cluster"].get(0.01, {}))["excludes_0"])
    pos5 = sum(1 for r in rows if r["dTPR_cluster"]
               .get("0.005", r["dTPR_cluster"].get(0.005, {}))["ci"][0] > 0
               or r["dTPR_cluster"]
               .get("0.005", r["dTPR_cluster"].get(0.005, {}))["ci"][1] > 0)
    print(f"\n@0.5%: CI excludes 0 in {ex5}/{n} seeds; "
          f"@1%: {ex1}/{n}; positive point direction in {pos5}/{n}")
    rep = {"seeds": rows, "summary": {"n_seeds": n, "excl0_05pct": ex5,
                                      "excl0_1pct": ex1}}
    d = PROJECT_ROOT / "runs" / "ms_agg"
    d.mkdir(parents=True, exist_ok=True)
    (d / "report.json").write_text(json.dumps(rep, indent=1, default=str),
                                   encoding="utf-8")
    print("[wrote] runs/ms_agg/report.json")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--agg", action="store_true")
    ap.add_argument("--n-boot", type=int, default=2000)
    args = ap.parse_args()
    if args.seed is not None:
        eval_seed(args.seed, n_boot=args.n_boot)
    if args.agg:
        aggregate()


if __name__ == "__main__":
    main()
