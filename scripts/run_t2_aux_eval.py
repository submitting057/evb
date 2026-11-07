# -*- coding: utf-8 -*-
"""T2-A1 acceptance analysis for the aux-expert DF member (4th member).

Plain-language summary of what this script answers:

  1. Member check-up: is the aux-only model any good on its own?
     (closed-set top1 on probes; confidence distribution on impostors)
  2. Independence gate (pre-registered kill): an extra member only helps if
     it makes DIFFERENT mistakes than the existing members.  On val impostor
     sessions the argmax agreement between aux and the existing DF members
     should stay low (chance ~ 1/386).  Kill if agreement > 10%.
  3. Veto power on magnets: on the val-side top-0.5% magnet events of the
     current 3-member geo ensemble, is the aux member's confidence low?
     (low confidence there = it can pull those events down in a geo product)
  4. Final acceptance: 4-member geo (weights chosen by pooled LOVO in the
     combiner run, passed via --w4) vs the frozen 3-member geo baseline
     (.30/.35/.35) on TEST, judged by the paired video-cluster bootstrap
     (same procedure as run_strict_diag.py: both rules resample the same
     probe/impostor video clusters, each rule calibrates on its own
     resampled impostor scores; acceptance = dTPR CI excluding 0).
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
DEFAULT_TAGS = "dfs_branch_all_e100,dfs_branch_all_dt,dfs_branch_aux"


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
    ap.add_argument("--dataset", default="10s")
    ap.add_argument("--w4", required=True,
                    help="4 weights for lgbm,e100,dt,aux (from combiner LOVO pick)")
    ap.add_argument("--w3", default="0.30,0.35,0.35,0",
                    help="baseline 3-member weights (aux weight 0)")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--out", default="t2_aux_eval")
    ap.add_argument("--tags", default=DEFAULT_TAGS,
                    help="comma list of the 3 DF run tags in order e100,dt,aux")
    args = ap.parse_args()
    w4 = [float(x) for x in args.w4.split(",")]
    w3 = [float(x) for x in args.w3.split(",")]
    assert len(w4) == 4 and len(w3) == 4
    members = ["lgbm"] + [t for t in args.tags.split(",") if t]
    assert len(members) == 4, "--tags must name exactly 3 DF runs"
    short = {"lgbm": "lgbm", members[1]: "e100", members[2]: "dt",
             members[3]: "aux"}

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
    for t in members[1:]:
        mem[t] = {}
        for s in ("val", "test"):
            d = np.load(PROJECT_ROOT / "runs" / t / f"probs_{s}.npz")
            mem[t][s] = (norm(d["probe"]), norm(d["imp"]))
    y = {"val": np.load(PROJECT_ROOT / "runs" / members[1]
                        / "probs_val.npz")["probe_y"],
         "test": np.load(PROJECT_ROOT / "runs" / members[1]
                         / "probs_test.npz")["probe_y"]}

    rep = {"config": {"dataset": args.dataset, "w4": w4, "w3": w3,
                      "n_boot": args.n_boot}}

    # ---- 1) member check-up ------------------------------------------------
    print("=== 1) member check-up (strict protocol) ===")
    print(f"{'member':6s} {'top1_val':>9s} {'top1_test':>9s} "
          f"{'imp_q50':>8s} {'imp_q99':>8s} {'imp_max':>8s}   (imp = val)")
    rep["members"] = {}
    for n in members:
        tv = float((mem[n]["val"][0].argmax(1) == y["val"]).mean())
        tt = float((mem[n]["test"][0].argmax(1) == y["test"]).mean())
        smax = mem[n]["val"][1].max(1)
        row = {"top1_val": tv, "top1_test": tt,
               "imp_q50": float(np.quantile(smax, .5)),
               "imp_q99": float(np.quantile(smax, .99)),
               "imp_max": float(smax.max())}
        rep["members"][short[n]] = row
        print(f"{short[n]:6s} {tv:9.3f} {tt:9.3f} {row['imp_q50']:8.3f} "
              f"{row['imp_q99']:8.3f} {row['imp_max']:8.3f}")

    # ---- 2) independence gate ----------------------------------------------
    print("\n=== 2) independence gate: argmax agreement on VAL impostors ===")
    args_ = {short[n]: mem[n]["val"][1].argmax(1) for n in members}
    n_imp = len(args_["lgbm"])
    rep["agreement"] = {}
    pairs = [("e100", "dt"), ("aux", "e100"), ("aux", "dt"), ("aux", "lgbm"),
             ("lgbm", "e100")]
    for a, b in pairs:
        ag = float((args_[a] == args_[b]).mean())
        rep["agreement"][f"{a}_vs_{b}"] = ag
        print(f"  {a:5s} vs {b:5s}: {ag:6.3f}  (chance ~ {1/386:.4f})")
    gate = max(rep["agreement"]["aux_vs_e100"], rep["agreement"]["aux_vs_dt"])
    print(f"  GATE: aux-vs-DF agreement max = {gate:.3f} -> "
          f"{'KILL (>10%)' if gate > 0.10 else 'pass'}")
    rep["gate_independence_pass"] = bool(gate <= 0.10)
    # probe-side sanity: on correct val probes members SHOULD agree a lot
    cor = mem["lgbm"]["val"][0].argmax(1) == y["val"]
    argp = {short[n]: mem[n]["val"][0].argmax(1) for n in members}
    for a, b in [("aux", "e100"), ("aux", "dt")]:
        ag = float((argp[a][cor] == argp[b][cor]).mean())
        rep["agreement"][f"{a}_vs_{b}_correct_probes"] = ag
        print(f"  (sanity) {a} vs {b} on CORRECT val probes: {ag:.3f}")

    # ---- 3) veto power on val magnets --------------------------------------
    print("\n=== 3) veto power on val magnets (top-0.5% of 3-member geo) ===")
    G3v = geo(w3, [mem[n]["val"][0] for n in members])
    I3v = geo(w3, [mem[n]["val"][1] for n in members])
    si3 = I3v.max(1)
    th = _threshold_for_fpr(si3, 0.005)
    mag_rows = np.where(si3 >= th)[0]
    mag_vids = vid["val_imp"][mag_rows]
    rep["magnets"] = {"n_events": int(len(mag_rows)),
                      "n_videos": int(len(set(mag_vids.tolist())))}
    print(f"  val top-0.5% events: {len(mag_rows)} from "
          f"{len(set(mag_vids.tolist()))} videos")
    rep["magnets"]["member_conf"] = {}
    for n in members:
        smax = mem[n]["val"][1].max(1)
        row = {"mag_median": float(np.median(smax[mag_rows])),
               "mag_mean": float(smax[mag_rows].mean()),
               "probe_correct_median": float(np.median(
                   mem[n]["val"][0].max(1)[
                       mem[n]["val"][0].argmax(1) == y["val"]]))}
        rep["magnets"]["member_conf"][short[n]] = row
        print(f"  {short[n]:5s} maxprob on magnets: median "
              f"{row['mag_median']:.3f} mean {row['mag_mean']:.3f} "
              f"(correct-probe median {row['probe_correct_median']:.3f})")

    # ---- 4) final acceptance: paired cluster bootstrap on TEST -------------
    print(f"\n=== 4) TEST acceptance: geo4{tuple(round(x,2) for x in w4)} "
          f"vs geo3{tuple(round(x,2) for x in w3)} "
          f"({args.n_boot} paired cluster reps) ===")
    rules = {"geo4": w4, "geo3": w3}
    P, SP, CO, SI = {}, {}, {}, {}
    for rn, w in rules.items():
        P[rn] = geo(w, [mem[n]["test"][0] for n in members])
        SP[rn] = P[rn].max(1)
        CO[rn] = P[rn].argmax(1) == y["test"]
        SI[rn] = geo(w, [mem[n]["test"][1] for n in members]).max(1)

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

    # deployment caliber (val-side threshold), for the record
    dep = {}
    for rn, w in rules.items():
        Iv = geo(w, [mem[n]["val"][1] for n in members]).max(1)
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
            for rn in rules:
                th = _threshold_for_fpr(SI[rn][ci], fr)
                boot[rn][str(fr)].append(
                    float(np.mean(CO[rn][cpi] & (SP[rn][cpi] >= th))))
            th4 = _threshold_for_fpr(SI["geo4"][ci], fr)
            th3 = _threshold_for_fpr(SI["geo3"][ci], fr)
            dtp[str(fr)].append(
                float(np.mean(CO["geo4"][cpi] & (SP["geo4"][cpi] >= th4))
                      - np.mean(CO["geo3"][cpi] & (SP["geo3"][cpi] >= th3))))
    rep["bootstrap_cluster"] = {
        rn: {str(fr): [float(np.percentile(boot[rn][str(fr)], 2.5)),
                       float(np.percentile(boot[rn][str(fr)], 97.5))]
             for fr in GRID} for rn in rules}
    rep["dTPR_cluster"] = {str(fr): [float(np.percentile(dtp[str(fr)], 2.5)),
                                     float(np.percentile(dtp[str(fr)], 97.5))]
                           for fr in GRID}
    for rn in rules:
        print(f"  {rn} cluster CI @.5% "
              f"[{rep['bootstrap_cluster'][rn]['0.005'][0]:.3f},"
              f"{rep['bootstrap_cluster'][rn]['0.005'][1]:.3f}]")
    verdict = {}
    for fr in GRID:
        lo, hi = rep["dTPR_cluster"][str(fr)]
        exc = bool(lo > 0 or hi < 0)
        verdict[str(fr)] = {"ci": [lo, hi], "excludes_0": exc}
        print(f"  paired dTPR(geo4-geo3) @{fr:g}% CI [{lo:+.3f},{hi:+.3f}] "
              f"{'EXCLUDES 0' if exc else 'includes 0'}")
    rep["verdict"] = verdict

    out = PROJECT_ROOT / "runs" / args.out
    out.mkdir(parents=True, exist_ok=True)
    (out / "report.json").write_text(json.dumps(rep, indent=1, default=str),
                                     encoding="utf-8")
    print(f"[wrote] {out / 'report.json'}")


if __name__ == "__main__":
    main()
