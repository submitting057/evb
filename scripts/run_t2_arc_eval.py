# -*- coding: utf-8 -*-
"""T2-A2 acceptance analysis for the ArcFace member (5th member).

Plain-language summary:

  1. Member check-up: closed-set top1 + impostor confidence distribution.
  2. Independence gate (kill): argmax agreement of arc with the existing
     DF members on VAL impostors must stay <= 10%.
  3. Joint-overconfidence audit (pre-registered stop gate for adding
     members): the fraction of val impostor sessions where EVERY member of
     the system is confident (maxprob > .5) must DROP by at least 30% when
     the new member joins (compared with the system without it).
  4. Final acceptance: 5-member geo with weights chosen by the pooled-LOVO
     combiner run (passed via --w5) vs the frozen geo4 baseline, judged by
     the paired video-cluster bootstrap on TEST (dTPR CI excluding 0).
  Anchor: geo with w=(.35,.10/.20/.35, 0) must reproduce the geo4 numbers.
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
MEMBERS = ["lgbm", "dfs_branch_all_e100", "dfs_branch_all_dt",
           "dfs_branch_aux", "dfs_branch_arc"]
SHORT = ["lgbm", "e100", "dt", "aux", "arc"]
W4 = [0.35, 0.10, 0.20, 0.35, 0.0]


def norm(P):
    return P / (P.sum(1, keepdims=True) + 1e-12)


def geo(ws, mats):
    w = np.asarray(ws, float)
    w = w / w.sum()
    L = sum(wi * np.log(np.clip(P, 1e-12, 1)) for wi, P in zip(w, mats))
    L -= L.max(1, keepdims=True)
    E = np.exp(L)
    return E / E.sum(1, keepdims=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="10s")
    ap.add_argument("--w5", default=None,
                    help="5 weights from the LOVO combiner pick (skip "
                         "section 4 if omitted)")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--out", default="t2_arc_eval")
    args = ap.parse_args()
    w5 = [float(x) for x in args.w5.split(",")] if args.w5 else None

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
    y = {"val": np.load(PROJECT_ROOT / "runs" / MEMBERS[1]
                        / "probs_val.npz")["probe_y"],
         "test": np.load(PROJECT_ROOT / "runs" / MEMBERS[1]
                         / "probs_test.npz")["probe_y"]}

    rep = {}

    # ---- 1) member check-up ------------------------------------------------
    print("=== 1) member check-up ===")
    print(f"{'member':6s} {'top1_val':>9s} {'top1_test':>9s} {'imp_q50':>8s} "
          f"{'imp_q99':>8s} {'imp_max':>8s}")
    rep["members"] = {}
    for n, s in zip(MEMBERS, SHORT):
        tv = float((mem[n]["val"][0].argmax(1) == y["val"]).mean())
        tt = float((mem[n]["test"][0].argmax(1) == y["test"]).mean())
        smax = mem[n]["val"][1].max(1)
        rep["members"][s] = {"top1_val": tv, "top1_test": tt,
                             "imp_q50": float(np.quantile(smax, .5)),
                             "imp_q99": float(np.quantile(smax, .99)),
                             "imp_max": float(smax.max())}
        print(f"{s:6s} {tv:9.3f} {tt:9.3f} {rep['members'][s]['imp_q50']:8.3f}"
              f" {rep['members'][s]['imp_q99']:8.3f}"
              f" {rep['members'][s]['imp_max']:8.3f}")

    # ---- 2) independence gate ----------------------------------------------
    print("\n=== 2) independence gate: argmax agreement on VAL impostors ===")
    ag_ = {s: mem[n]["val"][1].argmax(1) for n, s in zip(MEMBERS, SHORT)}
    rep["agreement"] = {}
    pairs = [("e100", "dt"), ("aux", "e100"), ("arc", "e100"),
             ("arc", "dt"), ("arc", "aux"), ("arc", "lgbm")]
    for a, b in pairs:
        ag = float((ag_[a] == ag_[b]).mean())
        rep["agreement"][f"{a}_vs_{b}"] = ag
        print(f"  {a:5s} vs {b:5s}: {ag:6.3f}")
    gate = max(rep["agreement"][k] for k in
               ("arc_vs_e100", "arc_vs_dt", "arc_vs_aux"))
    rep["gate_independence_pass"] = bool(gate <= 0.10)
    print(f"  GATE arc-vs-DF max = {gate:.3f} -> "
          f"{'KILL (>10%)' if gate > 0.10 else 'pass'}")

    # ---- 3) joint-overconfidence audit -------------------------------------
    print("\n=== 3) joint-overconfidence audit (val impostors, all "
          "maxprob>.5) ===")
    conf = {s: mem[n]["val"][1].max(1) > 0.5 for n, s in zip(MEMBERS, SHORT)}
    sys3 = conf["lgbm"] & conf["e100"] & conf["dt"]
    sys4 = sys3 & conf["aux"]
    sys5 = sys4 & conf["arc"]
    r3, r4, r5 = sys3.mean(), sys4.mean(), sys5.mean()
    rep["joint_overconf"] = {"sys3": float(r3), "sys4": float(r4),
                             "sys5": float(r5)}
    print(f"  3-member (lgbm,e100,dt):   {r3:.4f}")
    print(f"  4-member (+aux):           {r4:.4f}  "
          f"(drop {1 - r4 / r3:.0%} vs 3-member)")
    print(f"  5-member (+arc):           {r5:.4f}  "
          f"(drop {1 - r5 / r4:.0%} vs 4-member)")
    rep["gate_joint_drop_pass"] = bool(1 - r5 / r4 >= 0.30)
    print(f"  GATE (arc adds >=30% drop vs 4-member): "
          f"{'pass' if rep['gate_joint_drop_pass'] else 'FAIL'}")

    if w5 is None:
        print("\n(no --w5 given: stopping after the gates)")
        out = PROJECT_ROOT / "runs" / args.out
        out.mkdir(parents=True, exist_ok=True)
        (out / "report.json").write_text(json.dumps(rep, indent=1,
                                                    default=str),
                                         encoding="utf-8")
        print(f"[wrote] {out / 'report.json'}")
        return

    # ---- 4) final acceptance ------------------------------------------------
    print(f"\n=== 4) TEST acceptance: geo5{tuple(w5)} vs "
          f"geo4{tuple(W4)} ({args.n_boot} paired cluster reps) ===")
    G0 = geo(W4, [mem[n]["test"][0] for n in MEMBERS])
    I0 = geo(W4, [mem[n]["test"][1] for n in MEMBERS])
    det0 = G0.max(1), G0.argmax(1) == y["test"], I0.max(1)
    a = (round(float(np.mean(det0[1])), 3),
         round(float(np.mean(det0[1] & (det0[0] >= _threshold_for_fpr(
             det0[2], GRID[0])))), 3))
    print(f"  anchor geo4: top1={a[0]:.3f} @.5%={a[1]:.3f} "
          f"(expect .975/.654)")
    assert a == (0.975, 0.654), "geo4 anchor mismatch -- abort"

    G5 = geo(w5, [mem[n]["test"][0] for n in MEMBERS])
    I5 = geo(w5, [mem[n]["test"][1] for n in MEMBERS])
    SP5, CO5, SI5 = G5.max(1), G5.argmax(1) == y["test"], I5.max(1)
    det = {}
    for rn, (sp, co, si) in (("geo5", (SP5, CO5, SI5)),
                             ("geo4", det0)):
        row = {"top1": float(co.mean())}
        for fr in GRID:
            th = _threshold_for_fpr(si, fr)
            row[fr] = float(np.mean(co & (sp >= th)))
        det[rn] = row
        print(f"  {rn} top1={row['top1']:.3f} " + " ".join(
            f"@{fr:g}%={row[fr]:.3f}" for fr in GRID))
    rep["test_oracle"] = det

    pv = sorted(set(vid["test_probe"].tolist()))
    iv = sorted(set(vid["test_imp"].tolist()))
    pg = [np.where(vid["test_probe"] == v)[0] for v in pv]
    ig = [np.where(vid["test_imp"] == v)[0] for v in iv]
    rng = np.random.default_rng(0)
    dtp = {str(fr): [] for fr in GRID}
    for _ in range(args.n_boot):
        ci = np.concatenate([ig[j] for j in rng.integers(0, len(iv), len(iv))])
        cpi = np.concatenate([pg[j] for j in rng.integers(0, len(pv), len(pv))])
        for fr in GRID:
            th5 = _threshold_for_fpr(SI5[ci], fr)
            th4 = _threshold_for_fpr(det0[2][ci], fr)
            dtp[str(fr)].append(float(
                np.mean(CO5[cpi] & (SP5[cpi] >= th5))
                - np.mean(det0[1][cpi] & (det0[0][cpi] >= th4))))
    rep["dTPR_cluster"] = {}
    for fr in GRID:
        lo, hi = np.percentile(dtp[str(fr)], [2.5, 97.5])
        exc = bool(lo > 0 or hi < 0)
        rep["dTPR_cluster"][str(fr)] = {"ci": [float(lo), float(hi)],
                                        "excludes_0": exc}
        print(f"  paired dTPR(geo5-geo4) @{fr:g}% CI [{lo:+.3f},{hi:+.3f}] "
              f"{'EXCLUDES 0' if exc else 'includes 0'}")

    out = PROJECT_ROOT / "runs" / args.out
    out.mkdir(parents=True, exist_ok=True)
    (out / "report.json").write_text(json.dumps(rep, indent=1, default=str),
                                     encoding="utf-8")
    print(f"[wrote] {out / 'report.json'}")


if __name__ == "__main__":
    main()
