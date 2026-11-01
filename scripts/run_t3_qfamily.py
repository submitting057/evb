# -*- coding: utf-8 -*-
"""T3 first item: power-mean (generalized mean) q family for the 4 members.

Plain-language summary:

  The current combiner is a weighted GEOMETRIC mean (geo) -- a "soft AND"
  over members.  The power mean M_q generalizes the whole spectrum:
      q = 1   arithmetic mean (weakest AND, most forgiving)
      q = 0   geometric mean  (= current geo4, must reproduce it exactly)
      q = -1  harmonic mean   (harder AND, low member dominates more)
      q -> -inf  minimum      (hard AND, the least confident member decides)
  Hypothesis from M2/T1b/T2-A1: the aux member is the independent vote;
  harder AND (q < 0) should amplify its veto on magnets -- but it may also
  drag down correct probes whenever ANY member is unsure, so the optimum q
  is an empirical question.

  Pre-registered procedure (roadmap T3):
    * calibration-free scores -> selection on VAL (= pooled LOVO equivalent
      for plain rules, as established in P1);
    * weight candidates: frozen geo4 pick (.35/.10/.20/.35) + the top
      distinct LOVO weight vectors from runs/combo_10s_strict_t2a1 + equal;
    * q grid: {-4, -3, -2, -1.5, -1, -0.5, -0.25, 0, 0.25, 0.5, 1};
    * pick best (q, w) by val @0.5% (tie-break @1%, @5%, top1);
    * KEEP gate: paired video-cluster bootstrap dTPR(picked - geo4 frozen)
      on TEST must EXCLUDE 0 at some tier ("beyond the delta noise floor");
      otherwise the family adds nothing and geo4 stays.
  Anchor: the q=0 row at the frozen weights must reproduce the accepted
  geo4 numbers exactly (val and test) -- otherwise the implementation is
  wrong and everything stops.
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
MEMBERS = ["lgbm", "dfs_branch_all_e100", "dfs_branch_all_dt", "dfs_branch_aux"]
SHORT = ["lgbm", "e100", "dt", "aux"]
W_FROZEN = [0.35, 0.10, 0.20, 0.35]
QGRID = [-4.0, -3.0, -2.0, -1.5, -1.0, -0.5, -0.25, 0.0, 0.25, 0.5, 1.0]


def norm(P):
    return P / (P.sum(1, keepdims=True) + 1e-12)


def pmean_log(ws, mats, q):
    """Log-domain weighted power mean, row-normalized to probabilities.

    log S_c = (1/q) * log(sum_i w_i * p_ic^q);  q=0 -> weighted geometric.
    Returns a proper probability matrix (rows sum to 1).
    """
    w = np.asarray(ws, float)
    w = w / w.sum()
    Ls = [np.log(np.clip(P, 1e-12, 1)) for P in mats]      # log p per member
    if q == 0.0:
        L = sum(wi * L for wi, L in zip(w, Ls))
    else:
        # logsumexp over members of (q * log p + log w)
        A = np.stack([q * L + np.log(max(wi, 1e-12)) for wi, L in zip(w, Ls)])
        mx = A.max(0)
        S = np.log(np.exp(A - mx).sum(0) + 1e-300) + mx
        L = S / q
    L -= L.max(1, keepdims=True)
    E = np.exp(L)
    return E / E.sum(1, keepdims=True)


def quick(P, y, I):
    best, correct = P.max(1), P.argmax(1) == y
    out = {"top1": float(np.mean(correct))}
    for f in GRID:
        th = _threshold_for_fpr(I.max(1), f)
        out[f] = float(np.mean(correct & (best >= th)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="10s")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--out", default="t3_qfamily")
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
    y = {"val": np.load(PROJECT_ROOT / "runs" / MEMBERS[1]
                        / "probs_val.npz")["probe_y"],
         "test": np.load(PROJECT_ROOT / "runs" / MEMBERS[1]
                         / "probs_test.npz")["probe_y"]}

    # ---- anchor check: q=0 at frozen weights must equal the geo4 numbers --
    P0v = pmean_log(W_FROZEN, [mem[n]["val"][0] for n in MEMBERS], 0.0)
    I0v = pmean_log(W_FROZEN, [mem[n]["val"][1] for n in MEMBERS], 0.0)
    a_val = quick(P0v, y["val"], I0v)
    P0t = pmean_log(W_FROZEN, [mem[n]["test"][0] for n in MEMBERS], 0.0)
    I0t = pmean_log(W_FROZEN, [mem[n]["test"][1] for n in MEMBERS], 0.0)
    a_test = quick(P0t, y["test"], I0t)
    print("=== anchor (q=0, w=.35/.10/.20/.35) ===")
    print(f"  val  {a_val[0.005]:.3f}/{a_val[0.01]:.3f}/{a_val[0.05]:.3f} "
          f"top1={a_val['top1']:.3f}")
    print(f"  test {a_test[0.005]:.3f}/{a_test[0.01]:.3f}/{a_test[0.05]:.3f} "
          f"top1={a_test['top1']:.3f}")
    exp = (0.654, 0.729, 0.880, 0.975)
    got = (round(a_test[0.005], 3), round(a_test[0.01], 3),
           round(a_test[0.05], 3), round(a_test["top1"], 3))
    print(f"  expected test (t2_aux_eval) {exp} -> "
          f"{'OK' if got == exp else 'MISMATCH ' + str(got) + ' -- ABORT'}")
    if got != exp:
        return

    # ---- weight candidates: frozen + top distinct LOVO vectors + equal ----
    rep_t2 = json.load(open(PROJECT_ROOT / "runs" / "combo_10s_strict_t2a1"
                            / "report.json", encoding="utf-8"))
    wcands = [tuple(W_FROZEN)]
    for row in rep_t2["lovo"]["top15"]:
        wv = tuple(round(float(x), 2) for x in row["w"])
        if wv not in wcands:
            wcands.append(wv)
    wcands.append(tuple(round(1 / 4, 4) for _ in range(4)))
    print(f"\nweight candidates: {len(wcands)} "
          f"(frozen + top{len(wcands) - 2} LOVO + equal)")

    # ---- val scan over (q, w) ----------------------------------------------
    rows = []
    for w in wcands:
        for q in QGRID:
            Pv = pmean_log(w, [mem[n]["val"][0] for n in MEMBERS], q)
            Iv = pmean_log(w, [mem[n]["val"][1] for n in MEMBERS], q)
            m = quick(Pv, y["val"], Iv)
            rows.append({"q": q, "w": list(w), "val": m})
    rows.sort(key=lambda r: (-r["val"][0.005], -r["val"][0.01],
                             -r["val"][0.05], -r["val"]["top1"]))

    print(f"\n=== val selection (top 12 of {len(rows)} configs) ===")
    print(f"{'q':>6} {'weights':24s} | val @.5%  @1%  @5%  top1")
    for r in rows[:12]:
        print(f"{r['q']:6.2f} {str(r['w']):24s} | {r['val'][0.005]:.3f} "
              f"{r['val'][0.01]:.3f} {r['val'][0.05]:.3f}  {r['val']['top1']:.3f}")

    # q spectrum at frozen weights (direction diagnostic)
    spec = [r for r in rows if tuple(r["w"]) == tuple(W_FROZEN)]
    spec.sort(key=lambda r: QGRID.index(r["q"]))
    print("\nq spectrum at frozen weights (val):")
    print("  " + "  ".join(f"q={r['q']:g}:{r['val'][0.005]:.3f}" for r in spec))

    pick = rows[0]
    base = next(r for r in rows if r["q"] == 0.0
                and tuple(r["w"]) == tuple(W_FROZEN))
    print(f"\npicked: q={pick['q']} w={pick['w']} "
          f"(val {pick['val'][0.005]:.3f}) vs baseline q=0 "
          f"(val {base['val'][0.005]:.3f})")

    # ---- test finals + paired cluster bootstrap ---------------------------
    def eval_rule(w, q):
        P = pmean_log(w, [mem[n]["test"][0] for n in MEMBERS], q)
        I = pmean_log(w, [mem[n]["test"][1] for n in MEMBERS], q)
        return P.max(1), P.argmax(1) == y["test"], I.max(1)

    SPp, COp, SIp = eval_rule(pick["w"], pick["q"])
    SPb, COb, SIb = eval_rule(W_FROZEN, 0.0)
    det = {}
    for rn, sp, co in (("picked", SPp, COp), ("geo4_base", SPb, COb)):
        si = SIp if rn == "picked" else SIb
        row = {"top1": float(co.mean())}
        for fr in GRID:
            th = _threshold_for_fpr(si, fr)
            row[fr] = float(np.mean(co & (sp >= th)))
        det[rn] = row
        print(f"  {rn:9s} top1={row['top1']:.3f} " + " ".join(
            f"@{fr:g}%={row[fr]:.3f}" for fr in GRID))

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
            thp = _threshold_for_fpr(SIp[ci], fr)
            thb = _threshold_for_fpr(SIb[ci], fr)
            dtp[str(fr)].append(
                float(np.mean(COp[cpi] & (SPp[cpi] >= thp))
                      - np.mean(COb[cpi] & (SPb[cpi] >= thb))))
    print(f"\n=== KEEP gate: paired cluster dTPR(picked - geo4) "
          f"({args.n_boot} reps) ===")
    verdict = {}
    for fr in GRID:
        lo, hi = np.percentile(dtp[str(fr)], [2.5, 97.5])
        exc = bool(lo > 0 or hi < 0)
        verdict[str(fr)] = {"ci": [float(lo), float(hi)], "excludes_0": exc}
        print(f"  @{fr:g}% CI [{lo:+.3f},{hi:+.3f}] "
              f"{'EXCLUDES 0' if exc else 'includes 0'}")

    rep = {"anchor": {"val": a_val, "test": a_test, "expected_test": list(exp)},
           "q_grid": QGRID, "weight_candidates": [list(w) for w in wcands],
           "val_top12": rows[:12], "q_spectrum_frozen": [
               {"q": r["q"], "val": r["val"]} for r in spec],
           "picked": pick, "test": det, "dTPR_cluster": verdict}
    out = PROJECT_ROOT / "runs" / args.out
    out.mkdir(parents=True, exist_ok=True)
    (out / "report.json").write_text(json.dumps(rep, indent=1, default=str),
                                     encoding="utf-8")
    print(f"[wrote] {out / 'report.json'}")


if __name__ == "__main__":
    main()
