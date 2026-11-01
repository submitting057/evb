# -*- coding: utf-8 -*-
"""T3 second item: LOVO cross-fitted stacking on disagreement features.

Plain-language summary:

  geo4 applies ONE formula (weighted geometric mean) to every session.
  Stacking lets the score be adjusted PER SESSION using the structure of
  how much the four members agree with each other.  The bet: magnets that
  fool everyone in a coordinated way might look different from correct
  probes in argmax agreement / top-5 overlap / distribution spread, and a
  small learned model can use that.

  Discipline (pre-registered in the roadmap):
  * features: s0 (geo4 max prob, the base) + argmax agreement count +
    mean pairwise top-5 Jaccard + mean pairwise Wasserstein-1 of the
    top-5 probability profiles;
  * cross-fitting: the unknown val videos are split into the SAME 5 folds
    as everywhere else, and the val PROBE videos get their own 5 folds;
    the model of fold f never sees fold f's impostor or probe videos ->
    the pooled LOVO score is honest;
  * monotone constraint: the base score s0 must keep a non-negative
    coefficient (never reward lower member confidence);
  * selection: stacked pooled LOVO must beat the s0 baseline (.773) on
    val first; only then is the final model (fit on all val) judged on
    test;
  * admission gate: paired video-cluster bootstrap dTPR(stacked - geo4),
    95% LOWER bound must exceed 0 at some tier; otherwise fall back to
    geo4 and close the line.
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
W4 = [0.35, 0.10, 0.20, 0.35]
FOLDS = 5
FEATS = ["s0", "agree", "jaccard", "w1"]


def norm(P):
    return P / (P.sum(1, keepdims=True) + 1e-12)


def geo(ws, mats):
    w = np.asarray(ws, float)
    w = w / w.sum()
    L = sum(wi * np.log(np.clip(P, 1e-12, 1)) for wi, P in zip(w, mats))
    L -= L.max(1, keepdims=True)
    E = np.exp(L)
    return E / E.sum(1, keepdims=True)


def features(member_mats, G):
    """Per-session disagreement features. member_mats: list of 4 prob
    matrices; G: the geo4 ensemble matrix (for s0 / argmax / top-5 order).
    """
    s0 = G.max(1)
    g_arg = G.argmax(1)
    n = len(G)
    agree = np.zeros(n)
    jacc = np.zeros(n)
    w1 = np.zeros(n)
    tops = [np.argsort(M, 1)[:, -5:] for M in member_mats]
    args = [M.argmax(1) for M in member_mats]
    for r in range(n):
        agree[r] = sum(a[r] == g_arg[r] for a in args)
        js, ws_ = [], []
        for i in range(4):
            for j in range(i + 1, 4):
                si, sj = set(tops[i][r].tolist()), set(tops[j][r].tolist())
                inter = len(si & sj)
                js.append(inter / (10 - inter))
                # 1D EMD between the two members' mass on the union top-5,
                # ordered by ensemble probability (descending)
                uni = list(si | sj)
                order = sorted(uni, key=lambda c: -G[r, c])
                mi = member_mats[i][r, uni]
                mj = member_mats[j][r, uni]
                mi = mi / (mi.sum() + 1e-12)
                mj = mj / (mj.sum() + 1e-12)
                vi = np.array([mi[uni.index(c)] for c in order])
                vj = np.array([mj[uni.index(c)] for c in order])
                ws_.append(np.abs(np.cumsum(vi) - np.cumsum(vj)).sum())
        jacc[r] = np.mean(js)
        w1[r] = np.mean(ws_)
    return np.stack([s0, agree / 4.0, jacc, w1], 1)


def _irls(X, ybin):
    """Raw standardized logistic regression; returns (pred, beta)."""
    mu, sd = X.mean(0), X.std(0) + 1e-9
    Z = (X - mu) / sd
    Z = np.concatenate([np.ones((len(Z), 1)), Z], 1)
    beta = np.zeros(Z.shape[1])
    for _ in range(100):
        p = 1 / (1 + np.exp(-Z @ beta))
        W = np.clip(p * (1 - p), 1e-6, None)
        grad = Z.T @ (ybin - p)
        H = (Z * W[:, None]).T @ Z + 1e-6 * np.eye(Z.shape[1])
        step = np.linalg.solve(H, grad)
        beta += step
        if np.abs(step).max() < 1e-8:
            break

    def pred(Xnew):
        Zn = (Xnew - mu) / sd
        Zn = np.concatenate([np.ones((len(Zn), 1)), Zn], 1)
        return Zn @ beta          # logit = linear score

    return pred, beta


def fit_lr(X, ybin):
    """Monotone constraint: column 0 is the base score s0; if its fitted
    coefficient comes out negative (the score should never reward LOWER
    member confidence), s0 is dropped and the model refit on the rest."""
    pred, beta = _irls(X, ybin)
    if beta[1] >= 0:
        return pred, beta
    pred2, beta2 = _irls(X[:, 1:], ybin)
    beta_out = np.concatenate([beta2[:1], [0.0], beta2[1:]])
    return (lambda Xn: pred2(Xn[:, 1:])), beta_out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="10s")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--out", default="t3_stacking")
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

    G = {s: geo(W4, [mem[n][s][0] for n in MEMBERS]) for s in ("val", "test")}
    GI = {s: geo(W4, [mem[n][s][1] for n in MEMBERS]) for s in ("val", "test")}
    F = {"val": features([mem[n]["val"][0] for n in MEMBERS], G["val"]),
         "test": features([mem[n]["test"][0] for n in MEMBERS], G["test"])}
    FI = {"val": features([mem[n]["val"][1] for n in MEMBERS], GI["val"]),
          "test": features([mem[n]["test"][1] for n in MEMBERS], GI["test"])}
    correct = {s: G[s].argmax(1) == y[s] for s in ("val", "test")}

    # fold assignment: impostor folds == global recipe; probe folds separate
    def folds_of(vrow):
        uv = np.array(sorted(set(vrow.tolist())))
        perm = np.random.default_rng(0).permutation(len(uv))
        sets = [set(uv[perm[f::FOLDS]].tolist()) for f in range(FOLDS)]
        v2f = {v: f for f, S in enumerate(sets) for v in S}
        return np.array([v2f[v] for v in vrow])

    rowf_p = folds_of(vid["val_probe"])
    rowf_i = folds_of(vid["val_imp"])

    # ---- pooled cross-fitted stacking on VAL ------------------------------
    SP, SI = [], []
    for f in range(FOLDS):
        tr_p = np.where(rowf_p != f)[0]
        tr_i = np.where(rowf_i != f)[0]
        X = np.concatenate([F["val"][tr_p], FI["val"][tr_i]])
        yb = np.concatenate([np.ones(len(tr_p)), np.zeros(len(tr_i))])
        pred, beta = fit_lr(X, yb)
        SP.append(pred(F["val"][rowf_p == f]))
        SI.append(pred(FI["val"][rowf_i == f]))
    SP, SI = np.concatenate(SP), np.concatenate(SI)

    def tpr_rows(sp, si, cor):
        out = {}
        for fr in GRID:
            th = _threshold_for_fpr(si, fr)
            out[fr] = float(np.mean(cor & (sp >= th)))
        return out

    m_stack = tpr_rows(SP, SI, correct["val"])
    m_base = tpr_rows(G["val"].max(1), GI["val"].max(1), correct["val"])
    print("=== VAL pooled LOVO (cross-fitted) ===")
    print(f"  stacked : {m_stack[0.005]:.3f}/{m_stack[0.01]:.3f}/"
          f"{m_stack[0.05]:.3f}")
    print(f"  s0 base : {m_base[0.005]:.3f}/{m_base[0.01]:.3f}/{m_base[0.05]:.3f}"
          "  (= geo4, anchor .773/.813/.897)")
    print(f"  LR coefs (bias,s0,agree,jaccard,w1) = "
          f"{np.round(beta, 3).tolist()}")

    rep = {"val_stacked": m_stack, "val_base": m_base,
           "coefs": beta.tolist(), "features": FEATS}

    beat = (m_stack[0.005], m_stack[0.01], m_stack[0.05]) > \
           (m_base[0.005], m_base[0.01], m_base[0.05])
    if not any([m_stack[k] > m_base[k] for k in GRID]):
        print("\nSELECTION GATE FAILED on val: stacking does not beat the "
              "base score -> line closes, geo4 stays. No test claim.")
        rep["verdict"] = "closed_on_val"
        out = PROJECT_ROOT / "runs" / args.out
        out.mkdir(parents=True, exist_ok=True)
        (out / "report.json").write_text(json.dumps(rep, indent=1, default=str),
                                         encoding="utf-8")
        print(f"[wrote] {out / 'report.json'}")
        return

    # ---- final model on all val -> test + paired cluster bootstrap --------
    X = np.concatenate([F["val"], FI["val"]])
    yb = np.concatenate([np.ones(len(F["val"])), np.zeros(len(FI["val"]))])
    pred, beta = fit_lr(X, yb)
    sp_t = pred(F["test"])
    si_t = pred(FI["test"])
    rep["coefs_full"] = beta.tolist()
    m_test = tpr_rows(sp_t, si_t, correct["test"])
    m_test_base = tpr_rows(G["test"].max(1), GI["test"].max(1), correct["test"])
    print("\n=== TEST ===")
    print(f"  stacked : {m_test[0.005]:.3f}/{m_test[0.01]:.3f}/{m_test[0.05]:.3f}")
    print(f"  geo4    : {m_test_base[0.005]:.3f}/{m_test_base[0.01]:.3f}/"
          f"{m_test_base[0.05]:.3f}  (anchor .654/.729/.880)")

    pv = sorted(set(vid["test_probe"].tolist()))
    iv = sorted(set(vid["test_imp"].tolist()))
    pg = [np.where(vid["test_probe"] == v)[0] for v in pv]
    ig = [np.where(vid["test_imp"] == v)[0] for v in iv]
    rng = np.random.default_rng(0)
    dtp = {str(fr): [] for fr in GRID}
    s0b, cob = G["test"].max(1), correct["test"]
    sib = GI["test"].max(1)
    for _ in range(args.n_boot):
        ci = np.concatenate([ig[j] for j in rng.integers(0, len(iv), len(iv))])
        cpi = np.concatenate([pg[j] for j in rng.integers(0, len(pv), len(pv))])
        for fr in GRID:
            thp = _threshold_for_fpr(si_t[ci], fr)
            thb = _threshold_for_fpr(sib[ci], fr)
            dtp[str(fr)].append(float(
                np.mean(correct["test"][cpi] & (sp_t[cpi] >= thp))
                - np.mean(cob[cpi] & (s0b[cpi] >= thb))))
    print(f"\n=== ADMISSION gate: paired cluster dTPR(stacked - geo4) ===")
    verdict = {}
    for fr in GRID:
        lo, hi = np.percentile(dtp[str(fr)], [2.5, 97.5])
        verdict[str(fr)] = {"ci": [float(lo), float(hi)],
                            "lcb": float(lo)}
        print(f"  @{fr:g}% CI [{lo:+.3f},{hi:+.3f}] LCB={lo:+.3f} "
              f"{'LCB>0 ADMIT' if lo > 0 else 'lcB<=0 -> fall back to geo4'}")
    rep["test_stacked"] = m_test
    rep["test_base"] = m_test_base
    rep["dTPR_cluster"] = verdict
    rep["verdict"] = ("admitted" if any(v["lcb"] > 0 for v in verdict.values())
                      else "fallback_geo4")

    out = PROJECT_ROOT / "runs" / args.out
    out.mkdir(parents=True, exist_ok=True)
    (out / "report.json").write_text(json.dumps(rep, indent=1, default=str),
                                     encoding="utf-8")
    print(f"[wrote] {out / 'report.json'}")


if __name__ == "__main__":
    main()
