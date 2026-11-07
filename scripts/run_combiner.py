# -*- coding: utf-8 -*-
"""Open-set combiner search over saved member probabilities (no retraining).

Members: LGBM (runs/cache/lgbm_probs_{ds}.npz) + DF variants (runs/{tag}/probs_*.npz).
Rules:
  arith : weighted arithmetic mean of probs
  geo   : weighted geometric mean  p_c ∝ prod_i p_i[c]^w_i   (soft AND gate)
  min   : elementwise min over members (hard AND gate, equal weights)
  + temperature: per-member T fitted on val probes (NLL) before combining.

B1 p-space rules (ECDF calibration, scale-free soft AND): each member's scalar
score s = max prob is first converted to a p-value against that member's own
VAL-impostor score distribution (conservative empirical p =
(1 + #{ref >= s}) / (n_ref + 1)), then the p-values are combined:
  pfisher   : S = -sum_i w_i ln p_i            (weighted Fisher)
  pstouffer : S = sum_i w_i Phi^-1(1 - p_i)    (weighted Stouffer z)
  ptippett  : S = -ln min_i p_i                (weakest evidence; equal weights)
  pmean     : S = sum_i w_i (1 - p_i)          (linear soft AND)
For p-space rules the argmax/top1 come from the geo ensemble with the same
weights; only the open-set score lives in p-space. No temperature for p-rules
(the ECDF is the calibration).

Class-wise ECDF rules (B1 fix; scalar p-rules lose the per-class AND gate that
the prob-space geo mean gets for free -- impostors have ~1% cross-member argmax
agreement vs ~91% for correct probes, so a score that ignores WHICH class is
agreed on lets disagreeing-but-confident impostors float to the top):
  cfisher : S_c = -sum_i w_i ln p_i,c  per class, score = max_c S_c (argmax too)
  cpmean  : S_c = sum_i w_i (1 - p_i,c) per class
where p_i,c compares member i's prob FOR CLASS c against the val-impostor
distribution of probs for that same class.
  gcfisher: hybrid -- argmax/top1 from the geo ensemble (best closed-set),
            open-set score from cfisher with the same weights.

A2 score families: per-member scalar confidence statistics beyond max-prob
(max-logit / energy need runs/{tag}/logits_*.npz from run_tta_dump.py;
neg-entropy / margin work from probs).  Each member's best family is chosen
on VAL and added to gcfisher as auxiliary scalar evidence:
  gcfisher_a: score = max_c class-wise Fisher + alpha * weighted scalar evidence
              (whole-vector veto on top of the per-class AND gate).

A3 gallery-cosine members (--cos-tags): per-class cosine-to-gallery-template
matrices from scripts/run_gallery_cos.py.  They are SCORE-ONLY members: they
join the class-wise ECDF and the scalar evidence (with their own raw-space
tail extension), but NOT the prob-space rules or the argmax (cosines are not
probabilities; the closed-set answer stays with the prob ensemble).  Weights
live on the full member list; prob rules use the renormalized prob subset.

Selection: weights/temperature/rule chosen by VAL TPR@0.5%FPR (tie-break @1%).
Test is only reported, never used for selection.

P1 (--lovo, strict-protocol repair): the class-wise ECDF prices class c against
the val-impostor distribution FOR THAT class; with ~10 firing impostors per
class those prices overfit the specific unknown videos in the calibration pool
(sixth-round collapse).  Two repairs, jointly selected:
  1) shrinkage  p_shrunk = (k_c * p_emp + kappa * p_pop) / (k_c + kappa),
     k_c = #complement val impostors whose class-c value exceeds the pooled
     median (fire depth), p_pop = price against the POOLED distribution of all
     complement cells (identity-agnostic).  kappa=0 keeps the old per-class
     pricing; kappa=-1 is pure population pricing.
  2) LOVO selection: the unknown VAL videos are split into --folds groups;
     every candidate (rule/weights/kappa/alpha) is calibrated on the other 4/5
     and judged on the held-out fifth (threshold from the calibration folds,
     probes scored under calibration-fold pricing).  Selection uses only the
     LOVO-aggregated TPR; TEST numbers use the ordinary full-val calibration.
"""
import argparse
import itertools
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scipy.stats import norm as _norm

from src.config import PROJECT_ROOT
from src.evaluate import (_threshold_for_fpr, open_set_metrics,
                          open_set_with_custom_score, tpr_at_fpr)

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


def quick_score(correct, sp, si):
    """quick() for scalar-score rules: top1 from precomputed argmax-correctness,
    open-set TPR from explicit probe/impostor scores."""
    out = {"top1": float(np.mean(correct))}
    for f in GRID:
        th = _threshold_for_fpr(si, f)
        out[f] = float(np.mean(correct & (sp >= th)))
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
    ap.add_argument("--protocol", default="legacy", choices=["legacy", "strict"])
    ap.add_argument("--df-tags", required=True)
    ap.add_argument("--step", type=float, default=0.05)
    ap.add_argument("--out", default=None,
                    help="output dir name under runs/ (default combo_{dataset})")
    ap.add_argument("--cos-tags", default="",
                    help="comma list of gallery-cosine member dirs (score-only)")
    ap.add_argument("--lovo", action="store_true",
                    help="P1: leave-unknown-video-out selection + per-class "
                         "pricing shrinkage (strict protocol)")
    ap.add_argument("--lgbm-suffix", default="",
                    help="load lgbm_probs_{dataset}_strict{suffix}.npz instead "
                         "of the default cache (e.g. _v3 for packet-aware v3)")
    ap.add_argument("--families", default="all", choices=["all", "plain"],
                    help="'plain': only arith/geo/min (memory-safe for many "
                         "members -- the cfisher/A2 weight-grid cache grows "
                         "linearly with the weight count and OOMs at 4+ members)")
    ap.add_argument("--folds", type=int, default=5)
    args = ap.parse_args()
    tags = [t for t in args.df_tags.split(",") if t]
    cos_tags = [t for t in args.cos_tags.split(",") if t]

    if args.protocol == "strict":
        lgbm_cache = PROJECT_ROOT / "runs" / "cache" / f"lgbm_probs_{args.dataset}_strict{args.lgbm_suffix}.npz"
    else:
        lgbm_cache = PROJECT_ROOT / "runs" / "cache" / f"lgbm_probs_{args.dataset}{args.lgbm_suffix}.npz"
    z = np.load(lgbm_cache)
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

    LOG = {}                                        # A2: raw logits if dumped
    for t in tags:
        for split in ("val", "test"):
            p = PROJECT_ROOT / "runs" / t / f"logits_{split}.npz"
            if p.exists():
                d = np.load(p)
                assert (d["probe_y"] == y[split]).all(), "logit row order mismatch"
                LOG[(t, split)] = (d["probe"], d["imp"])

    # A3: gallery-cosine members, raw cosines (NOT row-normalized)
    cosset = set(cos_tags)
    members = names + cos_tags
    COS = {}
    for t in cos_tags:
        for split in ("val", "test"):
            d = np.load(PROJECT_ROOT / "runs" / t / f"probs_{split}.npz")
            assert (d["probe_y"] == y[split]).all(), "cos row order mismatch"
            COS[(t, split)] = (d["probe"], d["imp"])

    def srcm(n, split):
        return COS[(n, split)] if n in cosset else raw[(n, split)]

    def mat(n, split, k):
        return srcm(n, split)[k]

    temps = {n: fit_temp(raw[(n, "val")][0], y["val"]) for n in names}
    print("members+temperature:", {n: temps[n] for n in names})

    # ---- B1: ECDF p-values against each member's own VAL-impostor scores ----
    REF = {n: np.sort(raw[(n, "val")][1].max(1)) for n in names}
    PV = {}
    for n in names:
        m = len(REF[n])
        for split in ("val", "test"):
            pls = []
            for k, M in enumerate(raw[(n, split)]):        # k=0 probe, k=1 impostor
                # leave-one-out for val impostors (their scores are in the ref):
                # with the plain +1 they can never reach the saturation level that
                # out-of-sample rows can, which made val selection blind to
                # knife-edge configs -- LOO makes them exactly exchangeable.
                add = 0 if (split == "val" and k == 1) else 1
                pls.append((add + m - np.searchsorted(REF[n], M.max(1), side="left"))
                           / (m + 1.0))
            PV[(n, split)] = tuple(pls)
    print("pval ref sizes (val impostors):", {n: len(REF[n]) for n in names})

    def combine_pscore(rule, ws, split):
        """Combined scalar open-set score in p-space -> (probe_scores, imp_scores).
        Higher = more confidently 'known' on every rule."""
        pairs = [(n, w) for n, w in zip(names, ws) if w > 0]
        s = sum(w for _, w in pairs)
        pairs = [(n, w / s) for n, w in pairs]
        A = np.stack([PV[(n, split)][0] for n, _ in pairs])
        B = np.stack([PV[(n, split)][1] for n, _ in pairs])
        if rule == "pfisher":
            f = lambda M: -np.log(M)
        elif rule == "pstouffer":
            f = lambda M: _norm.ppf(1.0 - np.clip(M, 1e-12, 1.0 - 1e-12))
        elif rule == "ptippett":
            return -np.log(A.min(0)), -np.log(B.min(0))
        elif rule == "pmean":
            f = lambda M: 1.0 - M
        else:
            raise ValueError(rule)
        W = np.array([w for _, w in pairs])[:, None]
        return (W * f(A)).sum(0), (W * f(B)).sum(0)

    def score_metrics(P, yy, sp, si, n_boot=1000, seed=0):
        """Open-set metrics for a custom scalar score (argmax still from P),
        incl. detection AUROC and bootstrap CI (mirrors open_set_metrics)."""
        m = open_set_with_custom_score(P, yy, sp, si)
        lab = np.concatenate([np.ones(len(sp)), np.zeros(len(si))])
        sc = np.concatenate([sp, si])
        rk = np.argsort(np.argsort(sc)) + 1
        pos = lab == 1
        npos, nneg = int(pos.sum()), int((~pos).sum())
        m["auroc_detection"] = float((rk[pos].sum() - npos * (npos + 1) / 2) / (npos * nneg))
        m["n_probes"], m["n_impostors"] = int(len(sp)), int(len(si))
        correct = P.argmax(1) == yy
        rng = np.random.default_rng(seed)
        for f in GRID:
            boot = []
            for _ in range(n_boot):
                pi = rng.integers(0, len(sp), len(sp))
                ii = rng.integers(0, len(si), len(si))
                t, _ = tpr_at_fpr(sp[pi], correct[pi], si[ii], f)
                boot.append(t)
            m["fpr"][f]["tpr_ci95"] = [float(np.percentile(boot, 2.5)),
                                       float(np.percentile(boot, 97.5))]
        return m

    # ---- B1 fix: class-wise ECDF (keeps the per-class AND gate in p-space) ----
    # Empirical p floors at 1/(n+1): without an extension, ~24% of impostors and
    # ~91% of correct probes tie exactly at the ceiling S=ln(n+1) and TPR@0.5%
    # degenerates into float-rounding luck. Extend the tail beyond each class's
    # reference max with a shifted exponential: in logit space for prob members,
    # in raw space for cosine members; scale beta = top-decile mean excess.
    def _logit(x):
        x = np.clip(x, 1e-7, 1 - 1e-7)
        return np.log(x / (1 - x))
    SPACE = {}
    for n in names:
        g = _logit(raw[(n, "val")][1].max(1))
        t = np.quantile(g, 0.9)
        SPACE[n] = ("logit", float(max((g[g > t] - t).mean(), 1e-3)))
    for n in cos_tags:
        s = COS[(n, "val")][1].max(1)
        t = np.quantile(s, 0.9)
        SPACE[n] = ("raw", float(max((s[s > t] - t).mean(), 1e-4)))
    print("tail scales:", {n: f"{m}:{round(b, 3)}" for n, (m, b) in SPACE.items()})
    _CT = {}

    def ctrans(rule, n, split):
        """Transformed per-class p matrices for member n; cached.
        Reference is ALWAYS the member's VAL-impostor distribution."""
        key = (rule, n, split)
        if key not in _CT:
            mode, beta = SPACE[n]
            tf = _logit if mode == "logit" else (lambda x: x)
            r = np.sort(srcm(n, "val")[1], axis=0)
            m = r.shape[0]
            if rule == "cfisher":
                f = lambda M: -np.log(M)
            elif rule == "cpmean":
                f = lambda M: 1.0 - M
            else:
                raise ValueError(rule)
            out = []
            for k, M in enumerate(srcm(n, split)):         # k=0 probe, k=1 impostor
                add = 0 if (split == "val" and k == 1) else 1   # LOO, see PV above
                Pc = np.empty(M.shape, dtype=np.float64)
                for j in range(M.shape[1]):
                    rj = r[:, j]
                    cnt = m - np.searchsorted(rj, M[:, j], side="left")
                    Pc[:, j] = (add + cnt) / (m + 1.0)
                    hi = cnt == 0                           # above the whole reference
                    if hi.any():
                        dg = tf(M[hi, j]) - tf(rj[-1])
                        Pc[hi, j] = np.exp(-dg / beta) / (m + 1.0)
                out.append(f(Pc).astype(np.float32))
            _CT[key] = tuple(out)
        return _CT[key]

    def combine_cmat(rule, ws, split):
        """Combined per-class score matrix -> (probe_S, imp_S)."""
        pairs = [(n, w) for n, w in zip(members, ws) if w > 0]
        s = sum(w for _, w in pairs)
        pairs = [(n, w / s) for n, w in pairs]
        Tp = sum(w * ctrans(rule, n, split)[0] for n, w in pairs)
        Ti = sum(w * ctrans(rule, n, split)[1] for n, w in pairs)
        return Tp, Ti

    # ---- A2: per-member scalar score families + auxiliary evidence ----
    def _families(n, split, k):
        M = mat(n, split, k)
        part = np.partition(M, -2, 1)
        if n in cosset:                       # cosines: only shape statistics
            return {"maxval": M.max(1),
                    "margin": part[:, -1] - part[:, -2]}
        out = {"maxprob": M.max(1),
               "negent": (M * np.log(np.clip(M, 1e-12, 1))).sum(1),  # = -entropy
               "margin": part[:, -1] - part[:, -2]}
        if (n, split) in LOG:
            L = np.asarray(LOG[(n, split)][k], dtype=np.float64)
            mx = L.max(1, keepdims=True)
            out["maxlogit"] = mx[:, 0]
            out["energy"] = mx[:, 0] + np.log(np.exp(L - mx).sum(1))
        return out

    def _quick_family(n, fam, split):
        sp = _families(n, split, 0)[fam]
        si = _families(n, split, 1)[fam]
        correct = mat(n, split, 0).argmax(1) == y[split]
        return {f: float(np.mean(correct & (sp >= _threshold_for_fpr(si, f))))
                for f in GRID}

    print("\nA2 score-family ablation (single member; * = val-chosen):")
    FAM = {}
    for n in members:
        rows = []
        for fam in _families(n, "val", 0):
            rows.append((fam, _quick_family(n, fam, "val"),
                         _quick_family(n, fam, "test")))
        FAM[n] = max(rows, key=lambda r: (r[1][0.005], r[1][0.01]))[0]
        for fam, v, t in rows:
            mark = "*" if fam == FAM[n] else " "
            print(f"  {n:24s} {fam:9s}{mark} val@.5%={v[0.005]:.3f} @1%={v[0.01]:.3f}"
                  f"   test@.5%={t[0.005]:.3f} @1%={t[0.01]:.3f}")

    # scalar-evidence ECDF p-values (LOO for val impostors, raw-space tail ext)
    SREF2 = {n: np.sort(_families(n, "val", 1)[FAM[n]]) for n in members}
    SB = {}
    for n in members:
        r = SREF2[n]
        t0 = np.quantile(r, 0.9)
        SB[n] = float(max((r[r > t0] - t0).mean(), 1e-6))
    SEV = {}
    for n in members:
        r, m = SREF2[n], len(SREF2[n])
        for split in ("val", "test"):
            evs = []
            for k in (0, 1):
                s = _families(n, split, k)[FAM[n]]
                cnt = m - np.searchsorted(r, s, side="left")
                add = 0.0 if (split == "val" and k == 1) else 1.0
                p = (add + cnt) / (m + 1.0)
                hi = cnt == 0
                if hi.any():
                    p[hi] = np.exp(-(s[hi] - r[-1]) / SB[n]) / (m + 1.0)
                evs.append(-np.log(p))
            SEV[(n, split)] = tuple(evs)

    def wnrm(ws):
        s = sum(w for w in ws if w > 0)
        return [w / s if w > 0 else 0.0 for w in ws]

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

    def pw(w):
        """Renormalized prob-member subset of a full-member weight vector."""
        s = sum(w[:len(names)])
        return [wi / s for wi in w[:len(names)]] if s > 1e-9 else None

    w0full = tuple(round(1.0 / len(members), 4) for _ in members)
    p_wlist = [("equal", w0full)] + [
        (f"w={w}", w) for w in simplex(args.step, len(members))
        if sum(w[:len(names)]) > 1e-9]

    cmat_cache = {}                       # weight -> combined matrices (val only)

    def cmat_w(rule, wname, w):
        key = (rule, wname)
        if key not in cmat_cache:
            cmat_cache[key] = combine_cmat(rule, list(w), "val")
        return cmat_cache[key]

    variants = []
    plain_scores = {}            # P1 LOVO reuse: (rule,T,wname) -> (sp, si, correct)
    for use_T in (False, True):
        for rule in ("arith", "geo", "min"):
            cands = [("equal", w0full)] if rule == "min" else p_wlist
            for wname, w in cands:
                Pw = pw(list(w))
                Pv = combine(rule, Pw, [scaled(n, "val", use_T)[0] for n in names])
                Iv = combine(rule, Pw, [scaled(n, "val", use_T)[1] for n in names])
                plain_scores[(rule, use_T, wname)] = (Pv.max(1), Iv.max(1),
                                                      Pv.argmax(1) == y["val"])
                variants.append({"rule": rule, "temp": use_T, "w": list(w),
                                 "wname": wname, "val": quick(Pv, y["val"], Iv)})

    # B1 p-space rules (no temperature: the ECDF is the calibration).
    # argmax/top1 come from the geo ensemble over PROB members -> cache them.
    plain_only = args.families == "plain"
    correct_cache = {}
    if not plain_only:
        for wname, w in p_wlist:
            Pv = combine("geo", pw(list(w)), [raw[(n, "val")][0] for n in names])
            correct_cache[wname] = Pv.argmax(1) == y["val"]
        sp, si = combine_pscore("ptippett", pw(list(w0full)), "val")
        variants.append({"rule": "ptippett", "temp": False, "w": list(w0full),
                         "wname": "equal",
                         "val": quick_score(correct_cache["equal"], sp, si)})
        for rule in ("pfisher", "pstouffer", "pmean"):
            for wname, w in p_wlist:
                sp, si = combine_pscore(rule, pw(list(w)), "val")
                variants.append({"rule": rule, "temp": False, "w": list(w),
                                 "wname": wname,
                                 "val": quick_score(correct_cache[wname], sp, si)})
        for rule in ("cfisher", "cpmean"):
            for wname, w in p_wlist:
                Tp, Ti = cmat_w(rule, wname, w)
                if rule == "cfisher":  # hybrid twin: geo argmax + cfisher score
                    variants.append({"rule": "gcfisher", "temp": False, "w": list(w),
                                     "wname": wname,
                                     "val": quick_score(correct_cache[wname],
                                                        Tp.max(1), Ti.max(1))})
                correct = Tp.argmax(1) == y["val"]
                variants.append({"rule": rule, "temp": False, "w": list(w),
                                 "wname": wname,
                                 "val": quick_score(correct, Tp.max(1), Ti.max(1))})

        # A2: gcfisher + auxiliary scalar evidence (whole-vector veto)
        for alpha in (0.15, 0.3, 0.6, 1.0):
            for wname, w in p_wlist:
                wn = wnrm(list(w))
                Tp, Ti = cmat_w("cfisher", wname, w)
                aux_p = sum(u * SEV[(n, "val")][0] for n, u in zip(members, wn) if u > 0)
                aux_i = sum(u * SEV[(n, "val")][1] for n, u in zip(members, wn) if u > 0)
                variants.append({"rule": "gcfisher_a", "temp": False, "alpha": alpha,
                                 "w": list(w), "wname": wname,
                                 "val": quick_score(correct_cache[wname],
                                                    Tp.max(1) + alpha * aux_p,
                                                    Ti.max(1) + alpha * aux_i)})

    # ---- P1: LOVO selection + per-class pricing shrinkage (see module doc) ----
    lovo_report = None
    if args.lovo:
        from src.config import RunConfig
        from src.data import load_splits, session_pool

        nf = args.folds
        KAPPAS = (0.0, 4.0, 16.0, 64.0, 256.0, -1.0)   # -1 = pure population pricing
        ALPHAS = (0.0, 0.15, 0.3, 0.6, 1.0)     # 0.0 -> plain gcfisher

        rcfg = RunConfig(dataset=args.dataset, protocol=args.protocol)
        spl = load_splits(rcfg.splits_path)
        vis = session_pool(spl, "unknown", "val")
        vid_row = np.array([spl["sessions"][s]["video"] for s in vis])
        assert len(vis) == raw[(names[0], "val")][1].shape[0], \
            "val impostor row count mismatch (splits vs saved probs)"
        uvid = np.array(sorted(set(vid_row.tolist())))
        perm = np.random.default_rng(0).permutation(len(uvid))  # FIXED criterion seed
        fold_sets = [set(uvid[perm[g::nf]].tolist()) for g in range(nf)]
        v2f = {v: g for g, S in enumerate(fold_sets) for v in S}
        row_fold = np.array([v2f[v] for v in vid_row])
        print(f"\n=== P1 LOVO: {len(uvid)} unknown val videos -> {nf} folds | "
              f"videos {'/'.join(str(len(S)) for S in fold_sets)} | "
              f"sessions {'/'.join(str(int((row_fold == g).sum())) for g in range(nf))} ===")

        def _tf_of(n):
            return _logit if SPACE[n][0] == "logit" else (lambda x: x)

        def _floor_of(n):
            return 1e-3 if SPACE[n][0] == "logit" else 1e-4

        def _beta_td(vals, floor):
            t = np.quantile(vals, 0.9)
            return float(max((vals[vals > t] - t).mean(), floor))

        def _ecdf_p(ref, q, insamp, tf, beta):
            """Empirical p of q vs ascending ref, exp tail past the ref max.

            insamp rows are part of the reference: their own value is removed
            from the count (true LOO) and the tail anchor is the max of the
            OTHERS (ref[-2]) -- without this, in-sample rows can never exceed
            the ln(m+1) ceiling while held-out rows can, which breaks the
            exchangeability of calibration vs evaluation impostors (realized
            FPR ~26% at a nominal 0.5% in the first P1 run)."""
            m = ref.size
            cnt = m - np.searchsorted(ref, q, side="left")
            if isinstance(insamp, np.ndarray):
                p = np.where(insamp, cnt, cnt + 1.0) / (m + 1.0)
                hi_in = insamp & (cnt == 1)
                hi_out = (~insamp) & (cnt == 0)
                if hi_in.any():
                    p[hi_in] = np.exp(-(tf(q[hi_in]) - tf(ref[-2])) / beta) / (m + 1.0)
                if hi_out.any():
                    p[hi_out] = np.exp(-(tf(q[hi_out]) - tf(ref[-1])) / beta) / (m + 1.0)
                return p
            p = (cnt + (0.0 if insamp else 1.0)) / (m + 1.0)
            hi = cnt == (1 if insamp else 0)
            if hi.any():
                rtop = ref[-2] if (insamp and m >= 2) else ref[-1]
                p[hi] = np.exp(-(tf(q[hi]) - tf(rtop)) / beta) / (m + 1.0)
            return p

        registry = {}

        def pooled_metrics(correct, SP, SI):
            """TPR@FPR from the pooled LOVO scores: fold impostors pooled across
            folds are the honest 'unseen unknown videos' sample, so the nominal
            and realized FPR coincide by construction (the first two P1 runs
            compared candidates at nominal thresholds whose realized FPR
            differed 26x -- gcfisher kappa=0 rode unpatrolled classes to 13%
            realized FPR and an invalid top rank)."""
            out = {"top1": float(np.mean(correct))}
            for fr in GRID:
                th = _threshold_for_fpr(SI, fr)
                out[fr] = float(np.mean(correct & (SP >= th)))
            return out

        def fold_class_p(n, g):
            """Class-wise p_emp / p_pop + fire depth k_c, fold-g complement ref.

            k_c counts complement impostors whose class-c value exceeds the
            POOLED Q99 (upper-tail fire depth): the first P1 run used the
            pooled median, which makes every class ~1950 'deep' and the
            kappa<=64 shrinkage weight <2% -- inert by parameterization."""
            keep = row_fold != g
            Iref = srcm(n, "val")[1][keep]
            r = np.sort(Iref, axis=0)
            tf = _tf_of(n)
            beta_emp = _beta_td(tf(Iref.max(1)), _floor_of(n))
            Pq, Iq = srcm(n, "val")
            pe_p = np.empty(Pq.shape, dtype=np.float64)
            pe_i = np.empty(Iq.shape, dtype=np.float64)
            for j in range(Pq.shape[1]):
                rj = r[:, j]
                pe_p[:, j] = _ecdf_p(rj, Pq[:, j], False, tf, beta_emp)
                pe_i[:, j] = _ecdf_p(rj, Iq[:, j], keep, tf, beta_emp)
            pool = np.sort(Iref.ravel())
            beta_pop = _beta_td(tf(pool), _floor_of(n))
            pp_p = _ecdf_p(pool, Pq.ravel(), False, tf, beta_pop).reshape(Pq.shape)
            pp_i = _ecdf_p(pool, Iq.ravel(), False, tf, beta_pop).reshape(Iq.shape)
            k_c = (Iref > np.quantile(pool, 0.99)).sum(0).astype(np.float64)
            return pe_p, pe_i, pp_p, pp_i, k_c

        def fold_sev(n, g):
            """Scalar-evidence -ln p under fold-g complement reference."""
            keep = row_fold != g
            r = np.sort(_families(n, "val", 1)[FAM[n]][keep])
            beta = _beta_td(r, 1e-6)
            out = []
            for k in (0, 1):
                s = _families(n, "val", k)[FAM[n]]
                insamp = keep if k == 1 else False
                out.append(-np.log(_ecdf_p(r, s, insamp, lambda x: x, beta))
                           .astype(np.float32))
            return tuple(out)

        # plain prob-space rules: scores are calibration-free; the pooled LOVO
        # metric equals the plain val metric up to fold sampling.
        wmap = dict(p_wlist)
        wmap["equal"] = w0full
        for (rule, use_T, wname), (sp, si, correct) in plain_scores.items():
            registry[("plain", rule, use_T, wname)] = {
                "rule": rule, "temp": use_T, "kappa": None, "alpha": None,
                "wname": wname, "w": list(wmap[wname]), "correct": correct,
                "sp": [sp.astype(np.float32)], "si": [si.astype(np.float32)]}

        # shrunk class-wise pricing (gcfisher family); per-fold probe and
        # held-out-impostor scores are pooled across folds for the criterion
        for g in range(nf) if not plain_only else []:
            CP = {n: fold_class_p(n, g) for n in members}
            SV = {n: fold_sev(n, g) for n in members}
            m_imp = row_fold == g
            for kap in KAPPAS:
                Tm = {}
                for n in members:
                    pe_p, pe_i, pp_p, pp_i, k_c = CP[n]
                    if kap == 0.0:
                        mp = (pe_p, pe_i)
                    elif kap < 0:
                        mp = (pp_p, pp_i)
                    else:
                        wgt = (k_c / (k_c + kap))[None, :]
                        mp = (wgt * pe_p + (1.0 - wgt) * pp_p,
                              wgt * pe_i + (1.0 - wgt) * pp_i)
                    Tm[n] = tuple(-np.log(M).astype(np.float32) for M in mp)
                for wname, w in p_wlist:
                    prs = [(n, u) for n, u in zip(members, wnrm(list(w))) if u > 0]
                    Spc = sum(u * Tm[n][0] for n, u in prs).max(1)
                    Sic = sum(u * Tm[n][1] for n, u in prs)[m_imp].max(1)
                    aux_p = sum(u * SV[n][0] for n, u in prs)
                    aux_i = sum(u * SV[n][1] for n, u in prs)[m_imp]
                    key = ("shrunk", kap, wname)
                    if key not in registry:
                        registry[key] = {"w": list(w), "correct": correct_cache[wname],
                                         "spc": [], "sic": [], "aux_p": [], "aux_i": []}
                    rec = registry[key]
                    rec["spc"].append(Spc.astype(np.float32))
                    rec["sic"].append(Sic.astype(np.float32))
                    rec["aux_p"].append(aux_p.astype(np.float32))
                    rec["aux_i"].append(aux_i.astype(np.float32))
            print(f"[lovo] fold {g} done")

        rows = []
        for key, rec in registry.items():
            if key[0] == "plain":
                m = pooled_metrics(np.tile(rec["correct"], len(rec["sp"])),
                                   np.concatenate(rec["sp"]),
                                   np.concatenate(rec["si"]))
                rows.append({"rule": rec["rule"], "temp": rec["temp"],
                             "kappa": None, "alpha": None, "wname": rec["wname"],
                             "w": rec["w"], "lovo": m, "top1": m["top1"]})
                continue
            _, kap, wname = key
            correct = np.tile(rec["correct"], nf)
            SPC = np.concatenate(rec["spc"])
            SIC = np.concatenate(rec["sic"])
            AXP = np.concatenate(rec["aux_p"])
            AXI = np.concatenate(rec["aux_i"])
            for alpha in ALPHAS:
                m = pooled_metrics(correct, SPC + alpha * AXP, SIC + alpha * AXI)
                rows.append({"rule": "gcfisher" if alpha == 0.0 else "gcfisher_a",
                             "temp": False, "kappa": kap, "alpha": alpha,
                             "wname": wname, "w": rec["w"], "lovo": m,
                             "top1": m["top1"]})
        rows.sort(key=lambda r: (-r["lovo"][0.005], -r["lovo"][0.01],
                                 -r["lovo"][0.05], -r["top1"]))

        def _first(pred):
            for r in rows:
                if pred(r):
                    return r
            return None

        print(f"\n{'rule':11s}{'kap':>5}{'a':>6} {'weights':24s} | "
              f"LOVO @.5%  @1%  @5%  top1")
        for r in rows[:15]:
            kap = "-" if r["kappa"] is None else (
                "inf" if r["kappa"] < 0 else ("%g" % r["kappa"]))
            al = "-" if r["alpha"] is None else ("%g" % r["alpha"])
            print(f"{r['rule']:11s}{kap:>5}{al:>6} {r['wname']:24s} | "
                  f"{r['lovo'][0.005]:.3f} {r['lovo'][0.01]:.3f} {r['lovo'][0.05]:.3f} "
                  f" {r['top1']:.3f}")
        for lbl, pred in (("best geo", lambda r: r["rule"] == "geo" and not r["temp"]),
                          ("best arith", lambda r: r["rule"] == "arith" and not r["temp"]),
                          ("best kap=0", lambda r: r["kappa"] == 0.0),
                          ("best kap=inf", lambda r: r["kappa"] == -1.0)):
            r = _first(pred)
            if r is not None:
                kap = "inf" if r["kappa"] == -1.0 else r["kappa"]
                print(f"{lbl:17s} kap={str(kap):>4} {r['wname']:24s} | "
                      f"{r['lovo'][0.005]:.3f} {r['lovo'][0.01]:.3f} {r['lovo'][0.05]:.3f} "
                      f" {r['top1']:.3f}")
        spectrum = {}
        for kap in KAPPAS:
            for rule in (None, "gcfisher", "gcfisher_a"):
                kk = _first(lambda r, k=kap, ru=rule: r["kappa"] == k
                            and (ru is None or r["rule"] == ru))
                if kk is not None:
                    spectrum[f"{kap}|{rule or 'any'}"] = {
                        "alpha": kk["alpha"], "wname": kk["wname"], "lovo": kk["lovo"]}

        kc_stats = {}
        for n in members:
            Iref = srcm(n, "val")[1]
            pool = np.sort(Iref.ravel())
            k_c = (Iref > np.quantile(pool, 0.99)).sum(0)
            kc_stats[n] = {"median": float(np.median(k_c)), "mean": float(k_c.mean()),
                           "p10": float(np.percentile(k_c, 10)),
                           "frac_lt2": float((k_c < 2).mean()),
                           "frac_lt5": float((k_c < 5).mean())}
            print(f"[k_c@Q99] {n:24s} median={kc_stats[n]['median']:.0f} "
                  f"mean={kc_stats[n]['mean']:.0f} p10={kc_stats[n]['p10']:.0f} "
                  f"<2:{kc_stats[n]['frac_lt2'] * 100:.0f}% "
                  f"<5:{kc_stats[n]['frac_lt5'] * 100:.0f}%")

        # test finals at the LOVO-selected configs (full-val calibration)
        def full_pop_p(n):
            Iref = srcm(n, "val")[1]
            tf = _tf_of(n)
            pool = np.sort(Iref.ravel())
            beta = _beta_td(tf(pool), _floor_of(n))
            out = [_ecdf_p(pool, M.ravel(), False, tf, beta).reshape(M.shape)
                   for M in srcm(n, "test")]
            return out, (Iref > np.quantile(pool, 0.99)).sum(0).astype(np.float64)

        FULLP = {n: full_pop_p(n) for n in members}

        def shrunk_test_m(kap, w, alpha):
            prs = [(n, u) for n, u in zip(members, wnrm(list(w))) if u > 0]
            Sp = Si = None
            aux_p = aux_i = 0.0
            for n, u in prs:
                r = np.sort(srcm(n, "val")[1], axis=0)
                tf = _tf_of(n)
                beta = SPACE[n][1]
                Pq, Iq = srcm(n, "test")
                pe_p = np.empty(Pq.shape, dtype=np.float64)
                pe_i = np.empty(Iq.shape, dtype=np.float64)
                for j in range(Pq.shape[1]):
                    rj = r[:, j]
                    pe_p[:, j] = _ecdf_p(rj, Pq[:, j], False, tf, beta)
                    pe_i[:, j] = _ecdf_p(rj, Iq[:, j], False, tf, beta)
                pp_p, pp_i = FULLP[n][0]
                k_c = FULLP[n][1]
                if kap == 0.0:
                    mp = (pe_p, pe_i)
                elif kap < 0:
                    mp = (pp_p, pp_i)
                else:
                    wgt = (k_c / (k_c + kap))[None, :]
                    mp = (wgt * pe_p + (1.0 - wgt) * pp_p,
                          wgt * pe_i + (1.0 - wgt) * pp_i)
                Tp = -np.log(mp[0]).astype(np.float32)
                Ti = -np.log(mp[1]).astype(np.float32)
                Sp = u * Tp if Sp is None else Sp + u * Tp
                Si = u * Ti if Si is None else Si + u * Ti
                aux_p = aux_p + u * SEV[(n, "test")][0]
                aux_i = aux_i + u * SEV[(n, "test")][1]
            Pv = combine("geo", pw(list(w)), [raw[(n, "test")][0] for n in names])
            return score_metrics(Pv, y["test"], Sp.max(1) + alpha * aux_p,
                                 Si.max(1) + alpha * aux_i, n_boot=1000)

        picked, seen = [], set()

        def _pick(r):
            if r is None:
                return
            k_ = (r["rule"], r["temp"], r["kappa"], r["alpha"], r["wname"])
            if k_ not in seen:
                seen.add(k_)
                picked.append(r)

        for r in rows[:3]:
            _pick(r)
        _pick(_first(lambda r: r["rule"] == "arith" and not r["temp"]
                     and r["wname"] == "equal"))
        _pick(_first(lambda r: r["rule"] == "geo" and not r["temp"]))
        _pick(_first(lambda r: r["kappa"] == 0.0))
        _pick(_first(lambda r: r["kappa"] == -1.0))
        _pick(_first(lambda r: r["rule"] == "gcfisher"))

        print("\n--- P1 LOVO TEST (full-val calibration at selected configs) ---")
        lovo_finals = []
        for r in picked:
            if r["rule"] in ("arith", "geo", "min"):
                wv = list(r["w"])
                Pv = combine(r["rule"], pw(wv),
                             [scaled(n, "test", r["temp"])[0] for n in names])
                Iv = combine(r["rule"], pw(wv),
                             [scaled(n, "test", r["temp"])[1] for n in names])
                m = open_set_metrics(Pv, y["test"], Iv.max(1), n_boot=1000)
            else:
                m = shrunk_test_m(r["kappa"], r["w"], r["alpha"] or 0.0)
            m["fpr"] = {str(k): v for k, v in m["fpr"].items()}
            kap = "inf" if r["kappa"] == -1.0 else r["kappa"]
            tag = (f"{r['rule']}|T={r['temp']}|{r['wname']}"
                   + (f"|kap={kap}|a={r['alpha']}" if r["kappa"] is not None else ""))
            print(f"{tag:52s} top1={m['closed']['top1']:.3f} "
                  f"@.5%={m['fpr']['0.005']['tpr']:.3f} "
                  f"@1%={m['fpr']['0.01']['tpr']:.3f} "
                  f"@5%={m['fpr']['0.05']['tpr']:.3f} "
                  f"AUROC={m['auroc_detection']:.3f}")
            lovo_finals.append({"config": tag, "lovo": r["lovo"], "test": m})

        lovo_report = {
            "folds": {"videos": [len(S) for S in fold_sets],
                      "sessions": [int((row_fold == g).sum()) for g in range(nf)],
                      "seed": 0},
            "kappas": list(KAPPAS), "alphas": list(ALPHAS),
            "k_c_stats": kc_stats,
            "spectrum": spectrum,
            "top15": [{k_: r[k_] for k_ in ("rule", "temp", "kappa", "alpha",
                                            "wname", "w", "top1", "lovo")}
                      for r in rows[:15]],
            "finals": lovo_finals,
        }

    variants.sort(key=lambda v: (-v["val"][0.005], -v["val"][0.01], -v["val"]["top1"]))
    print(f"\n{'rule':10s} {'T':5s} {'weights':30s} | val @.5%  @1%   @5%   top1")
    for v in variants[:12]:
        print(f"{v['rule']:10s} {str(v['temp']):5s} {v['wname']:30s} | "
              f"{v['val'][0.005]:.3f}  {v['val'][0.01]:.3f}  {v['val'][0.05]:.3f}  "
              f"{v['val']['top1']:.3f}")

    # test report: top-3 val configs + equal-weight arith baseline
    print("\n--- TEST ---")
    keys, finals = [], []
    baselines = [v for v in variants if not v["temp"] and v["rule"] == "arith"
                 and v["wname"] == "equal"]
    extra = []
    for v in variants:  # always also report best plain-geo and best p-space config
        if not v["temp"] and v["rule"] == "geo":
            extra.append(v)
            break
    for v in variants:
        if v["rule"][0] == "p":
            extra.append(v)
            break
    for v in variants:
        if v["rule"][0] == "c":
            extra.append(v)
            break
    for v in variants:
        if v["rule"] == "gcfisher_a":
            extra.append(v)
            break
    ordered = variants[:3] + baselines + extra
    for v in ordered:
        key = (v["rule"], v["temp"], v["wname"], v.get("alpha"))
        if key in keys:
            continue
        keys.append(key)
        if v["rule"] == "gcfisher_a":
            wn = wnrm(v["w"])
            Tp, Ti = combine_cmat("cfisher", v["w"], "test")
            aux_p = sum(u * SEV[(n, "test")][0] for n, u in zip(members, wn) if u > 0)
            aux_i = sum(u * SEV[(n, "test")][1] for n, u in zip(members, wn) if u > 0)
            Pv = combine("geo", pw(v["w"]), [raw[(n, "test")][0] for n in names])
            m = score_metrics(Pv, y["test"], Tp.max(1) + v["alpha"] * aux_p,
                              Ti.max(1) + v["alpha"] * aux_i, n_boot=1000)
        elif v["rule"] == "gcfisher":
            Tp, Ti = combine_cmat("cfisher", v["w"], "test")
            Pv = combine("geo", pw(v["w"]), [raw[(n, "test")][0] for n in names])
            m = score_metrics(Pv, y["test"], Tp.max(1), Ti.max(1), n_boot=1000)
        elif v["rule"] in ("cfisher", "cpmean"):
            Tp, Ti = combine_cmat(v["rule"], v["w"], "test")
            m = open_set_metrics(Tp, y["test"], Ti.max(1), n_boot=1000)
        elif v["rule"].startswith("p"):
            Pv = combine("geo", pw(v["w"]), [raw[(n, "test")][0] for n in names])
            sp, si = combine_pscore(v["rule"], pw(v["w"]), "test")
            m = score_metrics(Pv, y["test"], sp, si, n_boot=1000)
        else:
            Pv = combine(v["rule"], pw(v["w"]),
                         [scaled(n, "test", v["temp"])[0] for n in names])
            Iv = combine(v["rule"], pw(v["w"]),
                         [scaled(n, "test", v["temp"])[1] for n in names])
            m = open_set_metrics(Pv, y["test"], Iv.max(1), n_boot=1000)
        m["fpr"] = {str(k): v for k, v in m["fpr"].items()}
        tag = f"{v['rule']}|T={v['temp']}|{v['wname']}" + \
              (f"|a={v['alpha']}" if "alpha" in v else "")
        print(f"{tag:40s} top1={m['closed']['top1']:.3f} "
              f"@.5%={m['fpr']['0.005']['tpr']:.3f} "
              f"@1%={m['fpr']['0.01']['tpr']:.3f} @5%={m['fpr']['0.05']['tpr']:.3f} "
              f"AUROC={m['auroc_detection']:.3f}")
        finals.append({"config": tag, "val": v["val"], "test": m})

    out = PROJECT_ROOT / "runs" / (args.out or f"combo_{args.dataset}")
    out.mkdir(parents=True, exist_ok=True)
    rep = {"members": names, "temps": temps,
           "families": FAM, "pval_ref": {n: len(REF[n]) for n in names},
           "top_variants": variants[:12], "finals": finals}
    if lovo_report is not None:
        rep["lovo"] = lovo_report
    (out / "report.json").write_text(json.dumps(rep, indent=1, default=str),
                                     encoding="utf-8")
    print(f"[wrote] {out / 'report.json'}")


if __name__ == "__main__":
    main()
