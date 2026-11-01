# -*- coding: utf-8 -*-
"""T1b acceptance analysis: pooled LOVO for the OE ensemble vs frozen baseline.

Ensemble = 3-member geo (lgbm, dfs_branch_all_e100, dt-slot) with the fixed
strict weights (.3/.35/.35).  The dt slot is either the frozen member
(baseline, pooled LOVO .696) or the OE fold models (fold f's model scores
fold f's impostors + all val probes; the OTHER members are frozen, so only
the dt slot changes per fold).  Also reports the probe-side score guard and
per-magnet-video fold-out deltas.
"""
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import PROJECT_ROOT, RunConfig
from src.data import load_splits, session_pool
from src.evaluate import _threshold_for_fpr

GRID = (0.005, 0.01, 0.05)
W = (0.30, 0.35, 0.35)
FOLDS = 5


def norm(P):
    return P / (P.sum(1, keepdims=True) + 1e-12)


def geo(mats):
    L = sum(w * np.log(np.clip(P, 1e-12, 1)) for w, P in zip(W, mats))
    L -= L.max(1, keepdims=True)
    E = np.exp(L)
    return E / E.sum(1, keepdims=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--oe-dir", required=True,
                    help="e.g. runs/dfs_dt_oe (contains lam*/fold*.npz)")
    ap.add_argument("--dataset", default="10s")
    args = ap.parse_args()
    oe_root = PROJECT_ROOT / args.oe_dir

    rcfg = RunConfig(dataset=args.dataset, protocol="strict")
    splits = load_splits(rcfg.splits_path)
    val_imp = session_pool(splits, "unknown", "val")
    vid_row = np.array([splits["sessions"][s]["video"] for s in val_imp])
    uvid = np.array(sorted(set(vid_row.tolist())))
    perm = np.random.default_rng(0).permutation(len(uvid))
    fold_sets = [set(uvid[perm[f::FOLDS]].tolist()) for f in range(FOLDS)]
    v2f = {v: f for f, S in enumerate(fold_sets) for v in S}
    row_fold = np.array([v2f[v] for v in vid_row])

    z = np.load(PROJECT_ROOT / "runs" / "cache" / f"lgbm_probs_{args.dataset}_strict.npz")
    lg = {s: (norm(z[f"{s}_probe"]), norm(z[f"{s}_imp"])) for s in ("val",)}
    e1 = {s: (norm(np.load(PROJECT_ROOT / "runs" / "dfs_branch_all_e100"
                           / f"probs_{s}.npz")["probe"]),
              norm(np.load(PROJECT_ROOT / "runs" / "dfs_branch_all_e100"
                           / f"probs_{s}.npz")["imp"])) for s in ("val",)}
    dt = {s: (norm(np.load(PROJECT_ROOT / "runs" / "dfs_branch_all_dt"
                           / f"probs_{s}.npz")["probe"]),
              norm(np.load(PROJECT_ROOT / "runs" / "dfs_branch_all_dt"
                           / f"probs_{s}.npz")["imp"])) for s in ("val",)}
    y = dt["val"][0].shape[0]  # probe rows; labels from first fold npz

    def pooled(Pp, Pi):
        SP, SI, CO = [], [], []
        for f in range(FOLDS):
            rows_f = np.where(row_fold == f)[0]   # fold rows in pool order
            G = geo([lg["val"][0], e1["val"][0], Pp[f]])
            sp = G.max(1)
            co = G.argmax(1) == _y
            I = geo([lg["val"][1][rows_f], e1["val"][1][rows_f], Pi[f]])
            SP.append(sp)
            CO.append(co)
            SI.append(I.max(1))
        SP, SI = np.concatenate(SP), np.concatenate(SI)
        CO = np.concatenate(CO)
        out = {"top1": float(np.mean(CO))}
        for fr in GRID:
            th = _threshold_for_fpr(SI, fr)
            out[fr] = float(np.mean(CO & (SP >= th)))
        return out

    # labels
    f0 = np.load(next(oe_root.glob("lam*/fold0.npz")))
    _y = f0["probe_y"]

    # baseline: frozen dt everywhere
    dt_imp_folds = [dt["val"][1][np.where(row_fold == f)[0]] for f in range(FOLDS)]
    base = pooled([dt["val"][0] for _ in range(FOLDS)], dt_imp_folds)
    print(f"baseline (frozen dt) pooled LOVO: "
          f"@.5%={base[0.005]:.3f} @1%={base[0.01]:.3f} @5%={base[0.05]:.3f} "
          f"top1={base['top1']:.3f}")

    rep = {"baseline": base, "lams": {}}
    for lam_dir in sorted(oe_root.glob("lam*")):
        fold_files = {int(p.stem[4:]): p for p in lam_dir.glob("fold*.npz")}
        if len(fold_files) < FOLDS:
            print(f"{lam_dir.name}: {len(fold_files)}/{FOLDS} folds -- skipped")
            continue
        Pp = [norm(np.load(fold_files[f])["probe"]) for f in range(FOLDS)]
        Pi = [norm(np.load(fold_files[f])["imp"]) for f in range(FOLDS)]
        m = pooled(Pp, Pi)
        # probe-side guard: OE member probe maxprob quantiles vs frozen
        pq = [float(np.quantile(P.max(1), q)) for P in Pp for q in (.5, .99)]
        rep["lams"][lam_dir.name] = {"pooled": m, "probe_q": pq}
        print(f"{lam_dir.name}: @.5%={m[0.005]:.3f} @1%={m[0.01]:.3f} "
              f"@5%={m[0.05]:.3f} top1={m['top1']:.3f} | probe maxprob "
              f"q50={pq[0]:.3f} q99={pq[1]:.3f} (frozen "
              f"{np.quantile(dt['val'][0].max(1), .5):.3f}/"
              f"{np.quantile(dt['val'][0].max(1), .99):.3f})")

        # per-magnet fold-out delta (ensemble score, fold-out rows only)
        mag = defaultdict(list)
        for f in range(FOLDS):
            rows_f2 = np.where(row_fold == f)[0]
            Gi = geo([lg["val"][1][rows_f2], e1["val"][1][rows_f2], Pi[f]])
            Gf = geo([lg["val"][1][rows_f2], e1["val"][1][rows_f2],
                      dt["val"][1][rows_f2]])
            rows = rows_f2
            vids_f = vid_row[rows_f2]
            for v in sorted(set(vids_f.tolist())):
                s_new = float(Gi.max(1)[vids_f == v].max())
                s_old = float(Gf.max(1)[vids_f == v].max())
                mag[v].append((s_old, s_new))
        top_v = sorted(mag.items(), key=lambda kv: -kv[1][0][0])[:8]
        print("  fold-out magnet videos (old->new max ensemble score): " +
              ", ".join(f"{v.split('_')[-1][:8]}:{o:.2f}->{n:.2f}"
                        for v, d in top_v for o, n in d[:1]))
        rep["lams"][lam_dir.name]["magnets"] = {v: d for v, d in top_v}

    out = PROJECT_ROOT / "runs" / oe_root.name
    (out / "pooled_analysis.json").write_text(json.dumps(rep, indent=1,
                                                         default=str),
                                              encoding="utf-8")
    print(f"[wrote] {out / 'pooled_analysis.json'}")


if __name__ == "__main__":
    main()
