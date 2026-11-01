# -*- coding: utf-8 -*-
"""T0 measurement foundation + M2 magnet dissection for the STRICT protocol.

Zero-GPU post-processing over cached member probabilities. Opponents: geo
weighted (.3/.35/.35) and arith equal, the P0 strict baselines.

T0 -- honest measurement:
  * paired two-sided VIDEO-CLUSTER bootstrap (385 probe videos + 129 impostor
    videos resampled as whole clusters; both rules share every replicate ->
    paired dTPR), vs the session-iid bootstrap the old CI used (claimed ~1.7x
    too narrow);
  * TPR decomposition: TPR = top1 x TPR|correct (makes the closed-set ceiling
    visible);
  * dual-threshold calibers: oracle (threshold from test impostors, ROC
    convention) vs deployment (threshold from val impostors) -> realized test
    FPR + TPR under the deployment threshold;
  * per-video TPR buckets: score-limited vs closed-set-limited probe videos;
  * magnet census: distinct videos (and effective count 1/sum p_i^2) behind
    the top-k impostor events at each FPR tier, test AND val side; plus
    leave-one-magnet-video-out sensitivity of the 0.5% threshold/TPR.

M2 -- magnet anatomy (arbitrates aux-veto / disagreement-gate / ArcFace /
synthetic-family proposals):
  * member-argmax agreement distribution of magnet sessions vs random
    impostors vs correct probes ("consistent" vs "disagreeing-but-confident");
  * per-member membership of the tail (is the tail driven by one member?);
  * which known classes the magnets impersonate.
"""
import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import PROJECT_ROOT, RunConfig
from src.data import load_splits, session_pool
from src.evaluate import _threshold_for_fpr

GRID = (0.005, 0.01, 0.05)
W_GEO = (0.30, 0.35, 0.35)
TAGS = ("dfs_branch_all_e100", "dfs_branch_all_dt")


def norm(P):
    return P / (P.sum(1, keepdims=True) + 1e-12)


def geo(ws, mats):
    L = sum(w * np.log(np.clip(P, 1e-12, 1)) for w, P in zip(ws, mats))
    L -= L.max(1, keepdims=True)
    E = np.exp(L)
    return E / E.sum(1, keepdims=True)


def arith(ws, mats):
    return sum(w * P for w, P in zip(ws, mats))


def tpr_at(correct, sp, si, f):
    th = _threshold_for_fpr(si, f)
    return float(np.mean(correct & (sp >= th))), th


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="10s")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--out", default="strict_diag")
    args = ap.parse_args()

    rcfg = RunConfig(dataset=args.dataset, protocol="strict")
    splits = load_splits(rcfg.splits_path)
    known = sorted(splits["known_videos"])
    vidx = {v: i for i, v in enumerate(known)}

    pools = {k: session_pool(splits, *a) for k, a in dict(
        val_probe=("known", "val", False), val_imp=("unknown", "val"),
        test_probe=("known", "test", False), test_imp=("unknown", "test")).items()}
    vid = {k: np.array([splits["sessions"][s]["video"] for s in v])
           for k, v in pools.items()}

    z = np.load(PROJECT_ROOT / "runs" / "cache" / f"lgbm_probs_{args.dataset}_strict.npz")
    mem = {"lgbm": {s: (norm(z[f"{s}_probe"]), norm(z[f"{s}_imp"]))
                    for s in ("val", "test")}}
    y = None
    for t in TAGS:
        d = {s: np.load(PROJECT_ROOT / "runs" / t / f"probs_{s}.npz")
             for s in ("val", "test")}
        if y is None:
            y = {s: d[s]["probe_y"] for s in ("val", "test")}
        else:
            assert all((d[s]["probe_y"] == y[s]).all() for s in ("val", "test"))
        assert all(len(pools[f"{s}_imp"]) == d[s]["imp"].shape[0]
                   for s in ("val", "test")), f"{t} impostor row mismatch"
        mem[t] = {s: (norm(d[s]["probe"]), norm(d[s]["imp"])) for s in ("val", "test")}

    names = ["lgbm"] + list(TAGS)
    rules = {
        "geo": lambda s: geo(W_GEO, [mem[n][s][0] for n in names]),
        "arith": lambda s: arith([1 / 3] * 3, [mem[n][s][0] for n in names]),
    }
    imp_rules = {
        "geo": lambda s: geo(W_GEO, [mem[n][s][1] for n in names]),
        "arith": lambda s: arith([1 / 3] * 3, [mem[n][s][1] for n in names]),
    }

    rep = {"config": {"dataset": args.dataset, "protocol": "strict",
                      "w_geo": list(W_GEO), "n_boot": args.n_boot,
                      "pools": {k: len(v) for k, v in pools.items()},
                      "videos": {k: len(set(v.tolist())) for k, v in vid.items()}}}

    # ---- sanity anchors + TPR decomposition + dual calibers (test) -------
    print("\n=== T0: test metrics (oracle caliber) ===")
    det = {}
    for rn, f in rules.items():
        P = f("test")
        I = imp_rules[rn]("test")
        sp, si = P.max(1), I.max(1)
        correct = P.argmax(1) == y["test"]
        row = {"top1": float(np.mean(correct)), "fpr": {}}
        for fr in GRID:
            t, th = tpr_at(correct, sp, si, fr)
            t_det, _ = tpr_at(np.ones(sp[correct].shape, bool), sp[correct], si, fr)
            row["fpr"][str(fr)] = {"tpr": t, "tpr_given_correct": t_det,
                                   "threshold": th}
        det[rn] = row
        print(f"{rn:6s} top1={row['top1']:.3f} " + " ".join(
            f"@{fr:g}%={row['fpr'][str(fr)]['tpr']:.3f}"
            f"(det={row['fpr'][str(fr)]['tpr_given_correct']:.3f})" for fr in GRID))
    rep["test_oracle"] = det

    print("\n=== T0: deployment caliber (val-side threshold) ===")
    dep = {}
    for rn in rules:
        sp = rules[rn]("test").max(1)
        I = imp_rules[rn]("test").max(1)
        Iv = imp_rules[rn]("val").max(1)
        correct = rules[rn]("test").argmax(1) == y["test"]
        row = {}
        for fr in GRID:
            th = _threshold_for_fpr(Iv, fr)
            row[str(fr)] = {"realized_fpr": float(np.mean(I >= th)),
                            "tpr": float(np.mean(correct & (sp >= th)))}
        dep[rn] = row
        print(f"{rn:6s} " + " ".join(
            f"@{fr:g}%: FPR_real={row[str(fr)]['realized_fpr']:.4f} "
            f"TPR={row[str(fr)]['tpr']:.3f}" for fr in GRID))
    rep["test_deployment"] = dep

    # ---- paired video-cluster bootstrap vs session-iid bootstrap ----------
    print(f"\n=== T0: bootstrap ({args.n_boot} reps, paired) ===")
    pv = sorted(set(vid["test_probe"].tolist()))
    iv = sorted(set(vid["test_imp"].tolist()))
    pg = [np.where(vid["test_probe"] == v)[0] for v in pv]
    ig = [np.where(vid["test_imp"] == v)[0] for v in iv]
    P = {rn: rules[rn]("test") for rn in rules}
    SP = {rn: P[rn].max(1) for rn in rules}
    CO = {rn: P[rn].argmax(1) == y["test"] for rn in rules}
    SI = {rn: imp_rules[rn]("test").max(1) for rn in rules}
    npp, nip = len(vid["test_probe"]), len(vid["test_imp"])

    rng = np.random.default_rng(0)
    boot = {"cluster": {rn: {str(fr): [] for fr in GRID} for rn in rules},
            "session": {rn: {str(fr): [] for fr in GRID} for rn in rules},
            "dTPR_cluster": {str(fr): [] for fr in GRID}}
    for _ in range(args.n_boot):
        ci = np.concatenate([ig[j] for j in rng.integers(0, len(iv), len(iv))])
        cpi = np.concatenate([pg[j] for j in rng.integers(0, len(pv), len(pv))])
        si = rng.integers(0, nip, nip)
        spi = rng.integers(0, npp, npp)
        for fr in GRID:
            for rn in rules:  # paired = same resampled clusters, each rule
                # calibrates its threshold on ITS OWN impostor scores
                th = _threshold_for_fpr(SI[rn][ci], fr)
                ths = _threshold_for_fpr(SI[rn][si], fr)
                boot["cluster"][rn][str(fr)].append(
                    float(np.mean(CO[rn][cpi] & (SP[rn][cpi] >= th))))
                boot["session"][rn][str(fr)].append(
                    float(np.mean(CO[rn][spi] & (SP[rn][spi] >= ths))))
            thg = _threshold_for_fpr(SI["geo"][ci], fr)
            tha = _threshold_for_fpr(SI["arith"][ci], fr)
            boot["dTPR_cluster"][str(fr)].append(
                float(np.mean(CO["geo"][cpi] & (SP["geo"][cpi] >= thg))
                      - np.mean(CO["arith"][cpi] & (SP["arith"][cpi] >= tha))))
    bs = {}
    for kind in ("cluster", "session"):
        bs[kind] = {rn: {str(fr): [float(np.percentile(boot[kind][rn][str(fr)], 2.5)),
                                   float(np.percentile(boot[kind][rn][str(fr)], 97.5))]
                         for fr in GRID} for rn in rules}
    bs["dTPR_cluster"] = {str(fr): [float(np.percentile(boot["dTPR_cluster"][str(fr)], 2.5)),
                                    float(np.percentile(boot["dTPR_cluster"][str(fr)], 97.5))]
                          for fr in GRID}
    rep["bootstrap"] = bs
    for rn in rules:
        c = bs["cluster"][rn]["0.005"]
        s = bs["session"][rn]["0.005"]
        print(f"{rn:6s} @.5% cluster CI [{c[0]:.3f},{c[1]:.3f}] (w={c[1]-c[0]:.3f}) | "
              f"session CI [{s[0]:.3f},{s[1]:.3f}] (w={s[1]-s[0]:.3f}) "
              f"narrow-by {(c[1]-c[0]) / max(s[1]-s[0], 1e-9):.2f}x")
    d = bs["dTPR_cluster"]["0.005"]
    print(f"paired dTPR(geo-arith) @.5% CI [{d[0]:+.3f},{d[1]:+.3f}] "
          f"{'EXCLUDES 0' if d[0] > 0 or d[1] < 0 else 'includes 0 (not decidable)'}")

    # ---- per-video TPR buckets (geo, 0.5% oracle) -------------------------
    th5 = det["geo"]["fpr"]["0.005"]["threshold"]
    ok5 = CO["geo"] & (SP["geo"] >= th5)
    buckets = Counter()
    for v in pv:
        m_ = vid["test_probe"] == v
        buckets[(bool(CO["geo"][m_].all()), bool(ok5[m_].any()))] += 1
    rep["probe_video_buckets"] = {f"closed_ok={a},any_tpr={b}": int(n)
                                  for (a, b), n in buckets.items()}
    print("\n=== T0: per-video TPR buckets @.5% (geo) ===")
    for k, n in sorted(rep["probe_video_buckets"].items(), reverse=True):
        print(f"  {k}: {n} videos")

    # ---- magnet census -----------------------------------------------------
    print("\n=== M2: magnet census (geo combined score) ===")
    mag = {}
    for side, si_key in (("test", "test_imp"), ("val", "val_imp")):
        I = imp_rules["geo"](side).max(1)
        vids_side = vid[si_key]
        order = np.argsort(I)[::-1]
        row = {}
        for fr in GRID:
            k = int(np.floor(fr * len(I)))
            top = order[:max(k, 1)]
            shares = Counter(vids_side[top].tolist())
            p = np.array(list(shares.values()), float) / max(k, 1)
            row[str(fr)] = {"events": k, "distinct_videos": len(shares),
                            "eff_videos": float(1.0 / np.sum(p ** 2)),
                            "top": [(v, int(c), round(float(I[np.where(vids_side == v)[0]].max()), 3))
                                    for v, c in shares.most_common(8)]}
        mag[side] = row
        for fr in GRID:
            r = row[str(fr)]
            print(f"{side:4s} @{fr:g}%: {r['events']} events from {r['distinct_videos']} "
                  f"videos (eff {r['eff_videos']:.1f}) top: "
                  + ", ".join(f"{v.split('_')[-1]}x{c}" for v, c, _ in r["top"][:6]))
    rep["magnets"] = mag

    # leave-one-magnet-video-out sensitivity of the 0.5% tier (test)
    I = imp_rules["geo"]("test").max(1)
    vids_side = vid["test_imp"]
    k = int(0.005 * len(I))
    top = np.argsort(I)[::-1][:k]
    shares = Counter(vids_side[top].tolist())
    lomo = []
    for v, _ in shares.most_common(6):
        keep = vids_side != v
        t, th = tpr_at(CO["geo"], SP["geo"], I[keep], 0.005)
        lomo.append({"video": v, "tpr_excl": t})
    rep["lomo_magnet_0.5"] = lomo
    print("LOMO @.5%: " + " ".join(f"-{l['video'].split('_')[-1]}:{l['tpr_excl']:.3f}"
                                  for l in lomo))

    # ---- M2: member-consistency of the tail --------------------------------
    print("\n=== M2: member argmax agreement (geo tail) ===")
    arg = {n: {s: mem[n][s][0].argmax(1) for s in ("val", "test")} for n in names}
    argi = {n: {s: mem[n][s][1].argmax(1) for s in ("val", "test")} for n in names}

    def agree3(rows, s, imp=False):
        a = {n: (argi if imp else arg)[n][s][rows] for n in names}
        cnt = sum((a["lgbm"] == a[n]) for n in TAGS) + 1  # 1..3 (lgbm as ref)
        return cnt

    def agree_dist(rows, s, imp):
        c = agree3(rows, s, imp)
        return {str(k): int((c == k).sum()) for k in (1, 2, 3)}

    order = np.argsort(I)[::-1]
    agree = {"top19": agree_dist(order[:19], "test", True),
             "top100": agree_dist(order[:100], "test", True),
             "all_impostors": agree_dist(np.arange(len(I)), "test", True),
             "correct_probes": agree_dist(np.where(CO["geo"])[0], "test", False)}
    rep["agreement"] = agree
    for k, v in agree.items():
        n = sum(v.values())
        print(f"{k:16s} n={n:5d} 3-agg={v['3'] / n * 100:4.1f}% "
              f"2-agg={v['2'] / n * 100:4.1f}% 1-agg={v['1'] / n * 100:4.1f}%")

    # which member is confident when the tail disagrees (top-100)
    conf = defaultdict(int)
    confv = {n: mem[n]["test"][1].max(1) for n in names}
    for r in order[:100]:
        for n in names:
            if confv[n][r] > 0.5:
                conf[n] += 1
    rep["tail_confidence_driver"] = dict(conf)
    print("tail top-100 members with maxprob>.5: " +
          ", ".join(f"{n}:{c}" for n, c in conf.items()))

    # which known classes the magnets impersonate (top-19)
    cls = Counter(argi["lgbm"]["test"][order[:19]].tolist())
    rep["magnet_classes_top19"] = [
        {"class_idx": int(c), "count": int(n)} for c, n in cls.most_common()]
    print("magnet top-19 impersonated classes (lgbm argmax): " +
          ", ".join(f"{c}x{n}" for c, n in cls.most_common(8)))

    out = PROJECT_ROOT / "runs" / args.out
    out.mkdir(parents=True, exist_ok=True)
    (out / "report.json").write_text(json.dumps(rep, indent=1, default=str),
                                     encoding="utf-8")
    print(f"\n[wrote] {out / 'report.json'}")


if __name__ == "__main__":
    main()
