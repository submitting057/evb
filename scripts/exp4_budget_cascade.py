# -*- coding: utf-8 -*-
"""Exp4: observation-budget cascade — 5s ensemble as stage-1 early-exit /
escalation gate, 10s ensemble as stage-2 for uncertain sessions only.

Decision rule after 5s of observation (max5 = 5s-ensemble best score):
  max5 >= tau_hi -> decide NOW with the 5s model (identification = argmax5)
  max5 <  tau_lo -> reject as non-target NOW (no 10s observation)
  else           -> continue to 10s; accept iff max10 >= theta (argmax10)

Sessions are matched by session_id across the two datasets. Only sessions
with honest cached 5s probs are used (5s split in {val, test}; 5s-train
sessions are excluded — the 5s model trained on them). Thresholds are fit
on the 10s-val matched pool and applied unchanged to 10s-test.

Baselines on the identical matched test subset: 5s-only / 10s-only ensemble
with thresholds from matched val impostors (deployment-honest) and from
matched test impostors (protocol style, as in open_set_metrics).

Ablation: stage-2 masked to the 5s top-24 shortlist (renormalized).

Usage: python scripts/exp4_budget_cascade.py
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import PROJECT_ROOT, DATASET_DIRS
from src.data import load_splits, session_pool

TARGETS = (0.005, 0.01, 0.05)
N_HI, N_LO, N_TH = 26, 26, 31
ESC_CAPS = (1.0, 0.8, 0.6, 0.4, 0.2, 0.1)
MASK_K = 24


def ens_probs(dataset: str, split: str, kind: str) -> np.ndarray:
    """Mean-prob ensemble exactly as run_ensemble.py: (LGBM + mean(DFs)) / 2."""
    tags = {"10s": ["df_branch_all_e100", "df_branch_all_dt"],
            "5s": ["df5_branch_all_e100"]}[dataset]
    zl = np.load(PROJECT_ROOT / "runs" / "cache" / f"lgbm_probs_{dataset}.npz")
    pl = zl[f"{split}_{kind}"]
    pl = pl / pl.sum(1, keepdims=True)
    dfs = []
    for t in tags:
        z = np.load(PROJECT_ROOT / "runs" / t / f"probs_{split}.npz")
        p = z["probe" if kind == "probe" else "imp"]
        dfs.append(p / p.sum(1, keepdims=True))
    return (pl + np.mean(dfs, 0)) / 2


def manifest_costs():
    costs = {}
    for ds in ("10s", "5s"):
        d = {}
        with open(DATASET_DIRS[ds] / "manifests" / "sessions.jsonl", encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                d[r["session_id"]] = float(r.get("video_duration_s") or 0.0)
        costs[ds] = d
    return costs


def match_pool(sp10, sp5, split, role, kind):
    """10s pool order -> (matched sids, row index into the 5s cached arrays)."""
    pool10 = (session_pool(sp10, "known" if kind == "probe" else "unknown", split,
                           gallery=False if kind == "probe" else None))
    rows = {}
    for s5 in ("val", "test"):
        p5 = session_pool(sp5, "known" if kind == "probe" else "unknown", s5,
                          gallery=False if kind == "probe" else None)
        for i, sid in enumerate(p5):
            if sid not in rows:
                rows[sid] = (s5, i)
    out_sids, out_rows = [], []
    for sid in pool10:
        if sid in rows:
            out_sids.append(sid)
            out_rows.append(rows[sid])
    return out_sids, out_rows


def main() -> None:
    sp10 = load_splits(PROJECT_ROOT / "splits" / "splits_10s_s42.json")
    sp5 = load_splits(PROJECT_ROOT / "splits" / "splits_5s_s42.json")
    known = sorted(sp10["known_videos"])
    vidx = {v: i for i, v in enumerate(known)}

    costs = manifest_costs()

    data = {}
    for split in ("val", "test"):
        sids_p, rows5_p = match_pool(sp10, sp5, split, "known", "probe")
        sids_i, rows5_i = match_pool(sp10, sp5, split, "unknown", "imp")
        # 10s rows follow session_pool order of sp10
        pool10p = session_pool(sp10, "known", split, gallery=False)
        pool10i = session_pool(sp10, "unknown", split)
        pos10p = {s: i for i, s in enumerate(pool10p)}
        pos10i = {s: i for i, s in enumerate(pool10i)}

        p10_p = ens_probs("10s", split, "probe")[[pos10p[s] for s in sids_p]]
        p10_i = ens_probs("10s", split, "imp")[[pos10i[s] for s in sids_i]]
        p5_p = np.stack([ens_probs("5s", s5, "probe")[i] for s5, i in rows5_p])
        p5_i = np.stack([ens_probs("5s", s5, "imp")[i] for s5, i in rows5_i])

        y = np.array([vidx[sp10["sessions"][s]["video"]] for s in sids_p], dtype=np.int64)
        c5p = np.array([costs["5s"].get(s, 0) for s in sids_p])
        c10p = np.array([costs["10s"].get(s, 0) for s in sids_p])
        c5i = np.array([costs["5s"].get(s, 0) for s in sids_i])
        c10i = np.array([costs["10s"].get(s, 0) for s in sids_i])
        data[split] = dict(y=y, p5p=p5_p, p5i=p5_i, p10p=p10_p, p10i=p10_i,
                           c5p=c5p, c10p=c10p, c5i=c5i, c10i=c10i,
                           n_probe=len(sids_p), n_imp=len(sids_i))
        print(f"[match] {split}: probes {len(sids_p)}/{len(pool10p)}  "
              f"imps {len(sids_i)}/{len(pool10i)}")

    v = data["val"]
    m5p, a5p = v["p5p"].max(1), v["p5p"].argmax(1)
    m10p, a10p = v["p10p"].max(1), v["p10p"].argmax(1)
    ok5p, ok10p = a5p == v["y"], a10p == v["y"]
    m5i, m10i = v["p5i"].max(1), v["p10i"].max(1)

    t = data["test"]
    tm5p, ta5p = t["p5p"].max(1), t["p5p"].argmax(1)
    tm10p, ta10p = t["p10p"].max(1), t["p10p"].argmax(1)
    tok5p, tok10p = ta5p == t["y"], ta10p == t["y"]
    tm5i, tm10i = t["p5i"].max(1), t["p10i"].max(1)

    # masked (5s top-MASK_K shortlist) stage-2 ablation
    def masked_stats(P, K=MASK_K):
        order = np.argsort(-P, axis=1)[:, :K]
        M = np.zeros_like(P, dtype=bool)
        np.put_along_axis(M, order, True, axis=1)
        Q = np.where(M, P, 0.0)
        s = Q.sum(1, keepdims=True)
        s[s == 0] = 1.0
        return Q / s

    v_m10p, v_m10i = masked_stats(v["p10p"]).max(1), masked_stats(v["p10i"]).max(1)
    t_m10p, t_m10i = masked_stats(t["p10p"]), masked_stats(t["p10i"])
    tm10p_m, ta10p_m = t_m10p.max(1), t_m10p.argmax(1)
    tm10i_m = t_m10i.max(1)
    vma_ok = (v_m10p * 0).astype(bool)  # argmax-correctness under mask (val, unused in fit)

    hi_grid = np.unique(np.quantile(m5p, np.linspace(0.02, 0.98, N_HI)))
    lo_grid = np.unique(np.quantile(m5p, np.linspace(0.0, 0.9, N_LO)))
    th_grid = np.unique(np.quantile(m10i, np.linspace(0.70, 1.0, N_TH)))

    def eval_params(hi, lo, th):
        early_p = m5p >= hi
        esc_p = (~early_p) & (m5p >= lo)
        early_i = m5i >= hi
        esc_i = (~early_i) & (m5i >= lo)
        tpr = float(np.mean((early_p & ok5p) | (esc_p & (m10p >= th) & ok10p)))
        fpr = float(np.mean(early_i | (esc_i & (m10i >= th))))
        esc = float(np.mean(np.concatenate([esc_p, esc_i])))
        return tpr, fpr, esc, early_p, esc_p, early_i, esc_i

    def test_metrics(hi, lo, th, masked=False):
        early_p = tm5p >= hi
        esc_p = (~early_p) & (tm5p >= lo)
        early_i = tm5i >= hi
        esc_i = (~early_i) & (tm5i >= lo)
        if masked:
            acc_p = esc_p & (tm10p_m >= th) & (ta10p_m == t["y"])
            fpr = float(np.mean(early_i | (esc_i & (tm10i_m >= th))))
        else:
            acc_p = esc_p & (tm10p >= th) & tok10p
            fpr = float(np.mean(early_i | (esc_i & (tm10i >= th))))
        tpr = float(np.mean((early_p & tok5p) | acc_p))
        sec_p = float(np.mean(np.where(esc_p, t["c10p"], t["c5p"])))
        sec_i = float(np.mean(np.where(esc_i, t["c10i"], t["c5i"])))
        sec_all = (sec_p * t["n_probe"] + sec_i * t["n_imp"]) / (t["n_probe"] + t["n_imp"])
        sec_full = float((t["c10p"].sum() + t["c10i"].sum()) / (t["n_probe"] + t["n_imp"]))
        sec_5 = float((t["c5p"].sum() + t["c5i"].sum()) / (t["n_probe"] + t["n_imp"]))
        return dict(tpr=tpr, fpr=fpr, esc_p=float(np.mean(esc_p)), esc_i=float(np.mean(esc_i)),
                    mean_s=sec_all, mean_s_probe=sec_p, mean_s_imp=sec_i,
                    full_10s_s=sec_full, only_5s_s=sec_5)

    # ---- fit on val ----
    results = {"grid": {"n_hi": len(hi_grid), "n_lo": len(lo_grid), "n_th": len(th_grid)}}
    cache = []
    for hi in hi_grid:
        for lo in lo_grid:
            if lo > hi:
                continue
            early_i = m5i >= hi
            esc_i = (~early_i) & (m5i >= lo)
            fpr_early = float(np.mean(early_i))
            esc_i_frac = float(np.mean(np.concatenate([(m5p >= lo) & (m5p < hi), esc_i])))
            if fpr_early > max(TARGETS):
                continue
            for th in th_grid:
                tpr, fpr, esc, *_ = eval_params(hi, lo, th)
                cache.append((tpr, fpr, esc, float(hi), float(lo), float(th)))

    print(f"[fit] {len(cache)} param combos evaluated on val")

    def theta_for_budget(hi, lo, tgt, masked=False):
        """Theta so composite VAL FPR <= tgt (exact budget arithmetic, then
        theta keeps only the strongest escalated impostors within budget)."""
        early_i = m5i >= hi
        esc_i = (~early_i) & (m5i >= lo)
        k = int(np.floor(tgt * len(m5i))) - int(early_i.sum())
        if k < 0:
            return np.inf
        scores = (v_m10i if masked else m10i)[esc_i]
        if len(scores) == 0:
            return -np.inf
        s = np.sort(scores)[::-1]
        return float(s[min(k, len(s) - 1)]) + 1e-12

    def bootstrap_ci(hi, lo, th, n_boot=1000, seed=0):
        rng = np.random.default_rng(seed)
        n_p, n_i = len(tm5p), len(tm5i)
        early_p = tm5p >= hi
        esc_p = (~early_p) & (tm5p >= lo)
        early_i = tm5i >= hi
        esc_i = (~early_i) & (tm5i >= lo)
        acc = (early_p & tok5p) | (esc_p & (tm10p >= th) & tok10p)
        fp = early_i | (esc_i & (tm10i >= th))
        tprs, fprs = [], []
        for _ in range(n_boot):
            tprs.append(acc[rng.integers(0, n_p, n_p)].mean())
            fprs.append(fp[rng.integers(0, n_i, n_i)].mean())
        return [float(np.percentile(tprs, 2.5)), float(np.percentile(tprs, 97.5))], \
               [float(np.percentile(fprs, 2.5)), float(np.percentile(fprs, 97.5))]

    out_targets = {}
    for tgt in TARGETS:
        for mode, fit_tgt in (("fit", tgt), ("safe", tgt * 0.7)):
            cand = [c for c in cache if c[1] <= fit_tgt]
            best = max(cand, key=lambda c: (round(c[0], 5), -c[2]))
            _, _, _, hi, lo, th = best
            tm = test_metrics(hi, lo, th)
            # masked ablation with its own theta fitted on val for the same budget
            th_m = theta_for_budget(hi, lo, fit_tgt, masked=True)
            tm_masked = test_metrics(hi, lo, th_m, masked=True)
            ci_tpr, ci_fpr = bootstrap_ci(hi, lo, th)
            rec = dict(tau_hi=hi, tau_lo=lo, theta=th, fit_target=fit_tgt,
                       val_tpr=best[0], val_fpr=best[1], val_esc=best[2],
                       test=tm, test_masked=tm_masked,
                       test_tpr_ci95=ci_tpr, test_fpr_ci95=ci_fpr)
            out_targets[f"{tgt}_{mode}"] = rec
            print(f"\n=== target FPR {tgt*100:g}% [{mode}] (tau_hi={hi:.4f} tau_lo={lo:.4f} theta={th:.4f}) ===")
            print(f"  val : TPR={best[0]:.4f} FPR={best[1]:.4f} esc={best[2]:.3f}")
            print(f"  test: TPR={tm['tpr']:.4f} [{ci_tpr[0]:.3f},{ci_tpr[1]:.3f}] "
                  f"FPR={tm['fpr']:.4f} [{ci_fpr[0]:.3f},{ci_fpr[1]:.3f}] "
                  f"esc(p/i)={tm['esc_p']:.3f}/{tm['esc_i']:.3f} "
                  f"mean_t={tm['mean_s']:.2f}s (5s {tm['only_5s_s']:.2f}s / 10s {tm['full_10s_s']:.2f}s)")
            print(f"  test masked-24 stage-2 (own theta={th_m:.4f}): "
                  f"TPR={tm_masked['tpr']:.4f} FPR={tm_masked['fpr']:.4f}")

    results["targets"] = out_targets

    # ---- budget frontier at 0.5% ----
    frontier = []
    for cap in ESC_CAPS:
        cand = [c for c in cache if c[1] <= 0.005 and c[2] <= cap + 1e-9]
        if not cand:
            frontier.append({"esc_cap": cap, "val_tpr": None})
            continue
        best = max(cand, key=lambda c: c[0])
        _, _, _, hi, lo, th = best
        tm = test_metrics(hi, lo, th)
        frontier.append(dict(esc_cap=cap, tau_hi=hi, tau_lo=lo, theta=th,
                             val_tpr=best[0], val_esc=best[2], test=tm))
        print(f"frontier esc<= {cap:.1f}: val_TPR={best[0]:.4f} test_TPR={tm['tpr']:.4f} "
              f"FPR={tm['fpr']:.4f} mean_t={tm['mean_s']:.2f}s")
    results["frontier_0.5pct"] = frontier

    # ---- baselines on the identical matched test subset ----
    def tpr_at(imp_scores, probe_scores, correct, tgt):
        s = np.sort(imp_scores)[::-1]
        k = int(np.floor(tgt * len(s)))
        th = s[k] + 1e-12
        return float(np.mean(correct & (probe_scores >= th))), th

    base = {}
    vi = np.load(PROJECT_ROOT / "runs" / "cache" / "lgbm_probs_10s.npz")
    for name, pm, im, corr in (("ens10", t["p10p"], t["p10i"], t["p10p"].argmax(1) == t["y"]),
                               ("ens5", t["p5p"], t["p5i"], t["p5p"].argmax(1) == t["y"])):
        top1 = float(np.mean(pm.argmax(1) == t["y"]))
        row = {"top1": top1}
        for tgt in TARGETS:
            # deployment-honest: threshold from matched VAL impostors
            vimp = v["p10i"] if name == "ens10" else v["p5i"]
            vm = vimp.max(1)
            s = np.sort(vm)[::-1]
            k = int(np.floor(tgt * len(s)))
            th = s[k] + 1e-12
            row[f"tpr@{tgt}_valth"] = float(np.mean(corr & (pm.max(1) >= th)))
            # protocol style: threshold from matched TEST impostors
            tpr, thp = tpr_at(im.max(1), pm.max(1), corr, tgt)
            row[f"tpr@{tgt}_testth"] = tpr
        base[name] = row
        print(f"baseline {name} (matched test subset): top1={row['top1']:.4f} "
              + " ".join(f"@{tgt*100:g}%={row[f'tpr@{tgt}_valth']:.4f}/{row[f'tpr@{tgt}_testth']:.4f}"
                         for tgt in TARGETS)
              + "  (val-th / test-th)")
    results["baselines_matched_subset"] = base

    dest = PROJECT_ROOT / "runs" / "cascade" / "exp4_budget_10s.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(results, indent=1), encoding="utf-8")
    print(f"\n[wrote] {dest}")


if __name__ == "__main__":
    main()
