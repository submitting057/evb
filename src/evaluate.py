# -*- coding: utf-8 -*-
"""Unified closed-set + open-set evaluation.

Everything is expressed over a probe->class score matrix:
  probe_mat   (n_probes, n_classes)  higher = more similar / more probable
  probe_true  (n_probes,)            int class index of each probe
  imp_scores  (n_impostors,)         scalar open-set score of each unknown probe
                                     (max over classes, same scale as probe_mat)
"""
import numpy as np

FPR_GRID = (0.005, 0.01, 0.05)


def closed_set_metrics(probe_mat: np.ndarray, probe_true: np.ndarray) -> dict:
    order = np.argsort(-probe_mat, axis=1)
    top1 = float(np.mean(order[:, 0] == probe_true))
    top5 = float(np.mean(np.any(order[:, :5] == probe_true[:, None], axis=1)))
    return {"top1": top1, "top5": top5}


def _threshold_for_fpr(imp: np.ndarray, fpr: float) -> float:
    """Smallest threshold keeping empirical FPR <= fpr (imp scores >= th count as positive)."""
    n = len(imp)
    k = int(np.floor(fpr * n))          # allowed number of impostor accepts
    if k >= n:
        return -np.inf
    sorted_desc = np.sort(imp)[::-1]
    # threshold must exceed the k-th largest impostor so at most k are >= it
    return float(sorted_desc[k]) + 1e-12


def tpr_at_fpr(probe_scores: np.ndarray, probe_correct: np.ndarray,
               imp: np.ndarray, fpr: float) -> tuple[float, float]:
    th = _threshold_for_fpr(imp, fpr)
    tpr = float(np.mean(probe_correct & (probe_scores >= th)))
    return tpr, th


def open_set_with_custom_score(probe_mat: np.ndarray, probe_true: np.ndarray,
                               probe_scores: np.ndarray, imp_scores: np.ndarray,
                               fpr_grid=FPR_GRID) -> dict:
    """Open-set metrics where the accept score differs from the argmax matrix
    (e.g. margin or negative entropy); identification still uses probe_mat."""
    correct = probe_mat.argmax(1) == probe_true
    out = {"closed": closed_set_metrics(probe_mat, probe_true),
           "fpr": {f: {} for f in fpr_grid}}
    for f in fpr_grid:
        tpr, th = tpr_at_fpr(probe_scores, correct, imp_scores, f)
        out["fpr"][f] = {"tpr": tpr, "threshold": th}
    return out


def open_set_metrics(probe_mat: np.ndarray, probe_true: np.ndarray,
                     imp_scores: np.ndarray, fpr_grid=FPR_GRID,
                     n_boot: int = 1000, seed: int = 0) -> dict:
    """Open-set identification: accept iff score>=theta (and, for TPR, argmax correct)."""
    best = probe_mat.max(axis=1)
    pred = probe_mat.argmax(axis=1)
    correct = pred == probe_true
    out = {
        "n_probes": int(len(probe_true)),
        "n_impostors": int(len(imp_scores)),
        "closed": closed_set_metrics(probe_mat, probe_true),
        "fpr": {},
    }
    # detection AUROC (known-probe best score vs impostor score), independent of argmax
    labels = np.concatenate([np.ones(len(best)), np.zeros(len(imp_scores))])
    scores = np.concatenate([best, imp_scores])
    ranks = np.argsort(np.argsort(scores)) + 1
    pos = labels == 1
    n_pos, n_neg = pos.sum(), (~pos).sum()
    out["auroc_detection"] = float((ranks[pos].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))

    for f in fpr_grid:
        tpr, th = tpr_at_fpr(best, correct, imp_scores, f)
        out["fpr"][f] = {"tpr": tpr, "threshold": th}

    rng = np.random.default_rng(seed)
    boot = {f: [] for f in fpr_grid}
    n_p, n_i = len(best), len(imp_scores)
    for _ in range(n_boot):
        pi = rng.integers(0, n_p, n_p)
        ii = rng.integers(0, n_i, n_i)
        b, c, im = best[pi], correct[pi], imp_scores[ii]
        for f in fpr_grid:
            t, _ = tpr_at_fpr(b, c, im, f)
            boot[f].append(t)
    for f in fpr_grid:
        if n_boot > 0 and boot[f]:
            lo, hi = np.percentile(boot[f], [2.5, 97.5])
            out["fpr"][f]["tpr_ci95"] = [float(lo), float(hi)]
    return out


def format_open_set_report(tag: str, m: dict) -> str:
    lines = [f"--- {tag} ---",
             f"probes={m['n_probes']} impostors={m['n_impostors']}",
             f"closed-set: top1={m['closed']['top1']:.4f} top5={m['closed']['top5']:.4f}",
             f"open-set detection AUROC={m['auroc_detection']:.4f}"]
    for f, rec in m["fpr"].items():
        ci = rec.get("tpr_ci95")
        ci_s = f" [{ci[0]:.4f}, {ci[1]:.4f}]" if ci else ""
        lines.append(f"TPR@{f * 100:g}%FPR = {rec['tpr']:.4f}{ci_s}")
    return "\n".join(lines)
