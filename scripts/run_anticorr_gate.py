# -*- coding: utf-8 -*-
"""Gate-1 evaluation for anti-correlation trained B members (proposal 3.4).

Pre-registered kill line (complementarity_training.md):
  val TRUE-impostor argmax agreement vs A (vid_dt):
    > 30%  -> discard (worse cost-effectiveness than the tree bloc)
    < 25%  -> worth continuing into LOVO
  in between -> borderline, judgement call.

Everything here is val-only (selection-legal).  Also reports the three
degradation-mode diagnostics of proposal 3.3:
  (1) collapse-to-flat  : val probe top1 floor ~.65 (lgbm .667 / aux3 .683);
                          impostor maxprob distribution
  (2) memorization      : agreement must drop on TRUE val impostors, not just
                          on the K pseudo videos -> per-epoch hist + val imp
                          agreement/co-peak are the transfer measures
  (3) class suppression : blind-class probe top1 (main phase) and its repair
References (measured, openset_error_complementarity.md):
  vid<->vid_dt 64.6%, e100<->dt 52%, lgbm-vs-DF 16.0%, aux3 0.3-0.4%,
  chance ~0.26%.
"""
import glob
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import PROJECT_ROOT
from src.evaluate import _threshold_for_fpr

A_TAG = "ngL_dfs_branch_vid_dt"
VID_TAG = "ngL_dfs_branch_vid"
REFS = {
    "vid_vs_A": 0.646, "e100_vs_dt": 0.520, "lgbm_vs_DF": 0.160,
    "aux3_vs_DF": 0.0035, "chance": 0.0026,
}


def norm(P):
    return P / (P.sum(1, keepdims=True) + 1e-12)


def main():
    tags = sorted(Path(p).parent.name.replace("_rep", "")
                  for p in glob.glob(str(PROJECT_ROOT / "runs" / "ngL_ac_d?_l*"
                                         / "report.json")))
    tags = sorted(set(tags))
    if not tags:
        print("no ngL_ac_* runs found")
        return
    print(f"variants: {tags}")

    imp_ref, probe_ref = {}, {}
    for t in (A_TAG, VID_TAG):
        d = np.load(PROJECT_ROOT / "runs" / t / "probs_val.npz")
        imp_ref[t] = norm(d["imp"])
        probe_ref[t] = (norm(d["probe"]), d["probe_y"])
    ca = imp_ref[A_TAG].argmax(1)
    sa = imp_ref[A_TAG].max(1)
    t21 = np.argsort(-sa)[:21]
    cv = imp_ref[VID_TAG].argmax(1)
    sv = imp_ref[VID_TAG].max(1)
    t21v = np.argsort(-sv)[:21]

    # incumbent vid's own gate numbers as the "passive same-recipe" control
    pv_id, y_id = probe_ref[VID_TAG]
    pi_id = imp_ref[VID_TAG]
    copk_vid = pi_id[np.arange(len(ca)), ca]

    rows = {}
    for tag in tags:
        for phase, sfx in (("main", ""), ("rep", "_rep")):
            p = PROJECT_ROOT / "runs" / f"{tag}{sfx}" / "probs_val.npz"
            if not p.exists():
                continue
            d = np.load(p)
            P, y, I = norm(d["probe"]), d["probe_y"], norm(d["imp"])
            cb = I.argmax(1)
            copk = I[np.arange(len(ca)), ca]
            cor = P.argmax(1) == y
            cor_a = probe_ref[A_TAG][0].argmax(1) == probe_ref[A_TAG][1]
            # rescue/sabotage vs A on non-blind probes (B blind rows excluded
            # from the fairness of the ratio)
            rep = json.load(open(PROJECT_ROOT / "runs" / f"{tag}{sfx}"
                                 / "report.json", encoding="utf-8"))
            blind = np.isin(y, rep["blind_classes"])
            rescue = float(cor[~blind & ~cor_a].mean())       # P(B对|A错)
            sabotage = float((~cor[~blind & cor_a]).mean())   # P(B错|A对)
            p_true = P[np.arange(len(y)), y]
            r = {
                "top1": float(cor.mean()),
                "top1_nonblind": float(cor[~blind].mean()),
                "p_true_blind_mean": float(p_true[blind].mean()),
                "p_true_blind_med": float(np.median(p_true[blind])),
                "at05": float(np.mean(cor[~blind]
                                      & (P.max(1)[~blind]
                                         >= _threshold_for_fpr(I.max(1), .005)))),
                "agree_A": float((cb == ca).mean()),
                "agree_vid": float((cb == cv).mean()),
                "copeak_mean": float(copk.mean()),
                "copeak_top21A": float(copk[t21].mean()),
                "copeak_top21vid": float(copk[t21v].mean()),
                "imp_max_q50": float(np.quantile(I.max(1), .5)),
                "imp_max_q99": float(np.quantile(I.max(1), .99)),
                "rescue_vs_A": rescue, "sabotage_vs_A": sabotage,
                "best_epoch": rep.get("best_epoch"),
            }
            rows[f"{tag}{sfx}"] = r

    order = sorted(rows, key=lambda k: rows[k]["agree_A"])
    print(f"\n{'variant':22s} {'agreeA':>7s} {'<25%':>5s} {'agreeVid':>8s} "
          f"{'copk21A':>8s} {'vidcopk':>8s} {'top1':>6s} {'nonbnd':>7s} "
          f"{'@.5%':>6s} {'rescue':>7s} {'sabot':>6s} {'impQ99':>7s}")
    print(f"{'[vid incumbent]':22s} {REFS['vid_vs_A']:7.3f} {'-':>5s} "
          f"{'-':>8s} {copk_vid[t21].mean():8.3f} {copk_vid[t21v].mean():8.3f} "
          f"{'-':>6s} {float((pv_id.argmax(1) == y_id).mean()):7.3f} "
          f"{'-':>6s} {'-':>7s} {'-':>6s} "
          f"{float(np.quantile(pi_id.max(1), .99)):7.3f}")
    for k in order:
        r = rows[k]
        verdict = "PASS" if r["agree_A"] < .25 else ("KILL" if r["agree_A"] > .30
                                                     else "border")
        print(f"{k:22s} {r['agree_A']:7.3f} {verdict:>5s} {r['agree_vid']:8.3f} "
              f"{r['copeak_top21A']:8.3f} {r['copeak_top21vid']:8.3f} "
              f"{r['top1']:6.3f} {r['top1_nonblind']:7.3f} {r['at05']:6.3f} "
              f"{r['rescue_vs_A']:7.3f} {r['sabotage_vs_A']:6.3f} "
              f"{r['imp_max_q99']:7.3f}")

    out = PROJECT_ROOT / "runs" / "anticorr_gate"
    out.mkdir(parents=True, exist_ok=True)
    (out / "report.json").write_text(
        json.dumps({"rows": rows, "refs": REFS}, indent=1), encoding="utf-8")
    print(f"[wrote] {out / 'report.json'}")


if __name__ == "__main__":
    main()
