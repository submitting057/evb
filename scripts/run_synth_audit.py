# -*- coding: utf-8 -*-
"""T1a: zero-training coverage audit of synthetic pseudo-impostor families.

Question (gate for the whole synthetic/OE line): can pseudo-impostors built
from KNOWN train sessions cover the upper tail of the REAL val-unknown
impostors -- in combined score AND in member-argmax pattern?  Coverage is a
NECESSARY condition only: OE humility learned on a family whose tail does not
even reach the real magnets cannot transfer to them.

Families (deployment-legal: known-train sessions only; fixed channel order
[video_cdn, api, channel_info, thumbnail, other]):
  F0 anchor   : the dominant session itself (score ceiling reference)
  F1 splice   : video_cdn from A, all aux channels from B (contradictory
                evidence -> should mimic the "disagreeing" magnet subgroup)
  F2 near     : every channel lam*A + (1-lam)*B, lam ~ U(.85,.95) (near-anchor
                interpolation -> should mimic the "consistent/collinear" one)
  F3 auxanchor: aux channels from A, video_cdn lam*A+(1-lam)*B, lam~U(.7,.9)
                ("aux says A, video traffic shifted" -- the collinear subtype)

Subsystem: the two DF members (frozen), geometric mean, score = max prob,
agreement = e100.argmax == dt.argmax.  LGBM cannot score synthetic sessions
(its segv2 features are not mixable), so the audit lives in the 2-DF system;
real magnets are re-scored in the same subsystem for apples-to-apples.
"""
import argparse
import copy
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import PROJECT_ROOT, RunConfig
from src.data import SessionIndex, load_splits, session_pool
from src.dfnet import DFData, DFNet, df_collate, load_series_cache

TAGS = ("dfs_branch_all_e100", "dfs_branch_all_dt")
N_FAM = {"F0_anchor": 3000, "F1_splice": 3000, "F2_near": 3000, "F3_auxanchor": 3000}


def unpack(batch):
    *raw, y, idx = batch
    xs = list(raw[0]) if len(raw) == 1 and isinstance(raw[0], (tuple, list)) else raw
    return xs, y, idx


@torch.no_grad()
def predict(model, ds, device, batch=256):
    model.eval()
    ld = DataLoader(ds, batch_size=batch, shuffle=False, collate_fn=df_collate)
    out = []
    for b in ld:
        xs, _, _ = unpack(b)
        out.append(torch.softmax(model(*[x.to(device) for x in xs]), 1)
                   .float().cpu().numpy())
    return np.concatenate(out)


def _pad(x, L):
    if len(x) >= L:
        return x[:L].astype(np.float32)
    o = np.zeros(L, np.float32)
    o[:len(x)] = x
    return o


def mix_row(ra, rb, plan):
    """plan: list per channel of ('a'|'b'|('mix', lam)). Returns a new row."""
    out = []
    for ci, ca in enumerate(ra):
        cb = rb[ci]
        m = plan[ci]
        if m == "a":
            out.append(ca)
        elif m == "b":
            out.append(cb)
        else:
            lam = m[1]
            L = max(len(ca[0]), len(cb[0]))
            enc = (lam * _pad(ca[0], L) + (1 - lam) * _pad(cb[0], L)).astype(np.float16)
            dt = None
            if ca[2] is not None:
                dt = (lam * _pad(ca[2], L) + (1 - lam) * _pad(cb[2], L)).astype(np.float16)
            out.append((enc, ca[1], dt, ca[3]))
    return out


def geo2(P, Q):
    L = 0.5 * (np.log(np.clip(P, 1e-12, 1)) + np.log(np.clip(Q, 1e-12, 1)))
    L -= L.max(1, keepdims=True)
    E = np.exp(L)
    return E / E.sum(1, keepdims=True)


def tail_stats(name, score, agree, imp_cls, K=38):
    o = np.argsort(score)[::-1][:K]
    return {"family": name,
            "score_q50": float(np.quantile(score, .5)),
            "score_q99": float(np.quantile(score, .99)),
            "score_max": float(score.max()),
            "topK_agree_frac": float(agree[o].mean()),
            "topK_classes": Counter(imp_cls[o].tolist()).most_common(5)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="10s")
    ap.add_argument("--out", default="synth_audit")
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    rcfg = RunConfig(dataset=args.dataset, protocol="strict")
    index = SessionIndex(rcfg.dataset_dir, rcfg.bin_size_ms)
    splits = load_splits(rcfg.splits_path)
    known = sorted(splits["known_videos"])
    vidx = {v: i for i, v in enumerate(known)}
    labels = {s: vidx[r["video"]] if r["role"] == "known" else -1
              for s, r in splits["sessions"].items()}
    train_by_video = {}
    for s, r in splits["sessions"].items():
        if r["role"] == "known" and r["split"] == "train":
            train_by_video.setdefault(r["video"], []).append(s)
    vids = sorted(train_by_video)

    cache = load_series_cache(index, args.dataset)

    # ---- sample pairs (a, b) from different known videos --------------------
    rng = np.random.default_rng(0)
    pairs = []
    for _ in range(N_FAM["F1_splice"]):
        i, j = rng.choice(len(vids), 2, replace=False)
        va, vb = vids[i], vids[j]
        pairs.append((train_by_video[va][rng.integers(len(train_by_video[va]))],
                      train_by_video[vb][rng.integers(len(train_by_video[vb]))]))
    src_sids = sorted({s for p in pairs for s in p})
    pos = {s: i for i, s in enumerate(src_sids)}
    ai = np.array([pos[p[0]] for p in pairs])
    bi = np.array([pos[p[1]] for p in pairs])
    print(f"sources: {len(src_sids)} train sessions, {len(pairs)} cross-video pairs")

    # ---- frozen members ------------------------------------------------------
    models, margs = {}, {}
    for t in TAGS:
        saved = torch.load(PROJECT_ROOT / "runs" / t / "model.pt",
                           map_location=device, weights_only=False)
        margs[t] = saved["args"]
        m = DFNet(saved["channels"], margs[t]["fusion"], n_class=len(known),
                  F=2 if margs[t]["dt"] else 1, side_dim=0,
                  chan_emb=margs[t]["chan_emb"]).to(device)
        m.load_state_dict(saved["state"])
        models[t] = m

    def rows_of(t):
        ds = DFData(cache, src_sids, labels, margs[t]["channels"], margs[t]["fusion"],
                    margs[t]["value"], margs[t]["dt"], chan_emb=margs[t]["chan_emb"],
                    augment=False)
        return ds

    fam_probs = {f: {t: None for t in TAGS} for f in N_FAM}
    for t in TAGS:
        base = rows_of(t)
        data = base.data
        plans = {
            "F0_anchor": None,                       # pure dominant session
            "F1_splice": ["a", "b", "b", "b", "b"],
            "F2_near": None,                          # per-pair lam below
            "F3_auxanchor": None,
        }
        lam2 = rng.uniform(.85, .95, len(pairs))
        lam3 = rng.uniform(.70, .90, len(pairs))
        built = {}
        for fam in N_FAM:
            rows = []
            for k, (ia, ib) in enumerate(zip(ai, bi)):
                ra, rb = data[ia], data[ib]
                if fam == "F0_anchor":
                    rows.append(ra)
                elif fam == "F1_splice":
                    rows.append(mix_row(ra, rb, plans[fam]))
                elif fam == "F2_near":
                    rows.append(mix_row(ra, rb, [("mix", lam2[k])] * 5))
                else:
                    rows.append(mix_row(ra, rb, [("mix", lam3[k]), "a", "a", "a", "a"]))
            d2 = copy.copy(base)
            d2.data = rows
            d2.ids = [f"synth{i}" for i in range(len(rows))]
            d2.labels = np.zeros(len(rows), dtype=np.int64)
            d2.augment = False
            built[fam] = predict(models[t], d2, device)
            del d2, rows
        for fam in N_FAM:
            fam_probs[fam][t] = built[fam]
        del data, base
        print(f"[fwd] {t} done")

    # ---- real val impostors in the same 2-DF subsystem ----------------------
    P = {t: np.load(PROJECT_ROOT / "runs" / t / "probs_val.npz")["imp"] for t in TAGS}
    Pn = {t: p / (p.sum(1, keepdims=True) + 1e-12) for t, p in P.items()}
    realG = geo2(Pn[TAGS[0]], Pn[TAGS[1]])
    real_score = realG.max(1)
    real_agree = Pn[TAGS[0]].argmax(1) == Pn[TAGS[1]].argmax(1)
    # impersonated class: agreed class when agree else e100's argmax
    a0, a1 = Pn[TAGS[0]].argmax(1), Pn[TAGS[1]].argmax(1)
    real_cls = np.where(real_agree, a0, a0)

    rep = {"n_pairs": len(pairs), "families": {}, "real": tail_stats(
        "REAL_val_imp", real_score, real_agree, real_cls)}

    print("\n=== T1a: synthetic family coverage (2-DF geo subsystem) ===")
    for fam in N_FAM:
        G = geo2(fam_probs[fam][TAGS[0]], fam_probs[fam][TAGS[1]])
        sc = G.max(1)
        f0, f1 = fam_probs[fam][TAGS[0]], fam_probs[fam][TAGS[1]]
        agr = f0.argmax(1) == f1.argmax(1)
        cls = np.where(agr, f0.argmax(1), f0.argmax(1))
        st = tail_stats(fam, sc, agr, cls)
        # coverage: real top-38 events sitting inside the family's own 1% tail
        q = np.quantile(sc, 1 - 38 / len(sc))
        top38 = np.argsort(real_score)[::-1][:38]
        st["cov38_by_q99"] = float(np.mean(real_score[top38] <= q))
        st["cov38_by_max"] = float(np.mean(real_score[top38] <= sc.max()))
        # per-magnet-video: family max >= video's top-event score
        imp_sids = session_pool(splits, "unknown", "val")
        vrow = np.array([splits["sessions"][s]["video"] for s in imp_sids])
        cov_v = []
        for v in sorted(set(vrow[top38].tolist())):
            mv = real_score[vrow == v].max()
            cov_v.append(bool(sc.max() >= mv))
        st["cov_videos38"] = float(np.mean(cov_v))
        rep["families"][fam] = st
        print(f"{fam:12s} q50={st['score_q50']:.3f} q99={st['score_q99']:.3f} "
              f"max={st['score_max']:.3f} | top38 agree={st['topK_agree_frac']:.2f} "
              f"| cov38(q99)={st['cov38_by_q99']:.2f} cov38(max)={st['cov38_by_max']:.2f} "
              f"covV={st['cov_videos38']:.2f} | top cls {st['topK_classes'][:3]}")
    r = rep["real"]
    print(f"{'REAL val':12s} q50={r['score_q50']:.3f} q99={r['score_q99']:.3f} "
          f"max={r['score_max']:.3f} | top38 agree={r['topK_agree_frac']:.2f} "
          f"| top cls {r['topK_classes'][:3]}")

    out = PROJECT_ROOT / "runs" / args.out
    out.mkdir(parents=True, exist_ok=True)
    (out / "report.json").write_text(json.dumps(rep, indent=1, default=str),
                                     encoding="utf-8")
    print(f"[wrote] {out / 'report.json'}")


if __name__ == "__main__":
    main()
