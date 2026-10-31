# -*- coding: utf-8 -*-
"""Exp2: quantify the video_cdn_ips signal for stage-1 VID pruning.

Sources: manifests/sessions.jsonl (per-session video_cdn_ips, up to 10),
splits_10s_s42.json for protocol-correct pools.

Measures:
  A. basic coverage: how many sessions have IPs, distinct IPs /24s /16s
  B. same-video vs diff-video session-pair prefix-set Jaccard; same-day vs
     cross-day drift
  C. pruning test: reference profile from gallery sessions (enrollment),
     probe candidate set = VIDs whose profile intersects the probe prefix set;
     report recall / candidate-set size, per prefix granularity
  D. combined pruning: IP-candidates combined with LGBM top-K (intersect /
     union), using cached LGBM test-probe probs

Usage: python scripts/exp2_ip.py [10s|5s]   (manifest taken from the 10s set)
"""
import json
import random
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import PROJECT_ROOT, DATASET_DIRS
from src.data import load_splits

DF_TAGS = {"10s": ["df_branch_all_e100", "df_branch_all_dt"],
           "5s": ["df5_branch_all_e100"]}


def pref_set(ips: list[str], plen: int) -> frozenset:
    return frozenset(".".join(ip.split(".")[:plen]) for ip in ips)


def jaccard(a: frozenset, b: frozenset) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def main(dataset: str = "10s") -> None:
    splits = load_splits(PROJECT_ROOT / "splits" / f"splits_{dataset}_s42.json")
    known = sorted(splits["known_videos"])
    vidx = {v: i for i, v in enumerate(known)}

    recs = {}
    with open(DATASET_DIRS["10s"] / "manifests" / "sessions.jsonl", encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            recs[d["session_id"]] = d

    sess = {}
    for sid, sp in splits["sessions"].items():
        m = recs.get(sid)
        if m is None:
            continue
        date = re.search(r"-(\d{8})_\d{6}$", sid)
        sess[sid] = dict(video=sp["video"], role=sp["role"], split=sp["split"],
                         gallery=sp["gallery"], ips=m.get("video_cdn_ips") or [],
                         date=date.group(1) if date else "?")

    # ---- A. coverage -----------------------------------------------------
    all_ips = [ip for s in sess.values() for ip in s["ips"]]
    cov = {"n_sessions": len(sess),
           "sessions_with_ips": sum(bool(s["ips"]) for s in sess.values()),
           "distinct_ips": len(set(all_ips)),
           "distinct_24": len({".".join(ip.split(".")[:2]) for ip in all_ips}),
           "distinct_16": len({".".join(ip.split(".")[:1]) for ip in all_ips})}
    print("A. coverage:", json.dumps(cov))

    # ---- B. pair overlap: same vs diff video, same vs cross day -----------
    rng = random.Random(0)
    sids = list(sess)
    by_video = defaultdict(list)
    for sid, s in sess.items():
        by_video[s["video"]].append(sid)

    def pair_stats(pairs):
        vals = []
        for a, b in pairs:
            vals.append(jaccard(pref_set(sess[a]["ips"], 2), pref_set(sess[b]["ips"], 2)))
        return float(np.median(vals)), float(np.mean(vals)), float(np.mean(np.array(vals) > 0))

    same = same_day = diff_day = 0, [], []
    same_pairs, diff_pairs = [], []
    for v, lst in by_video.items():
        if len(lst) > 30:
            lst = rng.sample(lst, 30)
        for i in range(len(lst)):
            for j in range(i + 1, len(lst)):
                same_pairs.append((lst[i], lst[j]))
    rng.shuffle(same_pairs), same_pairs.__getitem__(slice(0, 0))
    same_pairs = rng.sample(same_pairs, min(20000, len(same_pairs)))
    diff_pairs = [(rng.choice(sids), rng.choice(sids)) for _ in range(20000)]
    diff_pairs = [(a, b) for a, b in diff_pairs if sess[a]["video"] != sess[b]["video"]][:20000]

    sd, dd = [], []
    for a, b in same_pairs:
        (sd if sess[a]["date"] == sess[b]["date"] else dd).append((a, b))
    med_s = pair_stats(same_pairs)
    med_d = pair_stats(diff_pairs)
    med_sd = pair_stats(sd)
    med_dd = pair_stats(dd)
    print(f"B. /24 Jaccard same-video : median={med_s[0]:.3f} mean={med_s[1]:.3f} overlap%={med_s[2]*100:.1f}%  (n={len(same_pairs)})")
    print(f"B. /24 Jaccard diff-video : median={med_d[0]:.3f} mean={med_d[1]:.3f} overlap%={med_d[2]*100:.1f}%  (n={len(diff_pairs)})")
    print(f"B. same-video same-day    : median={med_sd[0]:.3f} overlap%={med_sd[2]*100:.1f}%  (n={len(sd)})")
    print(f"B. same-video cross-day   : median={med_dd[0]:.3f} overlap%={med_dd[2]*100:.1f}%  (n={len(dd)})")

    # mutual information between /24 prefix of first IP and VID (all sessions)
    p24 = [".".join(s["ips"][0].split(".")[:2]) if s["ips"] else "none" for s in sess.values()]
    vids = [s["video"] for s in sess.values()]
    def nmi(xs, ys):
        from collections import Counter
        n = len(xs)
        cx, cy = Counter(xs), Counter(ys)
        cxy = Counter(zip(xs, ys))
        def H(c):
            return -sum((v / n) * np.log(v / n) for v in c.values())
        mi = sum((c / n) * np.log((c / n) / (cx[x] / n) / (cy[y] / n))
                 for (x, y), c in cxy.items())
        return mi / np.sqrt(H(cx) * H(cy))
    print(f"B. NMI(prefix/24, VID) = {nmi(p24, vids):.4f}")

    # ---- C. gallery-profile pruning test -----------------------------------
    out = {"dataset": dataset, "coverage": cov,
           "pairs": {"same_video": med_s, "diff_video": med_d,
                     "same_day": med_sd, "cross_day": med_dd},
           "prune": {}, "combined": {}}

    lgbm_z = np.load(PROJECT_ROOT / "runs" / "cache" / f"lgbm_probs_{dataset}.npz")
    from src.data import session_pool
    pools = dict(test_probe=session_pool(splits, "known", "test", gallery=False),
                 test_gallery=session_pool(splits, "known", "test", gallery=True))

    for plen, tag in ((2, "/24"), (3, "/16"), (4, "full")):
        prof = defaultdict(set)
        for sid in pools["test_gallery"]:
            prof[sess[sid]["video"]] |= pref_set(sess[sid]["ips"], plen)
        known_with_prof = [v for v in known if prof.get(v)]
        rows = []
        for sid in pools["test_probe"]:
            s = sess[sid]
            p = pref_set(s["ips"], plen)
            cand = frozenset(v for v in known_with_prof if prof[v] & p) if p else frozenset()
            rows.append((s["video"], cand, bool(p)))
        n = len(rows)
        in_set = np.array([v in c for v, c, _ in rows])
        sizes = np.array([len(c) for _, c, _ in rows])
        no_ip = np.array([not has for *_, has in rows])
        empty = sizes == 0
        rec = {"recall": float(in_set.mean()),
               "recall_givem_ip": float(in_set[~no_ip].mean()),
               "mean_cand": float(sizes.mean()), "median_cand": float(np.median(sizes)),
               "pct_empty_then_all": float(empty.mean() * 100),
               "pct_probe_no_ip": float(no_ip.mean() * 100),
               "n_known_with_profile": len(known_with_prof)}
        out["prune"][tag] = rec
        print(f"C. prune {tag:>5}: recall={rec['recall']:.4f} (probe has IP: {rec['recall_givem_ip']:.4f}) "
              f"mean|C|={rec['mean_cand']:.1f} med={rec['median_cand']:.0f} "
              f"empty->{rec['pct_empty_then_all']:.1f}% noIP={rec['pct_probe_no_ip']:.1f}% "
              f"(of {len(known)} known, {len(known_with_prof)} have profile)")

    # ---- D. combined IP + LGBM top-K (test probes) -------------------------
    y = np.load(PROJECT_ROOT / "runs" / DF_TAGS[dataset][0] / "probs_test.npz")["probe_y"]
    pl = lgbm_z["test_probe"]
    pl = pl / pl.sum(1, keepdims=True)
    order = np.argsort(-pl, axis=1)
    probe_sids = pools["test_probe"]
    assert len(probe_sids) == len(y)

    for plen, tag in ((2, "/24"),):
        prof = defaultdict(set)
        for sid in pools["test_gallery"]:
            prof[sess[sid]["video"]] |= pref_set(sess[sid]["ips"], plen)
        for K in (16, 32, 64, 128, 256):
            cand_ip, cand_k = [], []
            for i, sid in enumerate(probe_sids):
                s = sess[sid]
                p = pref_set(s["ips"], plen)
                ipset = {vidx[v] for v in prof if prof[v] & p} if p else set()
                kset = set(order[i, :K].tolist())
                cand_ip.append(ipset)
                cand_k.append(kset)
            inter = [a & b for a, b in zip(cand_ip, cand_k)]
            union = [a | b for a, b in zip(cand_ip, cand_k)]
            for name, cs in (("ip∩topK", inter), ("ip∪topK", union)):
                inr = float(np.mean([y[i] in c for i, c in enumerate(cs)]))
                sz = float(np.mean([len(c) for c in cs]))
                out["combined"][f"{tag}_{name}_K{K}"] = {"recall": inr, "mean_cand": sz}
                print(f"D. {tag} {name} K={K:>3}: recall={inr:.4f} mean|C|={sz:.1f}")

    dest = PROJECT_ROOT / "runs" / "cascade" / f"exp2_ip_{dataset}.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(f"[wrote] {dest}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "10s")
