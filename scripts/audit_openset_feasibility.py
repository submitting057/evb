"""Read-only model audit and validation-only score probes; never score test arrays.

Writes a separate audit report. Existing model weights, splits and caches are unchanged.
The fixed experts and weights already used validation data, so grouped checks below
are diagnostics, not an unbiased evaluation of the complete selection pipeline.
"""
import hashlib
import json
from collections import Counter
from pathlib import Path
import sys

import numpy as np
from scipy.special import logsumexp

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.evaluate import _threshold_for_fpr

TAGS = ["ngL_dfs_branch_vid", "ngL_dfs_branch_vid_dt", "ngL_dfs_branch_aux3"]
WEIGHTS = np.array([.15, .15, .30, .40])
FPRS = [.005, .01, .05]
BETAS = [0., .1, .25, .5, 1.]
DATA = Path(r"E:\evb\datasets\600classes_30samples_10s_no_gro\dataset")


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_manifest(path):
    with Path(path).open(encoding="utf-8") as f:
        return {r["session_id"]: r for r in map(json.loads, f) if r}


def pool(spl, role, split):
    return [sid for sid, r in spl["sessions"].items()
            if r["role"] == role and r["split"] == split
            and (role == "unknown" or not r["gallery"])]


def metric(correct, sp, si):
    out = {"top1": float(correct.mean()), "operating_points": {}}
    for f in FPRS:
        th = _threshold_for_fpr(si, f)
        out["operating_points"][str(f)] = {
            "tpr": float(np.mean(correct & (sp >= th))),
            "threshold": float(th), "fpr": float(np.mean(si >= th)),
        }
    return out


def fold_ids(vids, seed):
    uv = np.array(sorted(set(vids)))
    perm = np.random.default_rng(seed).permutation(uv)
    mapping = {v: i % 5 for i, v in enumerate(perm)}
    return np.array([mapping[v] for v in vids])


def main():
    split_path = ROOT / "splits/splits_10s_nogro_strict_s42.json"
    spl = read_json(split_path)
    ps, ins = pool(spl, "known", "val"), pool(spl, "unknown", "val")
    known = sorted(spl["known_videos"])
    yp = np.array([known.index(spl["sessions"][s]["video"]) for s in ps])
    pv = np.array([spl["sessions"][s]["video"] for s in ps])
    iv = np.array([spl["sessions"][s]["video"] for s in ins])
    with np.load(ROOT / "runs/cache/lgbm_probs_10s_nogro_strict.npz") as z:
        mats_p, mats_i = [z["val_probe"]], [z["val_imp"]]
    for tag in TAGS:
        with np.load(ROOT / "runs" / tag / "probs_val.npz") as z:
            assert np.array_equal(z["probe_y"], yp)
            mats_p.append(z["probe"])
            mats_i.append(z["imp"])
    assert all(p.shape == (len(ps), len(known)) for p in mats_p)
    assert all(p.shape == (len(ins), len(known)) for p in mats_i)
    # Match the production combiner's normalization and mixed-precision arithmetic.
    mats_p = [p / (p.sum(1, keepdims=True) + 1e-12) for p in mats_p]
    mats_i = [p / (p.sum(1, keepdims=True) + 1e-12) for p in mats_i]
    lp = sum(float(w) * np.log(np.clip(p, 1e-12, 1)) for w, p in zip(WEIGHTS, mats_p))
    li = sum(float(w) * np.log(np.clip(p, 1e-12, 1)) for w, p in zip(WEIGHTS, mats_i))
    zp, zi = logsumexp(lp, axis=1), logsumexp(li, axis=1)
    sp, si = lp.max(1) - zp, li.max(1) - zi
    predp, predi = lp.argmax(1), li.argmax(1)
    correct = predp == yp
    base = metric(correct, sp, si)
    expected = read_json(ROOT / "runs/combo_10s_nogro_L8k_chsplit4/report.json")["finals"][0]["val"]
    assert abs(base["top1"] - expected["top1"]) < 1e-10
    assert all(abs(base["operating_points"][str(f)]["tpr"] - expected[str(f)]) < 1e-10 for f in FPRS)
    th = _threshold_for_fpr(si, .005)
    acceptp, accepti = sp >= th, si >= th
    top = np.argsort(-si)[:int(np.floor(.005 * len(si)))]
    counts = Counter(iv[top].tolist())
    strata = {"correct_accepted": correct & acceptp,
              "correct_rejected": correct & ~acceptp, "wrong": ~correct}
    diagnostics = {}
    for name, mask in strata.items():
        diagnostics[name] = {"count": int(mask.sum()),
            "median_geo_probability": float(np.median(np.exp(sp[mask]))),
            "median_overlap_Z": float(np.median(np.exp(zp[mask]))),
            "member_predicted_class_support_median": {
                n: float(np.median(m[np.arange(len(ps)), predp][mask]))
                for n, m in zip(["lgbm"] + TAGS, mats_p)}}
    diagnostics["unknown_tail"] = {"count": len(top), "unique_videos": len(counts),
        "video_counts_desc": sorted(counts.values(), reverse=True),
        "median_overlap_Z": float(np.median(np.exp(zi[top]))),
        "member_predicted_class_support_median": {
            n: float(np.median(m[np.arange(len(ins)), predi][top]))
            for n, m in zip(["lgbm"] + TAGS, mats_i)}}
    probes = {str(b): metric(correct, sp + b*zp, si + b*zi) for b in BETAS}

    # Independent threshold/assessment folds for fixed scores. Optional beta
    # selection uses only the other four folds; upstream weights are not nested.
    grouped = []
    for seed in [0, 1, 2, 3, 4]:
        pf, inf = fold_ids(pv, seed), fold_ids(iv, seed + 100)
        for mode in [str(b) for b in BETAS] + ["selected_beta"]:
            ap, ai = np.zeros(len(ps), bool), np.zeros(len(ins), bool)
            choices = []
            for fold in range(5):
                cp, ci = pf != fold, inf != fold
                if mode == "selected_beta":
                    def criterion(b):
                        m = metric(correct[cp], (sp+b*zp)[cp], (si+b*zi)[ci])
                        return tuple(m["operating_points"][str(f)]["tpr"] for f in FPRS) + (-b,)
                    b = max(BETAS, key=criterion)
                else:
                    b = float(mode)
                threshold = _threshold_for_fpr((si+b*zi)[ci], .005)
                ap[~cp] = (sp+b*zp)[~cp] >= threshold
                ai[~ci] = (si+b*zi)[~ci] >= threshold
                choices.append(b)
            grouped.append({"seed": seed, "mode": mode, "betas": choices,
                "tpr": float(np.mean(correct & ap)), "realized_fpr": float(ai.mean())})

    # Audit provenance and sample identity. These are metadata, not test scores.
    manifest = read_manifest(DATA / "manifests/sessions.jsonl")
    oldmanifest = read_manifest(Path(r"E:\evb\datasets\600vids_30samples_10s\manifests\sessions.jsonl"))
    olds = read_json(ROOT / "splits/splits_10s_strict_s42.json")
    common = sorted(set(manifest) & set(oldmanifest))
    common_splits = sorted(set(spl["sessions"]) & set(olds["sessions"]))
    same_hash = sum(manifest[s].get("pcap_sha256") == oldmanifest[s].get("pcap_sha256") for s in common)
    role_changes = Counter()
    for s in common_splits:
        a, b = olds["sessions"][s], spl["sessions"][s]
        if (a["role"], a["split"]) != (b["role"], b["split"]):
            role_changes[f"{a['role']}/{a['split']} -> {b['role']}/{b['split']}"] += 1
    hash_groups = {}
    for sid, r in spl["sessions"].items():
        h = manifest[sid].get("pcap_sha256")
        if h:
            hash_groups.setdefault(h, set()).add((r["role"], r["split"]))
    provenance = {"new_sessions": len(manifest), "old_sessions": len(oldmanifest),
        "common_session_ids": len(common), "common_same_pcap_hash": same_hash,
        "common_split_session_ids": len(common_splits),
        "role_or_split_changes_among_common": dict(role_changes),
        "new_pcap_hashes_crossing_roles_or_splits": sum(len(v)>1 for v in hash_groups.values()),
        "gro_applied": dict(Counter(str(r.get("gro_applied")) for r in manifest.values())),
        "gro_segment": dict(Counter(str(r.get("gro_segment")) for r in manifest.values())),
        "video_batches": dict(Counter(str(r.get("video_batches")) for r in manifest.values())),
        "duration_quantiles_s": np.quantile([r["video_duration_s"] for r in manifest.values()], [.1,.5,.9]).tolist(),
        "parser": read_json(DATA / "provenance/parser_config_20261105_152445.json"),
        "preprocessing_version": read_json(DATA / "provenance/preprocessing_version.json")["version"]}
    # Per-stratum metadata summaries use validation only.
    for name, mask in strata.items():
        rows = [manifest[s] for s, m in zip(ps, mask) if m]
        diagnostics[name]["duration_median_s"] = float(np.median([r["video_duration_s"] for r in rows]))
        diagnostics[name]["datagrams_median"] = float(np.median([r["video_datagrams"] for r in rows]))
    # Inspect existing encrypted event features: path boundaries and batch markers
    # are retained on disk but discarded by the current DF sequence cache.
    limits = {"video_cdn": 8192, "api": 768, "channel_info": 192, "thumbnail": 384}
    quality = {}
    for sid in ps + [ins[i] for i in top]:
        r = spl["sessions"][sid]
        with np.load(DATA / "features/event_sequences" / r["video"] / (sid + ".npz")) as z:
            events, paths = z["events"], z["path_ids"]
            meta = json.loads(str(z["metadata"]))
        labels = {int(k): v[0]["channel_name"] for k, v in meta["channel_labels"].items()}
        channels = np.array([labels.get(int(p), "other") for p in paths])
        rec = {}
        for channel, limit in limits.items():
            mask = channels == channel
            rec[channel+"_missing"] = int(not mask.any())
            rec[channel+"_truncated"] = int(mask.sum() > limit)
            rec[channel+"_paths"] = len(set(paths[mask].tolist()))
            rec[channel+"_events"] = int(mask.sum())
        rec["batch_markers"] = len(meta.get("batch_end_rel_ms", []))
        quality[sid] = rec
    groups = {name: [s for s, m in zip(ps, mask) if m] for name, mask in strata.items()}
    groups["unknown_tail"] = [ins[i] for i in top]
    for name, sids in groups.items():
        diagnostics[name]["event_quality"] = {
            k: float(np.mean([quality[s][k] for s in sids]))
            if k.endswith(("_missing", "_truncated")) else float(np.median([quality[s][k] for s in sids]))
            for k in next(iter(quality.values()))}
    report = {"scope": "historical baseline reports + validation-only score diagnostics; no new test scoring",
        "limitations": ["Fixed experts and weights previously used all validation data",
            "Probability caches lack session IDs and class IDs; verified labels/counts, not independent row identity",
            "Grouped diagnostic is not a complete nested estimate or a new blind test"],
        "split_sha256": hashlib.sha256(split_path.read_bytes()).hexdigest(),
        "members": ["lgbm"] + TAGS, "weights": WEIGHTS.tolist(),
        "validation_counts": {"known": len(ps), "unknown": len(ins),
            "known_videos": len(set(pv)), "unknown_videos": len(set(iv))},
        "baseline_validation": base, "diagnostics": diagnostics,
        "overlap_score_beta_probe": probes, "grouped_calibration_diagnostics": grouped,
        "data_provenance": provenance}
    out = ROOT / "runs/open_set_feasibility_20261109"
    out.mkdir(parents=True, exist_ok=True)
    (out / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"baseline": base, "diagnostics": diagnostics, "beta_probe": probes,
        "grouped_mean": {mode: {k: float(np.mean([r[k] for r in grouped if r["mode"] == mode]))
            for k in ["tpr", "realized_fpr"]} for mode in [str(b) for b in BETAS]+["selected_beta"]},
        "provenance": {k:v for k,v in provenance.items() if k != "parser"}}, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
