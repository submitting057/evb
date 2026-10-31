# -*- coding: utf-8 -*-
"""Explore the encrypted-video traffic dataset under E:\\evb\\datasets."""
import json
import os
import sys
from collections import Counter, defaultdict

import numpy as np

DS_ROOT = r"E:\evb\datasets"


def inspect_npz_structure(dataset):
    base = os.path.join(DS_ROOT, dataset, "features")
    for sub in ["event_sequences", "time_bins_10ms", "time_bins_20ms"]:
        d = os.path.join(base, sub)
        vid = sorted(os.listdir(d))[0]
        f = sorted(os.listdir(os.path.join(d, vid)))[0]
        path = os.path.join(d, vid, f)
        z = np.load(path, allow_pickle=True)
        print("=" * 25, sub, "=" * 25)
        print("file:", os.path.basename(path))
        for k in z.files:
            arr = z[k]
            flat = arr.reshape(-1)
            print(f"  {k}: shape={arr.shape} dtype={arr.dtype}")
            if flat.size:
                print(f"     head={flat[:6]}")
                if np.issubdtype(arr.dtype, np.number):
                    print(f"     min={flat.min()} max={flat.max()} mean={flat.mean():.3f}")


def scan_manifest(dataset):
    """Aggregate the sessions manifest: videos, devices, durations, statuses."""
    mpath = os.path.join(DS_ROOT, dataset, "manifests", "sessions.jsonl")
    n = 0
    vids = Counter()
    devices = Counter()
    platforms = Counter()
    networks = Counter()
    statuses = Counter()
    conf = Counter()
    locate = Counter()
    durations = []
    batch_counts = Counter()
    ads = 0
    vids_per_dev = defaultdict(set)
    per_vid_sessions = Counter()
    for line in open(mpath, encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        n += 1
        vids[r["video_id"]] += 1
        devices[r["device"]] += 1
        platforms[r["platform"]] += 1
        networks[r["network"]] += 1
        statuses[r["quality_status"]] += 1
        conf[r["video_locate_confidence"]] += 1
        locate[r["video_locate_method"]] += 1
        durations.append(r["video_duration_s"])
        batch_counts[r.get("video_batches", -1)] += 1
        if r.get("contains_ad"):
            ads += 1
        vids_per_dev[r["device"]].add(r["video_id"])
        per_vid_sessions[r["video_id"]] += 1
    d = np.array(durations)
    print(f"\nsessions={n} unique_videos={len(vids)} contains_ad={ads}")
    print("devices:", dict(devices))
    print("platforms:", dict(platforms))
    print("networks:", dict(networks))
    print("quality_status:", dict(statuses))
    print("locate_confidence:", dict(conf))
    print("locate_method:", dict(locate))
    print("video_batches:", dict(batch_counts))
    print("duration_s: min=%.2f p5=%.2f median=%.2f p95=%.2f max=%.2f mean=%.2f"
          % (d.min(), np.percentile(d, 5), np.median(d), np.percentile(d, 95), d.max(), d.mean()))
    per_vid = np.array(list(per_vid_sessions.values()))
    print("sessions per video: min=%d median=%d max=%d" % (per_vid.min(), np.median(per_vid), per_vid.max()))
    print("videos with <30 sessions:", int((per_vid < 30).sum()))
    for dev, vset in sorted(vids_per_dev.items()):
        print(f"  device {dev}: {len(vset)} unique videos")
    # feature files count per video (should match manifest ok-sessions)
    fdir = os.path.join(DS_ROOT, dataset, "features", "event_sequences")
    counts = {vid: len(os.listdir(os.path.join(fdir, vid))) for vid in os.listdir(fdir)}
    fc = np.array(list(counts.values()))
    print(f"feature files: videos={len(counts)} total={fc.sum()} per-video min={fc.min()} median={np.median(fc)} max={fc.max()}")


def inspect_semantics(dataset):
    """Dig into the column semantics of events / bins arrays."""
    base = os.path.join(DS_ROOT, dataset, "features")
    d = os.path.join(base, "event_sequences")
    vid = sorted(os.listdir(d))[0]
    f = sorted(os.listdir(os.path.join(d, vid)))[0]
    z = np.load(os.path.join(d, vid, f), allow_pickle=True)
    ev = z["events"]
    print("\n--- events rows (first 12) ---")
    for i in range(min(12, len(ev))):
        print(f"  [{i}] size={ev[i,0]:>10.1f} c1={ev[i,1]:>10.4f} c2={ev[i,2]:>8.4f} c3={ev[i,3]:>8.4f}")
    col1 = ev[:, 1]
    print("col1: min=%.4f max=%.4f monotonic=%s" % (col1.min(), col1.max(), bool(np.all(np.diff(col1) >= -1e-9))))
    print("col2 unique sample:", np.unique(ev[:, 2])[:10])
    print("col3 unique sample:", np.unique(ev[:, 3])[:10])
    # path_ids vs channel labels
    meta = json.loads(str(z["metadata"]))
    print("path_id -> channel:", {pid: v[0]["channel_name"] for pid, v in sorted(meta["channel_labels"].items(), key=lambda kv: int(kv[0]))})
    pc = Counter(z["path_ids"].tolist())
    print("events per path_id:", dict(sorted(pc.items())))

    d = os.path.join(base, "time_bins_10ms")
    z2 = np.load(os.path.join(d, vid, f), allow_pickle=True)
    bins = z2["bins"]
    print("\n--- bins shape", bins.shape, "---")
    print("axis0 (path) totals of each feature:")
    for p in range(bins.shape[0]):
        tot = bins[p].sum(axis=0)
        nz = int((bins[p].sum(axis=1) > 0).sum())
        print(f"  path {p}: feature-sums={np.round(tot,1)} nonzero_bins={nz}")
    # find a nonzero bin to see feature patterns
    nz_idx = np.argwhere(bins.sum(axis=(0, 2)) > 0)[:3]
    for t in nz_idx[:, 0]:
        print(f"  t={t}: per-path rows={bins[:, t, :].round(1).tolist()}")


def check_zip(dataset):
    import zipfile
    zp = os.path.join(DS_ROOT, dataset, "dataset.zip")
    if not os.path.exists(zp):
        print("no zip")
        return
    print("\nzip size: %.2f GB" % (os.path.getsize(zp) / 1e9))
    with zipfile.ZipFile(zp) as zf:
        names = zf.namelist()
        print("zip entries:", len(names))
        print("first entries:", names[:5])


def main():
    datasets = sys.argv[1:] or ["600vids_30samples_10s", "600vids_30samples_5s"]
    for ds in datasets:
        print("\n" + "#" * 30, ds, "#" * 30)
        inspect_npz_structure(ds)
        inspect_semantics(ds)
        scan_manifest(ds)
    check_zip("600vids_30samples_10s")


if __name__ == "__main__":
    main()
