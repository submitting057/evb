# -*- coding: utf-8 -*-
"""Per-sample inference cost benchmark (deployment view, strict 10s_nogro).

Answers: is LGBM the inference bottleneck of the 4-expert geo ensemble?

Measures, for ONE session / one sample:
  A. feature extraction
       load_events        npz read + per-event channel map (shared by all members)
       df encode          per-channel signed-size series encode (DFData init / n)
       segment_features_v2  120-d tabular vector for LGBM (incl. load_events)
  B. DF member forward (ngL e100 / dt / aux): batch=1 latency (pure GPU
     compute), batch=64 per-sample, CPU batch=1 reference for e100_L
  C. LGBM: deterministic replica of run_lgbm_v2 (seed 42, 386 classes,
     1500 rounds -> ~579k trees), then
       - single-row predict latency vs num_iteration (tree-budget scaling)
       - batch predict throughput
     plus a max|dprob| check against the cached probs (replica sanity).

Writes runs/infer_bench/report.json and prints a per-sample deployment budget.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import PROJECT_ROOT, RunConfig
from src.data import SessionIndex, load_events, load_splits, session_pool
from src.dfnet import DFData, DFNet, df_collate, load_series_cache
from src.segment import segment_features_v2


def t_ms(f, n=100, warm=5):
    for _ in range(warm):
        f()
    t0 = time.perf_counter()
    for _ in range(n):
        f()
    return (time.perf_counter() - t0) / n * 1e3


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="10s_nogro")
    ap.add_argument("--out", default="infer_bench")
    args = ap.parse_args()

    rep = {"config": vars(args)}

    rcfg = RunConfig(dataset=args.dataset, protocol="strict")
    splits = load_splits(rcfg.splits_path)
    index = SessionIndex(rcfg.dataset_dir, rcfg.bin_size_ms)
    pools = dict(
        val_probe=session_pool(splits, "known", "val", gallery=False),
        val_imp=session_pool(splits, "unknown", "val"),
    )
    rng = np.random.default_rng(0)
    sample = [pools["val_probe"][i]
              for i in rng.choice(len(pools["val_probe"]), 128, replace=False)]
    print(f"[setup] dataset={args.dataset} sample=128 val_probe sessions")

    # ------------------------------------------------------------- A: features
    print("\n=== A) per-session feature extraction ===")
    fe = {}

    for sid in sample[:10]:                    # warmup: cold npz reads + lazy imports
        load_events(index, sid)
    n_ev = []
    t0 = time.perf_counter()
    for sid in sample[10:60]:
        ev, chan, _ = load_events(index, sid)
        n_ev.append(len(ev))
    fe["load_events_ms"] = (time.perf_counter() - t0) / 50 * 1e3
    print(f"  load_events                {fe['load_events_ms']:8.2f} ms/session "
          f"(median {int(np.median(n_ev))} events)")

    t0 = time.perf_counter()
    for sid in sample[:50]:
        segment_features_v2(index, sid)
    fe["segment_features_v2_ms"] = (time.perf_counter() - t0) / 50 * 1e3
    print(f"  segment_features_v2        {fe['segment_features_v2_ms']:8.2f} ms/session "
          f"(incl. load_events -> incremental "
          f"{fe['segment_features_v2_ms'] - fe['load_events_ms']:.2f} ms)")
    rep["features"] = fe

    # ------------------------------------------------------- B: DF members fwd
    print("\n=== B) DF member forward (GPU) ===")
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cache = load_series_cache(index, args.dataset)
    labels = {s: 0 for s in sample}          # labels unused for inference timing

    df_rows = {}
    for tag in ("ngL_dfs_branch_all_e100", "ngL_dfs_branch_all_dt",
                "ngL_dfs_branch_aux"):
        ck = torch.load(PROJECT_ROOT / "runs" / tag / "model.pt",
                        map_location="cpu", weights_only=False)
        a = ck["args"]
        channels, fusion, chan_len = ck["channels"], ck["fusion"], ck["chan_len"]
        F = 2 if a["dt"] else 1
        model = DFNet(channels, fusion, n_class=len(splits["known_videos"]),
                      F=F, side_dim=0, chan_emb=a["chan_emb"], lens=chan_len)
        model.load_state_dict(ck["state"])
        model.to(device).eval()
        ds = DFData(cache, sample, labels, channels, fusion, a["value"],
                    a["dt"], chan_emb=a["chan_emb"], augment=False, lens=chan_len)
        t0 = time.perf_counter()
        one = [ds[i][0] for i in range(64)]          # __getitem__ pad cost
        pad_ms = (time.perf_counter() - t0) / 64 * 1e3

        b1 = []
        for blocks in one[:32]:
            xs = [torch.from_numpy(np.stack([b])) for b in blocks]
            b1.append([x.to(device) for x in xs])

        @torch.no_grad()
        def fwd_b1(xs, m=model):
            m(*xs)

        ms1 = t_ms(lambda: [fwd_b1(xs) for xs in b1], n=10, warm=2) / len(b1)
        if device == "cuda":
            torch.cuda.synchronize()

        loader64 = torch.utils.data.DataLoader(ds, batch_size=64,
                                               collate_fn=df_collate)
        batches = []
        for i, b in enumerate(loader64):
            *raw, y, idx = b
            xs = list(raw[0]) if len(raw) == 1 and isinstance(raw[0], (tuple, list)) \
                else raw
            batches.append([x.to(device) for x in xs])
            if i >= 1:
                break

        @torch.no_grad()
        def fwd_b64(bs, m=model):
            for xs in bs:
                m(*xs)

        if device == "cuda":
            torch.cuda.synchronize()
        ms64 = t_ms(lambda: fwd_b64(batches), n=10, warm=2) / (len(batches) * 64)

        row = {"batch1_ms": float(ms1), "batch64_per_sample_ms": float(ms64),
               "getitem_pad_ms": float(pad_ms)}
        if tag.endswith("e100") and device == "cuda":
            mc = DFNet(channels, fusion, n_class=len(splits["known_videos"]),
                       F=F, side_dim=0, chan_emb=a["chan_emb"], lens=chan_len)
            mc.load_state_dict(ck["state"])
            mc.eval()
            cb1 = [[torch.from_numpy(np.stack([b])) for b in blocks]
                   for blocks in one[:8]]

            @torch.no_grad()
            def fwd_cpu(bs):
                for xs in bs:
                    mc(*xs)

            row["cpu_batch1_ms"] = float(t_ms(lambda: fwd_cpu(cb1), n=3, warm=1)
                                         / len(cb1))
        df_rows[tag] = row
        print(f"  {tag:28s} b1={row['batch1_ms']:7.2f} ms "
              f"b64/sample={row['batch64_per_sample_ms']:7.3f} ms "
              f"pad={row['getitem_pad_ms']:6.3f} ms"
              + (f" cpu_b1={row['cpu_batch1_ms']:8.2f} ms"
                 if "cpu_batch1_ms" in row else ""))
        del model
        if device == "cuda":
            torch.cuda.empty_cache()
    rep["df"] = df_rows

    # ------------------------------------------------------------- C: LGBM
    print("\n=== C) LGBM (deterministic replica of run_lgbm_v2) ===")
    import lightgbm as lgb
    known = sorted(splits["known_videos"])
    vidx = {v: i for i, v in enumerate(known)}
    all_pools = dict(
        train=session_pool(splits, "known", "train"),
        val_probe=pools["val_probe"],
        val_imp=pools["val_imp"],
        test_probe=session_pool(splits, "known", "test", gallery=False),
        test_imp=session_pool(splits, "unknown", "test"),
    )
    all_sids = sorted({s for v in all_pools.values() for s in v})
    X = np.load(PROJECT_ROOT / "runs" / "cache" / f"segv2_{args.dataset}.npy")
    assert X.shape[0] == len(all_sids), (X.shape, len(all_sids))
    pos = {s: i for i, s in enumerate(all_sids)}
    y_tr = np.array([vidx[splits["sessions"][s]["video"]]
                     for s in all_pools["train"]], dtype=np.int64)

    model_path = PROJECT_ROOT / "runs" / "cache" / f"lgbm_model_{args.dataset}.txt"
    if model_path.exists():
        model = lgb.Booster(model_file=str(model_path))
        print(f"  [load] {model_path.name} trees={model.num_trees()}")
    else:
        params = dict(objective="multiclass", num_class=len(known),
                      metric="multi_logloss", learning_rate=0.05, num_leaves=63,
                      min_data_in_leaf=20, feature_fraction=0.7,
                      bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0,
                      verbosity=-1, num_threads=8, seed=42)
        t0 = time.time()
        model = lgb.train(params, lgb.Dataset(X[[pos[s] for s in all_pools["train"]]],
                                              y_tr),
                          num_boost_round=1500,
                          valid_sets=[lgb.Dataset(
                              X[[pos[s] for s in all_pools["val_probe"]]],
                              np.array([vidx[splits["sessions"][s]["video"]]
                                        for s in all_pools["val_probe"]],
                                       dtype=np.int64))],
                          callbacks=[lgbearly_stopping_rep()])
        print(f"  trained {time.time() - t0:.0f}s trees={model.num_trees()}")
        model.save_model(str(model_path))
        print(f"  [saved] {model_path.name}")

    # replica sanity vs cached probs
    z = np.load(PROJECT_ROOT / "runs" / "cache"
                / f"lgbm_probs_{args.dataset}_strict.npz")
    vp = X[[pos[s] for s in all_pools["val_probe"]]]
    p_new = model.predict(vp[:200], num_iteration=model.best_iteration)
    dmax = float(np.abs(p_new - z["val_probe"][:200]).max())
    print(f"  replica sanity max|dprob| vs cache: {dmax:.2e}")

    lg = {"n_trees": int(model.num_trees()),
          "best_iteration": int(model.best_iteration or 0), "replica_dprob": dmax}
    row_i = vp[0:1]
    lg["single_row_ms"] = float(t_ms(lambda: model.predict(row_i), n=50, warm=3))
    print(f"  single-row predict (all {lg['n_trees']} trees): "
          f"{lg['single_row_ms']:.1f} ms")
    lg["iter_scaling"] = {}
    for k in (25, 50, 100, 200, 400, 800):
        ms = float(t_ms(lambda: model.predict(row_i, num_iteration=k),
                        n=50, warm=3))
        lg["iter_scaling"][k] = ms
        print(f"    num_iteration={k:4d} ({k * 386:6d} trees): {ms:7.2f} ms")
    xb = vp[:256]
    lg["batch256_per_sample_ms"] = float(
        t_ms(lambda: model.predict(xb), n=10, warm=2)) / 256
    print(f"  batch-256 predict: {lg['batch256_per_sample_ms']:.2f} ms/sample")
    rep["lgbm"] = lg

    # ---------------------------------------------------------- budget summary
    e1 = df_rows["ngL_dfs_branch_all_e100"]["batch1_ms"]
    a1 = df_rows["ngL_dfs_branch_aux"]["batch1_ms"]
    seg_extra = fe["segment_features_v2_ms"] - fe["load_events_ms"]
    # encode cost measured via DFData init below (same section A? do quick here)
    t0 = time.perf_counter()
    DFData(cache, sample, labels,
           ck_channels := ["video_cdn", "api", "channel_info", "thumbnail", "other"],
           "branch", "logsize", False, augment=False,
           lens={"video_cdn": 8192, "api": 768})
    enc_ms = (time.perf_counter() - t0) / len(sample) * 1e3
    rep["features"]["df_encode_ms"] = float(enc_ms)
    print(f"\n  DFData per-session encode: {enc_ms:.2f} ms")

    feats = fe["load_events_ms"] + enc_ms + max(seg_extra, 0)
    fwd = e1 + a1
    lgm = lg["single_row_ms"]
    total = feats + fwd + lgm
    rep["budget_per_sample_ms"] = {
        "features": feats, "load_events": fe["load_events_ms"],
        "df_encode": enc_ms, "segv2_extra": seg_extra,
        "df_forward_e100_aux": fwd, "lgbm": lgm, "total": total,
        "lgbm_share": lgm / total,
    }
    print(f"\n=== per-sample deployment budget (3-member: lgbm+e100_L+aux_L) ===")
    print(f"  features  {feats:7.2f} ms  (load_events {fe['load_events_ms']:.2f}"
          f" + encode {enc_ms:.2f} + segv2 {seg_extra:.2f})")
    print(f"  DF fwd    {fwd:7.2f} ms  (e100_L {e1:.2f} + aux_L {a1:.2f})")
    print(f"  LGBM      {lgm:7.2f} ms  <- {lgm / total * 100:.0f}% of total")
    print(f"  TOTAL     {total:7.2f} ms/sample")

    out = PROJECT_ROOT / "runs" / args.out
    out.mkdir(parents=True, exist_ok=True)
    (out / "report.json").write_text(json.dumps(rep, indent=1, default=str),
                                     encoding="utf-8")
    print(f"[wrote] {out / 'report.json'}")


def lgbearly_stopping_rep():
    """Early stopping identical to run_lgbm_v2 (patience 100, silent)."""
    import lightgbm as lgb
    return lgb.early_stopping(100, verbose=False)


if __name__ == "__main__":
    main()
