# -*- coding: utf-8 -*-
"""Distill the 579k-tree LGBM member into a tiny GPU MLP student.

Rationale: the LGBM expert's role in the geo ensemble is error-mode diversity
(a non-deep tabular model), not its standalone accuracy.  A student MLP trained
to MATCH the teacher's probability vectors keeps those error modes while making
the member's inference cost ~0 (a single small forward, batchable with the DF
members on the same GPU).

Teacher: runs/cache/lgbm_model_{dataset}.txt (deterministic replica saved by
run_infer_bench.py).  Student probs are written to the strict cache that
run_combiner consumes via --lgbm-suffix _mlp.

Usage:
  python scripts/run_lgbm_distill.py --dataset 10s_nogro
  python scripts/run_combiner.py --dataset 10s_nogro --protocol strict \
      --df-tags ngL_dfs_branch_all_e100,ngL_dfs_branch_all_dt,ngL_dfs_branch_aux \
      --lovo --lgbm-suffix _mlp --out combo_10s_nogro_L8k_mlp
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import lightgbm as lgb
import torch
import torch.nn as nn

from src.config import PROJECT_ROOT, RunConfig
from src.data import SessionIndex, load_splits, session_pool
from src.evaluate import _threshold_for_fpr


def norm(P):
    return P / (P.sum(1, keepdims=True) + 1e-12)


class Student(nn.Module):
    def __init__(self, d_in=120, n_class=386):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, 512), nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(512, 512), nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(512, n_class))

    def forward(self, x):
        return self.net(x)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="10s_nogro")
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--T", type=float, default=2.0)
    ap.add_argument("--hard-w", type=float, default=0.3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-suffix", default="mlp")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    rcfg = RunConfig(dataset=args.dataset, protocol="strict")
    rcfg.split.seed = args.seed
    splits = load_splits(rcfg.splits_path)
    known = sorted(splits["known_videos"])
    vidx = {v: i for i, v in enumerate(known)}
    pools = dict(
        train=session_pool(splits, "known", "train"),
        val_probe=session_pool(splits, "known", "val", gallery=False),
        val_imp=session_pool(splits, "unknown", "val"),
        test_probe=session_pool(splits, "known", "test", gallery=False),
        test_imp=session_pool(splits, "unknown", "test"),
    )
    all_sids = sorted({s for v in pools.values() for s in v})
    X = np.load(PROJECT_ROOT / "runs" / "cache" / f"segv2_{args.dataset}.npy")
    assert X.shape[0] == len(all_sids)
    pos = {s: i for i, s in enumerate(all_sids)}

    def y_of(pool):
        return np.array([vidx[splits["sessions"][s]["video"]] for s in pool],
                        dtype=np.int64)

    teacher = lgb.Booster(model_file=str(
        PROJECT_ROOT / "runs" / "cache" / f"lgbm_model_{args.dataset}.txt"))
    print(f"[teacher] trees={teacher.num_trees()}")

    # teacher probs for every pool (batched; predict cost is the reason this
    # member is being distilled, but batch mode is fast enough for one pass)
    t0 = time.time()
    Xall = X.astype(np.float32)
    TP = {}
    for name, pool in pools.items():
        rows = np.array([pos[s] for s in pool])
        out = []
        for i in range(0, len(rows), 512):
            out.append(teacher.predict(Xall[rows[i:i + 512]],
                                       num_iteration=teacher.best_iteration or None))
        TP[name] = np.concatenate(out).astype(np.float32)
    print(f"[teacher] probs for {sum(len(v) for v in pools.values())} rows "
          f"in {time.time() - t0:.0f}s")

    mu = Xall[[pos[s] for s in pools["train"]]].mean(0, keepdims=True)
    sd = Xall[[pos[s] for s in pools["train"]]].std(0, keepdims=True) + 1e-6
    Xn = (Xall - mu) / sd

    def pack(pool):
        r = torch.from_numpy(Xn[[pos[s] for s in pool]])
        return r

    tr_x, tr_y = pack(pools["train"]), torch.from_numpy(y_of(pools["train"]))
    tr_t = torch.from_numpy(TP["train"])
    va_x, va_y = pack(pools["val_probe"]), torch.from_numpy(y_of(pools["val_probe"]))

    model = Student(X.shape[1], len(known)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=1e-3, total_steps=args.epochs * ((len(tr_x) + 255) // 256),
        pct_start=0.1)
    T = args.T
    soft = (tr_t ** (1 / T))
    soft = soft / soft.sum(1, keepdims=True)

    best, hist = -1, {}
    for ep in range(args.epochs):
        model.train()
        perm = torch.randperm(len(tr_x))
        for i in range(0, len(tr_x), 256):
            j = perm[i:i + 256]
            x, y, t = tr_x[j].to(device), tr_y[j].to(device), soft[j].to(device)
            logits = model(x)
            kl = nn.functional.kl_div(
                torch.log_softmax(logits / T, 1), t, reduction="batchmean") * T * T
            ce = nn.functional.cross_entropy(logits, y)
            loss = (1 - args.hard_w) * kl + args.hard_w * ce
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
        model.eval()
        with torch.no_grad():
            pv = model(va_x.to(device)).softmax(1).cpu()
        top1 = float((pv.argmax(1) == va_y).float().mean())
        if top1 > best:
            best, hist = top1, {k: v for k, v in model.state_dict().items()}
        if ep % 20 == 19:
            print(f"ep{ep:03d} val_top1={top1:.4f} (best {best:.4f})")
    model.load_state_dict(hist)
    torch.save({"state": hist, "mu": mu, "sd": sd, "args": vars(args)},
               PROJECT_ROOT / "runs" / "cache" / f"lgbm_mlp_{args.dataset}.pt")

    # ---- member metrics + probs caches --------------------------------------
    probs_cache, row = {}, {"teacher_trees": int(teacher.num_trees())}
    model.eval()
    with torch.no_grad():
        # latency: batch-1 GPU forward (isolated compute + launch overhead)
        one = va_x[:1].to(device)
        for _ in range(5):
            model(one)
        if device == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(100):
            model(one)
        if device == "cuda":
            torch.cuda.synchronize()
        row["b1_gpu_ms"] = (time.perf_counter() - t0) / 100 * 1e3
        cpu_m = Student(X.shape[1], len(known))
        cpu_m.load_state_dict(hist)
        cpu_m.eval()
        t0 = time.perf_counter()
        for _ in range(100):
            cpu_m(va_x[:1])
        row["b1_cpu_ms"] = (time.perf_counter() - t0) / 100 * 1e3

        for split in ("val", "test"):
            P = []
            for name, pool in ((f"{split}_probe", pools[f"{split}_probe"]),
                               (f"{split}_imp", pools[f"{split}_imp"])):
                with torch.no_grad():
                    p = torch.cat([model(pack(pool)[i:i + 1024].to(device)).softmax(1)
                                   .cpu() for i in range(0, len(pool), 1024)])
                P.append(p.numpy())
            probs_cache[f"{split}_probe"], probs_cache[f"{split}_imp"] = P
            Pv, Iv = norm(P[0]), norm(P[1])
            y = y_of(pools[f"{split}_probe"])
            row[f"top1_{split}"] = float((Pv.argmax(1) == y).mean())
            co = Pv.argmax(1) == y
            row[f"tpr05_{split}"] = float(np.mean(
                co & (Pv.max(1) >= _threshold_for_fpr(Iv.max(1), 0.005))))

    # agreement with DF e100 on val impostors (independence proxy)
    d = np.load(PROJECT_ROOT / "runs" / "ngL_dfs_branch_all_e100"
                / "probs_val.npz")["imp"]
    row["agree_e100_valimp"] = float(
        (norm(probs_cache["val_imp"]).argmax(1) == norm(d).argmax(1)).mean())
    row["teacher_agree_e100_valimp"] = float(
        (norm(TP["val_imp"]).argmax(1) == norm(d).argmax(1)).mean())
    row["agree_teacher_valimp"] = float(
        (norm(probs_cache["val_imp"]).argmax(1)
         == norm(TP["val_imp"]).argmax(1)).mean())

    print(json.dumps({k: v for k, v in row.items() if not isinstance(v, float)
                      or True}, indent=1, default=str))
    np.savez(PROJECT_ROOT / "runs" / "cache"
             / f"lgbm_probs_{args.dataset}_strict_{args.out_suffix}.npz",
             **probs_cache)
    out = PROJECT_ROOT / "runs" / f"{args.dataset}_lgbm_distill"
    out.mkdir(parents=True, exist_ok=True)
    row["config"] = vars(args)
    (out / "report.json").write_text(json.dumps(row, indent=1, default=str),
                                     encoding="utf-8")
    print(f"[wrote] probs cache suffix _{args.out_suffix} + {out / 'report.json'}")


if __name__ == "__main__":
    main()
