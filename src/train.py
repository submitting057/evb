# -*- coding: utf-8 -*-
"""Metric-learning training loop shared by both representations."""
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .config import RunConfig
from .data import (PKSampler, PackedDataset, SessionIndex, load_splits,
                   session_pool)
from .evaluate import open_set_metrics
from .losses import ArcFaceHead
from .models import build_model


class Embedder:
    """Wraps a packed dataset + model to map session ids -> L2-normalized embeddings."""

    def __init__(self, model, dataset: PackedDataset, session_ids: list[str],
                 device, batch_size: int = 256):
        self.model, self.ds, self.device, self.bs = model, dataset, device, batch_size
        self.row = {sid: i for i, sid in enumerate(session_ids)}

    @torch.no_grad()
    def __call__(self, sids: list[str]) -> np.ndarray:
        rows = [self.row[s] for s in sids]
        out = []
        self.model.eval()
        for c0 in range(0, len(rows), self.bs):
            chunk = rows[c0:c0 + self.bs]
            if self.ds.rep in ("bins", "segs"):
                x = torch.stack([self.ds.x[i] for i in chunk]).to(self.device).float()
                mask = torch.stack([self.ds.mask[i] for i in chunk]).to(self.device)
                e = self.model(x, mask)
            else:
                x = torch.stack([self.ds.x[i] for i in chunk]).to(self.device).float()
                ch = torch.stack([self.ds.ch[i] for i in chunk]).to(self.device)
                mask = torch.stack([self.ds.mask[i] for i in chunk]).to(self.device)
                e = self.model(x, ch, mask)
            out.append(e.cpu())
        return F.normalize(torch.cat(out), dim=-1).numpy().astype(np.float32)


def train(rcfg: RunConfig) -> Path:
    torch.manual_seed(rcfg.split.seed)
    np.random.seed(rcfg.split.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    rcfg.run_dir.mkdir(parents=True, exist_ok=True)
    log_path = rcfg.run_dir / "train_log.txt"
    log_f = log_path.open("w", encoding="utf-8")

    def log(msg):
        print(msg, flush=True)
        log_f.write(msg + "\n")
        log_f.flush()

    log(f"config: {json.dumps(rcfg.to_dict())}")

    index = SessionIndex(rcfg.dataset_dir, rcfg.bin_size_ms)
    splits = load_splits(rcfg.splits_path)

    tr_sids = session_pool(splits, "known", "train")
    labels = [splits["sessions"][s]["video"] for s in tr_sids]
    known_videos = sorted(splits["known_videos"])
    vidx = {v: i for i, v in enumerate(known_videos)}
    y_all = np.array([vidx[v] for v in labels], dtype=np.int64)

    log(f"packing train {len(tr_sids)} sessions (rep={rcfg.rep}, augment={rcfg.rep == 'segs'}) ...")
    t0 = time.time()
    train_ds = PackedDataset(index, tr_sids, rcfg.rep, rcfg.dataset, seed=0,
                             augment=(rcfg.rep == "segs"))
    log(f"packed in {time.time() - t0:.0f}s")

    eval_sids = dict(
        val_gal=session_pool(splits, "known", "val", gallery=True),
        val_probe=session_pool(splits, "known", "val", gallery=False),
        val_imp=session_pool(splits, "unknown", "val"),
        test_gal=session_pool(splits, "known", "test", gallery=True),
        test_probe=session_pool(splits, "known", "test", gallery=False),
        test_imp=session_pool(splits, "unknown", "test"),
    )
    all_eval = sorted({s for v in eval_sids.values() for s in v})
    log(f"packing eval {len(all_eval)} sessions ...")
    t0 = time.time()
    eval_ds = PackedDataset(index, all_eval, rcfg.rep, rcfg.dataset, seed=0)
    log(f"packed in {time.time() - t0:.0f}s")

    tcfg = rcfg.train
    model = build_model(rcfg, dropout=tcfg.dropout).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    head = ArcFaceHead(tcfg.embed_dim, len(known_videos), tcfg.arcface_s, tcfg.arcface_m).to(device)
    log(f"model params: {n_params / 1e6:.2f}M")

    sampler = PKSampler(labels,
                        tcfg.batch_p if rcfg.rep == "bins" else tcfg.batch_p_events,
                        tcfg.batch_k, seed=rcfg.split.seed)
    loader = DataLoader(train_ds, batch_sampler=sampler, num_workers=0)
    opt = torch.optim.AdamW(list(model.parameters()) + list(head.parameters()),
                            lr=tcfg.lr, weight_decay=tcfg.weight_decay)
    steps_per_epoch = len(sampler)
    total_steps = steps_per_epoch * tcfg.epochs

    def lr_at(step):
        w = tcfg.warmup_epochs * steps_per_epoch
        if step < w:
            return step / max(1, w)
        p = (step - w) / max(1, total_steps - w)
        return 0.5 * (1 + math.cos(math.pi * p))

    emb = Embedder(model, eval_ds, all_eval, device)
    ckpt_path = rcfg.run_dir / "best.pt"
    best_score, best_epoch, no_improve = -1.0, -1, 0
    history = []

    def val_metrics():
        E = {k: emb(v) for k, v in eval_sids.items()}
        gal_by_video = {}
        for s in eval_sids["val_gal"]:
            gal_by_video.setdefault(splits["sessions"][s]["video"], []).append(s)
        vids = [v for v in known_videos if v in gal_by_video]
        G = np.stack([E["val_gal"][ [eval_sids["val_gal"].index(s) for s in gal_by_video[v]] ]
                      .mean(0) for v in vids])
        G = G / np.linalg.norm(G, axis=1, keepdims=True)
        mat = E["val_probe"] @ G.T
        y = np.array([vidx[splits["sessions"][s]["video"]] for s in eval_sids["val_probe"]])
        imp = (E["val_imp"] @ G.T).max(axis=1)
        return open_set_metrics(mat, y, imp, n_boot=0)

    for epoch in range(tcfg.epochs):
        sampler.set_epoch(epoch)
        model.train()
        # ArcFace margin warmup prevents early embedding collapse with many classes
        wm = min(1.0, (epoch + 1) / max(1, tcfg.margin_warmup_epochs))
        head.m = tcfg.arcface_m * wm
        t0, losses, tr_correct, tr_tot = time.time(), [], 0, 0
        for it, batch in enumerate(loader):
            step = epoch * steps_per_epoch + it
            for g in opt.param_groups:
                g["lr"] = tcfg.lr * lr_at(step)
            if rcfg.rep in ("bins", "segs"):
                x, mask, rows = batch
                x, mask = x.to(device), mask.to(device)
                e = model(x, mask)
            else:
                x, ch, mask, rows = batch
                x, ch, mask = x.to(device), ch.to(device), mask.to(device)
                e = model(x, ch, mask)
            rows_t = rows if torch.is_tensor(rows) else torch.tensor(rows)
            y = torch.from_numpy(y_all)[rows_t.long()].to(device)
            logits = head(e, y)
            loss = torch.nn.functional.cross_entropy(logits, y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            losses.append(loss.item())
            tr_correct += (logits.argmax(1) == y).sum().item()
            tr_tot += len(y)
        msg = (f"epoch {epoch + 1}/{tcfg.epochs} loss={np.mean(losses):.4f} "
               f"tracc={tr_correct / max(1, tr_tot):.3f} ({time.time() - t0:.0f}s)")
        if (epoch + 1) % tcfg.eval_every == 0 or epoch == tcfg.epochs - 1:
            m = val_metrics()
            score = 0.5 * m["closed"]["top1"] + 0.5 * m["fpr"][0.01]["tpr"]
            sel = score + 1e-4 * m["closed"]["top1"]
            msg += (f" | val top1={m['closed']['top1']:.4f}"
                    f" TPR@0.5%={m['fpr'][0.005]['tpr']:.4f}"
                    f" TPR@1%={m['fpr'][0.01]['tpr']:.4f}"
                    f" TPR@5%={m['fpr'][0.05]['tpr']:.4f}")
            history.append({"epoch": epoch + 1, "loss": float(np.mean(losses)), "val": m})
            if sel > best_score:
                best_score, best_epoch, no_improve = sel, epoch + 1, 0
                torch.save({"model": model.state_dict(), "head": head.state_dict(),
                            "epoch": best_epoch, "config": rcfg.to_dict()}, ckpt_path)
                msg += "  *best*"
            else:
                no_improve += 1
        log(msg)
        if no_improve >= tcfg.patience and (epoch + 1) >= tcfg.min_epochs:
            log(f"early stop at epoch {epoch + 1} (best={best_epoch})")
            break

    (rcfg.run_dir / "history.json").write_text(json.dumps(history, indent=1, default=str),
                                               encoding="utf-8")
    log(f"done. best epoch={best_epoch} ckpt={ckpt_path}")
    return ckpt_path


def main():
    dataset = sys.argv[1] if len(sys.argv) > 1 else "10s"
    rep = sys.argv[2] if len(sys.argv) > 2 else "bins"
    rcfg = RunConfig(dataset=dataset, rep=rep)
    train(rcfg)


if __name__ == "__main__":
    main()
