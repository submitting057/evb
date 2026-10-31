# -*- coding: utf-8 -*-
"""Backbones: BinTCN (time-bin sequences) and EventTransformer (packet event streams)."""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import EventTransformerConfig, SEMANTIC_CHANNELS, TcnConfig


# --------------------------------------------------------------------------- TCN

class TCNBlock(nn.Module):
    def __init__(self, ch: int, dilation: int, kernel: int = 3, dropout: float = 0.1):
        super().__init__()
        pad = (kernel - 1) * dilation // 2
        self.conv1 = nn.Conv1d(ch, ch, kernel, padding=pad, dilation=dilation)
        self.conv2 = nn.Conv1d(ch, ch, kernel, padding=pad, dilation=dilation)
        self.norm1 = nn.GroupNorm(8, ch)
        self.norm2 = nn.GroupNorm(8, ch)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, mask):  # x (B, C, T), mask (B, T) True=valid
        h = x
        h = self.drop(F.gelu(self.norm1(self.conv1(h))))
        h = self.drop(F.gelu(self.norm2(self.conv2(h))))
        return x + h


class MaskedAttentionPool(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.proj = nn.Linear(ch, ch)
        self.v = nn.Linear(ch, 1, bias=False)

    def forward(self, x, mask):  # x (B, C, T)
        h = torch.tanh(self.proj(x.transpose(1, 2)))     # (B, T, C)
        scores = self.v(h).squeeze(-1)                   # (B, T)
        scores = scores.masked_fill(~mask, -1e9)
        w = torch.softmax(scores, dim=-1).unsqueeze(-1)  # (B, T, 1)
        return (x @ w).squeeze(-1)                       # (B, C)


class BinTCN(nn.Module):
    """Input (B, 20, T) log1p bins + validity mask (B, T)."""

    def __init__(self, cfg: TcnConfig, in_channels: int = len(SEMANTIC_CHANNELS) * 4,
                 embed_dim: int = 128, dropout: float = 0.1):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, cfg.stem_channels, 5, padding=2),
            nn.GELU(), nn.GroupNorm(8, cfg.stem_channels),
        )
        self.down = nn.Conv1d(cfg.stem_channels, cfg.width, 1)
        self.blocks = nn.ModuleList([TCNBlock(cfg.width, d, dropout=dropout)
                                     for d in cfg.dilations])
        self.pool = MaskedAttentionPool(cfg.width)
        self.head = nn.Sequential(nn.Linear(cfg.width, embed_dim), nn.GELU(),
                                  nn.Dropout(dropout), nn.Linear(embed_dim, embed_dim))

    def forward(self, x, mask):
        h = self.stem(x)
        h = self.down(h)
        m = mask.unsqueeze(1)
        for blk in self.blocks:
            h = blk(h, mask)
            h = h * m.float()
        return F.normalize(self.head(self.pool(h, mask)), dim=-1)


# --------------------------------------------------------------------------- Event Transformer

class EventTransformer(nn.Module):
    """Input (B, N, 4) event feats, (B, N) channel ids (0..4), validity mask (B, N).

    Events are tokenized in non-overlapping groups of `token_stride` events by a
    strided conv, cutting attention length by 8x (4096 -> 512 tokens) so that the
    L x L attention matrices stay small on an 8GB GPU.
    """

    def __init__(self, cfg: EventTransformerConfig, in_dim: int = 4,
                 n_channels: int = len(SEMANTIC_CHANNELS), embed_dim: int = 128,
                 max_len: int = 4100, dropout: float = 0.1, token_stride: int = 8):
        super().__init__()
        self.stride = token_stride
        self.ch_emb = nn.Embedding(n_channels, cfg.channel_emb)
        self.in_proj = nn.Linear(in_dim + cfg.channel_emb, cfg.d_model)
        self.tokenizer = nn.Conv1d(cfg.d_model, cfg.d_model, token_stride, stride=token_stride)
        n_tokens = max_len // token_stride                 # 512
        self.cls = nn.Parameter(torch.zeros(1, 1, cfg.d_model))
        self.pos = nn.Parameter(torch.zeros(1, n_tokens + 1, cfg.d_model))
        nn.init.trunc_normal_(self.cls, std=0.02)
        nn.init.trunc_normal_(self.pos, std=0.02)
        layer = nn.TransformerEncoderLayer(
            cfg.d_model, cfg.nhead, cfg.ff_dim, dropout, batch_first=True,
            norm_first=True, activation="gelu")
        self.encoder = nn.TransformerEncoder(layer, cfg.layers,
                                             norm=nn.LayerNorm(cfg.d_model))
        self.head = nn.Sequential(nn.Linear(2 * cfg.d_model, embed_dim), nn.GELU(),
                                  nn.Dropout(dropout), nn.Linear(embed_dim, embed_dim))

    def forward(self, feats, chan, mask):
        B, N, _ = feats.shape
        tok = self.in_proj(torch.cat([feats, self.ch_emb(chan)], dim=-1))   # (B, N, D)
        x = tok.transpose(1, 2)                                             # (B, D, N)
        x = F.gelu(self.tokenizer(x))                                       # (B, D, N/8)
        m = F.max_pool1d(mask.float().unsqueeze(1), self.stride, self.stride).squeeze(1) > 0
        h = x.transpose(1, 2)                                               # (B, N/8, D)
        h = torch.cat([h, self.cls.expand(B, -1, -1)], dim=1)
        h = h + self.pos[:, :h.shape[1]]
        key_pad = torch.cat([~m, torch.zeros(B, 1, dtype=torch.bool, device=m.device)], dim=1)
        h = self.encoder(h, src_key_padding_mask=key_pad)
        cls_h = h[:, 0]
        w = m.unsqueeze(-1).float()
        mean_h = (h[:, 1:] * w).sum(1) / w.sum(1).clamp(min=1)
        return F.normalize(self.head(torch.cat([cls_h, mean_h], dim=-1)), dim=-1)


class SegEncoder(nn.Module):
    """Variable-length segment sequence (B, S, 5) + mask -> embedding."""

    def __init__(self, cfg: EventTransformerConfig, in_dim: int = 5,
                 embed_dim: int = 128, max_segs: int = 13, dropout: float = 0.1):
        super().__init__()
        self.in_proj = nn.Linear(in_dim, cfg.d_model)
        self.cls = nn.Parameter(torch.zeros(1, 1, cfg.d_model))
        self.pos = nn.Parameter(torch.zeros(1, max_segs + 1, cfg.d_model))
        nn.init.trunc_normal_(self.cls, std=0.02)
        nn.init.trunc_normal_(self.pos, std=0.02)
        layer = nn.TransformerEncoderLayer(
            cfg.d_model, cfg.nhead, cfg.ff_dim, dropout, batch_first=True,
            norm_first=True, activation="gelu")
        self.encoder = nn.TransformerEncoder(layer, cfg.layers,
                                             norm=nn.LayerNorm(cfg.d_model))
        self.head = nn.Sequential(nn.Linear(2 * cfg.d_model, embed_dim), nn.GELU(),
                                  nn.Dropout(dropout), nn.Linear(embed_dim, embed_dim))

    def forward(self, feats, mask):
        B, S, _ = feats.shape
        h = torch.cat([self.in_proj(feats), self.cls.expand(B, -1, -1)], dim=1)
        h = h + self.pos[:, :S + 1]
        key_pad = torch.cat([~mask, torch.zeros(B, 1, dtype=torch.bool, device=mask.device)],
                            dim=1)
        h = self.encoder(h, src_key_padding_mask=key_pad)
        w = mask.unsqueeze(-1).float()
        mean_h = (h[:, :S] * w).sum(1) / w.sum(1).clamp(min=1)
        return F.normalize(self.head(torch.cat([h[:, S], mean_h], dim=-1)), dim=-1)


class SegMLP(nn.Module):
    """Flattened segment-fingerprint MLP. Padded slots are exact zeros, so
    flattening (S, 5) -> (S*5,) is a valid fixed-dim encoding (order preserved)."""

    def __init__(self, in_dim: int = 5, n_segs: int = 12, embed_dim: int = 128,
                 dropout: float = 0.1):
        super().__init__()
        d = in_dim * n_segs
        self.net = nn.Sequential(
            nn.Linear(d, 512), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(512, 256), nn.GELU(),
            nn.Linear(256, embed_dim))

    def forward(self, feats, mask):
        x = feats * mask.unsqueeze(-1).float()
        return F.normalize(self.net(x.reshape(x.shape[0], -1)), dim=-1)


def build_model(rcfg, dropout: float = 0.1):
    if rcfg.rep == "bins":
        return BinTCN(rcfg.tcn, embed_dim=rcfg.train.embed_dim, dropout=dropout)
    if rcfg.rep == "events":
        return EventTransformer(rcfg.et, embed_dim=rcfg.train.embed_dim, dropout=dropout)
    if rcfg.rep == "segs":
        from .segment import MAX_SEGS
        return SegMLP(n_segs=MAX_SEGS, embed_dim=rcfg.train.embed_dim, dropout=dropout)
    raise ValueError(rcfg.rep)
