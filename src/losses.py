# -*- coding: utf-8 -*-
"""ArcFace metric-learning head."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ArcFaceHead(nn.Module):
    def __init__(self, embed_dim: int, n_classes: int, s: float = 32.0, m: float = 0.5):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(n_classes, embed_dim))
        nn.init.xavier_uniform_(self.weight)
        self.s, self.m = s, m

    def forward(self, emb: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """emb: L2-normalized (B, D); returns logits (B, n_classes)."""
        cos = F.linear(emb, F.normalize(self.weight, dim=-1))       # (B, C) in [-1, 1]
        clamp = torch.clamp(cos, -1 + 1e-7, 1 - 1e-7)
        theta_m = torch.acos(clamp[torch.arange(len(labels), device=labels.device), labels]) + self.m
        target_cos = torch.cos(theta_m)
        logits = cos.clone()
        logits[torch.arange(len(labels), device=labels.device), labels] = target_cos
        return logits * self.s
