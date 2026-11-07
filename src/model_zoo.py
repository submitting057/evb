# -*- coding: utf-8 -*-
"""Alternative expert trunks to replace the 1998-era DF-CNN (architecture axis).

All trunks consume the same DFData input (per-channel packet-index signed-size
series, F=1 or F=2 with the dt feature) and expose the DFTrunk-compatible
contract:  forward(x) -> flattened feature row,  out_len(lin) -> feature dim.

Downsampling is always MaxPool1d(4, 4, ceil_mode=True) -- the exact length
semantics of the DF funnel (4 halv... quarterings per channel, short channels
never collapse), so channel lengths stay interchangeable with DF members.

Architectures (sized to the DF branch budget, ~4-6M params on the vid channel
incl. the shared 512-wide classifier head):
  resnet : stem conv k15 + 4 stages x 2 BasicBlock1d (residual depth instead
           of the DF funnel's single conv pair per scale)
  tcn    : 4 groups x 2 gated dilated conv blocks (non-causal, symmetric pad),
           in-group dilations 1,2 and group-equivalent 4,8 / 16,32 / 64,128
           (large receptive field; the gate doubles as a soft attention)
  vit    : patchify (conv k=16 s=16) -> 4 pre-norm transformer encoder layers
           -> mean+max pooling (global context, no convolution locality bias)
"""
import torch
import torch.nn as nn


def _q4(l: int) -> int:
    """MaxPool1d(4,4,ceil) length update."""
    return max(-(-l // 4), 1)


# ----------------------------------------------------------------- resnet

class BasicBlock1d(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.c1 = nn.Conv1d(cin, cout, 3, padding=1, bias=False)
        self.b1 = nn.BatchNorm1d(cout)
        self.c2 = nn.Conv1d(cout, cout, 3, padding=1, bias=False)
        self.b2 = nn.BatchNorm1d(cout)
        self.skip = (nn.Identity() if cin == cout else
                     nn.Conv1d(cin, cout, 1, bias=False))
        self.act = nn.ReLU()

    def forward(self, x):
        y = self.act(self.b1(self.c1(x)))
        y = self.b2(self.c2(y))
        return self.act(y + self.skip(x))


class ResNetTrunk(nn.Module):
    WIDTH = 256          # flatten = WIDTH * slots

    def __init__(self, cin: int, widths=(32, 64, 128, 256), blocks_per_stage=2):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(cin, widths[0], 15, padding=7, bias=False),
            nn.BatchNorm1d(widths[0]), nn.ReLU())
        chans = (widths[0],) + tuple(widths)
        stages = []
        for k, w in enumerate(widths):
            blocks = [BasicBlock1d(chans[k] if b == 0 else w, w)
                      for b in range(blocks_per_stage)]
            stages.append(nn.Sequential(*blocks, nn.MaxPool1d(4, 4, ceil_mode=True)))
        self.body = nn.Sequential(*stages)

    def out_len(self, lin: int) -> int:
        l = lin
        for _ in range(4):
            l = _q4(l)
        return self.WIDTH * l

    def forward(self, x):
        return self.body(self.stem(x)).flatten(1)


# ----------------------------------------------------------------- tcn

class GatedDilatedBlock(nn.Module):
    """Non-causal gated dilated residual block (WaveNet-style gate, symmetric
    padding so the packet-index series is scanned in both directions)."""

    def __init__(self, cin, cout, dilation):
        super().__init__()
        pad = dilation
        self.t = nn.Conv1d(cin, cout, 3, padding=pad, dilation=dilation, bias=False)
        self.g = nn.Conv1d(cin, cout, 3, padding=pad, dilation=dilation, bias=False)
        self.bt, self.bg = nn.BatchNorm1d(cout), nn.BatchNorm1d(cout)
        self.skip = (nn.Identity() if cin == cout else
                     nn.Conv1d(cin, cout, 1, bias=False))

    def forward(self, x):
        h = torch.tanh(self.bt(self.t(x))) * torch.sigmoid(self.bg(self.g(x)))
        return h + self.skip(x)


class TCNTrunk(nn.Module):
    WIDTH = 256

    def __init__(self, cin: int, widths=(64, 128, 192, 256)):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(cin, widths[0], 7, padding=3, bias=False),
            nn.BatchNorm1d(widths[0]), nn.ReLU())
        chans = (widths[0],) + tuple(widths)
        groups = []
        for k, w in enumerate(widths):
            for b in (0, 1):
                groups.append(GatedDilatedBlock(chans[k] if b == 0 else w, w,
                                                (1, 2)[b] * (4 ** k)))
            groups.append(nn.MaxPool1d(4, 4, ceil_mode=True))
        self.body = nn.Sequential(*groups)
        self.act = nn.ReLU()

    def out_len(self, lin: int) -> int:
        l = lin
        for _ in range(4):
            l = _q4(l)
        return self.WIDTH * l

    def forward(self, x):
        return self.act(self.body(self.stem(x))).flatten(1)


# ----------------------------------------------------------------- vit

class PatchTransformerTrunk(nn.Module):
    def __init__(self, cin: int, lin: int, d_model=256, patch=16,
                 layers=4, nhead=8, ff=1024, drop=0.1):
        super().__init__()
        self.patch = patch
        self.embed = nn.Conv1d(cin, d_model, patch, stride=patch)
        n_tok = -(-lin // patch)
        self.pos = nn.Parameter(torch.zeros(1, n_tok, d_model))
        nn.init.trunc_normal_(self.pos, std=0.02)
        layer = nn.TransformerEncoderLayer(
            d_model, nhead, ff, dropout=drop, activation="gelu",
            batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(layer, layers)
        self.norm = nn.LayerNorm(d_model)
        self.feat = 2 * d_model          # mean + max pooling

    def out_len(self, lin: int) -> int:
        return self.feat                # pooled, length-independent

    def forward(self, x):
        t = self.embed(x).transpose(1, 2)
        t = t + self.pos[:, : t.shape[1]]
        t = self.norm(self.enc(t))
        return torch.cat([t.mean(1), t.max(1).values], -1)


# ----------------------------------------------------------------- factory

TRUNKS = {"resnet": ResNetTrunk, "tcn": TCNTrunk, "vit": PatchTransformerTrunk}


def make_trunk(arch: str, cin: int, lin: int):
    if arch in ("resnet", "tcn"):
        return TRUNKS[arch](cin)
    return TRUNKS[arch](cin, lin)       # vit needs lin for pos table


class BranchNet(nn.Module):
    """Multi-channel branch network over any trunk in this zoo (drop-in
    replacement for DFNet(fusion='branch') in scripts/run_arch.py)."""

    def __init__(self, arch: str, channels: list[str], lens: list[int],
                 n_class: int, F: int = 1, head_drop=(0.7, 0.5)):
        super().__init__()
        self.trunks = nn.ModuleList(make_trunk(arch, F, l) for l in lens)
        self._flat = flat = sum(t.out_len(l) for t, l in zip(self.trunks, lens))
        self.head = nn.Sequential(
            nn.Linear(flat, 512), nn.BatchNorm1d(512), nn.ReLU(),
            nn.Dropout(head_drop[0]),
            nn.Linear(512, 512), nn.BatchNorm1d(512), nn.ReLU(),
            nn.Dropout(head_drop[1]),
            nn.Linear(512, n_class))

    def forward(self, *xs):
        f = torch.cat([t(x) for t, x in zip(self.trunks, xs)], 1)
        return self.head(f)
