# -*- coding: utf-8 -*-
"""Central configuration for the encrypted-video VID recognition project."""
from dataclasses import dataclass, field, asdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASETS_ROOT = Path(r"E:\evb\datasets")

DATASET_DIRS = {
    "10s": DATASETS_ROOT / "600vids_30samples_10s",
    "5s": DATASETS_ROOT / "600vids_30samples_5s",
}

# bins arrays on disk are (5, T, 4) with axis-0 channel order:
DISK_BIN_CHANNELS = ["video_cdn", "thumbnail", "channel_info", "api", "other"]
# features on axis-2: [down_bytes, up_bytes, down_pkts, up_pkts]
SEMANTIC_CHANNELS = ["video_cdn", "api", "channel_info", "thumbnail", "other"]
# map disk channel -> canonical order index used everywhere in this project
BIN_CHANNEL_PERM = [DISK_BIN_CHANNELS.index(c) for c in SEMANTIC_CHANNELS]

MAX_BINS = {"10s": 1500, "5s": 500}   # 10ms bins, duration cap 15s / 5s
MAX_EVENTS = 4096


@dataclass
class SplitConfig:
    seed: int = 42
    known_video_frac: float = 0.8        # 644 -> 520 known / 124 unknown
    train_frac: float = 0.6              # per known video
    val_frac: float = 0.2
    test_frac: float = 0.2
    unknown_val_frac: float = 0.5        # unknown sessions -> val/test impostor pools
    gallery_k: int = 3                   # enrollment sessions per known video (val & test each)


@dataclass
class TrainConfig:
    epochs: int = 120
    batch_p: int = 32                    # classes per batch
    batch_k: int = 4                     # sessions per class
    batch_p_events: int = 16             # smaller P for the events transformer (memory)
    lr: float = 1e-3
    weight_decay: float = 1e-4
    warmup_epochs: int = 3
    margin_warmup_epochs: int = 10       # arcface m ramps 0 -> m over this many epochs
    dropout: float = 0.1
    embed_dim: int = 128
    arcface_s: float = 32.0
    arcface_m: float = 0.5
    num_workers: int = 4
    eval_every: int = 1
    patience: int = 25                   # early stop epochs without val improvement
    min_epochs: int = 60                 # never early-stop before this epoch


@dataclass
class TcnConfig:
    width: int = 128
    stem_channels: int = 64
    dilations: tuple = (1, 2, 4, 8, 16, 32)


@dataclass
class EventTransformerConfig:
    d_model: int = 128
    nhead: int = 4
    ff_dim: int = 512
    layers: int = 6
    channel_emb: int = 8


@dataclass
class RunConfig:
    dataset: str = "10s"                 # '10s' | '5s'
    protocol: str = "legacy"             # 'legacy' | 'strict' (video-disjoint impostors)
    rep: str = "bins"                    # 'bins' | 'events'
    bin_size_ms: int = 10
    split: SplitConfig = field(default_factory=SplitConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    tcn: TcnConfig = field(default_factory=TcnConfig)
    et: EventTransformerConfig = field(default_factory=EventTransformerConfig)

    @property
    def dataset_dir(self) -> Path:
        return DATASET_DIRS[self.dataset]

    @property
    def run_name(self) -> str:
        return f"{self.dataset}_{self.rep}_s{self.split.seed}"

    @property
    def run_dir(self) -> Path:
        return PROJECT_ROOT / "runs" / self.run_name

    @property
    def splits_path(self) -> Path:
        name = f"splits_{self.dataset}_strict_s{self.split.seed}.json" \
            if self.protocol == "strict" else \
            f"splits_{self.dataset}_s{self.split.seed}.json"
        return PROJECT_ROOT / "splits" / name

    def to_dict(self) -> dict:
        return asdict(self)
