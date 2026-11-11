"""Training configuration: a dataclass that can be loaded from YAML and overridden from the CLI."""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any

MODEL_CHOICES = ("baseline_cnn", "efficientnet_b0", "mobilenet_v3_small")
SPLIT_MODES = ("patient", "cell")


@dataclass
class TrainConfig:
    # --- experiment ---
    run_name: str = "efficientnet_b0"
    data_dir: str = "data/cell_images"
    output_dir: str = "artifacts"
    seed: int = 42

    # --- model ---
    model: str = "efficientnet_b0"
    weights: str | None = "imagenet"  # "imagenet" or None (random init); ignored by baseline_cnn
    img_size: int = 128
    dropout: float = 0.3

    # --- data ---
    split_mode: str = "patient"  # "patient" (no leakage) or "cell" (naive random split, for the ablation)
    n_folds: int = 7  # StratifiedGroupKFold folds -> test = 1 fold, val = 1 fold (~14% each)
    test_fold: int = 0
    val_fold: int = 1
    max_samples: int | None = None  # subsample for quick experiments
    cache: bool = True
    augment: bool = True

    # --- optimisation ---
    batch_size: int = 64
    epochs_head: int = 8  # phase 1: frozen backbone (or all epochs for baseline_cnn)
    epochs_finetune: int = 10  # phase 2: unfreeze top layers of the backbone
    lr_head: float = 1e-3
    lr_finetune: float = 1e-5
    finetune_layers: int = 60
    weight_decay: float = 1e-4
    label_smoothing: float = 0.0
    early_stopping_patience: int = 4

    # --- decision policy ---
    # Screening tool: a missed parasitized cell costs more than a false alarm, so the operating threshold
    # is chosen on the *validation* set as the highest threshold that reaches this sensitivity.
    target_sensitivity: float = 0.98
    review_margin: float = 0.15  # |p - threshold| below this -> flagged "needs review"

    # --- evaluation ---
    bootstrap_iterations: int = 1000
    gradcam_examples: int = 8

    def validate(self) -> "TrainConfig":
        if self.model not in MODEL_CHOICES:
            raise ValueError(f"model must be one of {MODEL_CHOICES}, got {self.model!r}")
        if self.split_mode not in SPLIT_MODES:
            raise ValueError(f"split_mode must be one of {SPLIT_MODES}, got {self.split_mode!r}")
        if self.val_fold == self.test_fold:
            raise ValueError("val_fold and test_fold must differ")
        if not (0 < self.target_sensitivity <= 1):
            raise ValueError("target_sensitivity must be in (0, 1]")
        if isinstance(self.weights, str) and self.weights.lower() in {"none", "null", ""}:
            self.weights = None
        return self

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_config(path: str | Path | None = None, **overrides: Any) -> TrainConfig:
    data: dict[str, Any] = {}
    if path:
        import yaml  # optional dependency (train extra)

        with open(path, encoding="utf-8") as fh:
            data = yaml.safe_load(fh) or {}
    known = {f.name for f in fields(TrainConfig)}
    unknown = set(data) - known
    if unknown:
        raise ValueError(f"Unknown config keys in {path}: {sorted(unknown)}")
    data.update({k: v for k, v in overrides.items() if v is not None})
    return TrainConfig(**data).validate()
