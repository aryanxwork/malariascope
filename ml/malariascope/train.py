"""Train a model end-to-end and produce a deployable, fully evaluated artifact.

    malariascope-train --config configs/efficientnet_b0.yaml
    malariascope-train --config configs/baseline_cnn.yaml --epochs-head 3 --max-samples 2000   # quick check
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
from pathlib import Path

import numpy as np

from .config import TrainConfig, load_config
from .data import class_weights, make_dataset, scan_dataset, split_dataset, subsample, summarize_splits, write_split_csv
from .evaluate import print_summary, promote, run_evaluation, write_json

log = logging.getLogger("malariascope")


def _compile(model, lr: float, cfg: TrainConfig):
    from tensorflow import keras

    model.compile(
        optimizer=keras.optimizers.AdamW(learning_rate=lr, weight_decay=cfg.weight_decay),
        loss=keras.losses.BinaryCrossentropy(from_logits=True, label_smoothing=cfg.label_smoothing),
        metrics=[keras.metrics.BinaryAccuracy(threshold=0.0, name="accuracy"), keras.metrics.AUC(from_logits=True, name="auc")],
    )


def _callbacks(run_dir: Path, cfg: TrainConfig):
    from tensorflow import keras

    return [
        keras.callbacks.EarlyStopping(monitor="val_auc", mode="max", patience=cfg.early_stopping_patience,
                                      restore_best_weights=True, verbose=1),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.3, patience=2, min_lr=1e-7, verbose=1),
        keras.callbacks.CSVLogger(str(run_dir / "training_log.csv"), append=True),
    ]


def _merge(h: dict, new: dict) -> dict:
    for k, v in new.items():
        h.setdefault(k, []).extend(float(x) for x in v)
    return h


def train(cfg: TrainConfig) -> dict:
    import tensorflow as tf
    from tensorflow import keras

    from .models import backbone, build_model, count_params, unfreeze_top

    keras.utils.set_random_seed(cfg.seed)
    run_id = f"{cfg.run_name}-{cfg.split_mode}-{time.strftime('%Y%m%d-%H%M%S')}"
    run_dir = Path(cfg.output_dir) / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    write_json(cfg.to_dict(), run_dir / "config.json")
    log.info("Run directory: %s", run_dir)
    log.info("TensorFlow %s | GPUs: %s", tf.__version__, tf.config.list_physical_devices("GPU"))

    # ---- data ----
    samples = subsample(scan_dataset(cfg.data_dir), cfg.max_samples, cfg.seed)
    n_pat = len({s.patient for s in samples})
    log.info("Found %d images from %d patient groups", len(samples), n_pat)
    splits = split_dataset(samples, cfg.split_mode, cfg.n_folds, cfg.val_fold, cfg.test_fold, cfg.seed)
    write_split_csv(splits, run_dir / "splits.csv")
    summary = summarize_splits(splits)
    log.info("Splits: %s", json.dumps(summary))

    train_ds = make_dataset(splits["train"], cfg.img_size, cfg.batch_size, True, cfg.seed, cfg.augment, cfg.cache)
    val_ds = make_dataset(splits["val"], cfg.img_size, cfg.batch_size, False, cache=cfg.cache)
    cw = class_weights(splits["train"])

    # ---- model ----
    model = build_model(cfg.model, cfg.img_size, cfg.dropout, cfg.weights, cfg.weight_decay)
    history: dict = {}
    t0 = time.perf_counter()

    # Phase 1: head only (or the whole network for the from-scratch baseline).
    _compile(model, cfg.lr_head, cfg)
    log.info("Phase 1: %s trainable params", f"{count_params(model)['trainable']:,}")
    h1 = model.fit(train_ds, validation_data=val_ds, epochs=cfg.epochs_head, class_weight=cw,
                   callbacks=_callbacks(run_dir, cfg), verbose=2)
    _merge(history, h1.history)
    finetune_start = None

    # Phase 2: fine-tune the top of the pretrained backbone at a low learning rate.
    if backbone(model) is not None and cfg.epochs_finetune > 0:
        finetune_start = len(history.get("loss", []))
        n = unfreeze_top(model, cfg.finetune_layers)
        _compile(model, cfg.lr_finetune, cfg)
        log.info("Phase 2: unfroze %d backbone layers (%s trainable params)", n, f"{count_params(model)['trainable']:,}")
        h2 = model.fit(train_ds, validation_data=val_ds, initial_epoch=finetune_start,
                       epochs=finetune_start + cfg.epochs_finetune, class_weight=cw,
                       callbacks=_callbacks(run_dir, cfg), verbose=2)
        _merge(history, h2.history)

    train_seconds = time.perf_counter() - t0
    model.save(run_dir / "model.keras")

    # ---- evaluation ----
    metrics = run_evaluation(model, splits, cfg, run_dir, history, finetune_start, count_params(model), train_seconds)
    promote(run_dir, Path(cfg.output_dir) / "latest")
    print_summary(metrics)
    log.info("Artifacts written to %s and promoted to %s", run_dir, Path(cfg.output_dir) / "latest")
    return metrics


def main(argv: list[str] | None = None) -> None:
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    ap = argparse.ArgumentParser(description="Train a MalariaScope model.")
    ap.add_argument("--config", help="YAML config (see configs/)")
    ap.add_argument("--data-dir")
    ap.add_argument("--output-dir")
    ap.add_argument("--run-name")
    ap.add_argument("--model")
    ap.add_argument("--weights", help="'imagenet' or 'none'")
    ap.add_argument("--split-mode", choices=["patient", "cell"])
    ap.add_argument("--img-size", type=int)
    ap.add_argument("--batch-size", type=int)
    ap.add_argument("--epochs-head", type=int)
    ap.add_argument("--epochs-finetune", type=int)
    ap.add_argument("--max-samples", type=int)
    ap.add_argument("--bootstrap-iterations", type=int)
    ap.add_argument("--seed", type=int)
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    overrides = {k: v for k, v in vars(a).items() if k != "config"}
    cfg = load_config(a.config, **overrides)
    np.set_printoptions(precision=4)
    train(cfg)


if __name__ == "__main__":
    main()
