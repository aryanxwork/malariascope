"""Dataset discovery, leakage-free patient-level splitting, and tf.data pipelines.

Why patient-level splitting matters
-----------------------------------
The NIH dataset holds 27,558 cell crops from ~200 patients, so each patient contributes ~100+ visually
similar cells (same slide, same stain batch, same microscope). A naive random *cell-level* split puts cells
from the same patient in both train and test, and the model gets credit for recognising the slide rather
than the parasite. Rajaraman et al. (2018) report 98.61% cell-level vs 95.90% patient-level accuracy for
the same family of models. This module therefore splits by patient by default and exposes the naive
split only as an ablation (`split_mode="cell"`).

Patient IDs are encoded in NIH file names, e.g. ``C33P1thinF_IMG_20150619_114756a_cell_179.png`` -> ``P1``.
Healthy-donor files look like ``C1_thinF_IMG_..._cell_9.png`` and are grouped by their slide prefix.
"""

from __future__ import annotations

import csv
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import tensorflow as tf

CLASS_DIRS = {"Uninfected": 0, "Parasitized": 1}
IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}

_INFECTED_PATIENT_RE = re.compile(r"^C\d+P(\d+)", re.IGNORECASE)
_SLIDE_RE = re.compile(r"^(C\d+)", re.IGNORECASE)


@dataclass(frozen=True)
class Sample:
    path: str
    label: int
    patient: str


def patient_id_from_filename(name: str) -> str:
    """Map an NIH cell filename to a patient group key.

    Grouping is deliberately conservative: if two files *might* share a patient they share a key, because
    over-merging only makes splits coarser, while under-merging leaks data.
    """
    stem = Path(name).stem
    m = _INFECTED_PATIENT_RE.match(stem)
    if m:
        return f"P{int(m.group(1))}"
    m = _SLIDE_RE.match(stem)
    if m:
        return f"H-{m.group(1).upper()}"
    # Unknown naming scheme: each image becomes its own group (equivalent to a cell-level split).
    return f"U-{stem}"


def scan_dataset(data_dir: str | Path) -> list[Sample]:
    """Find ``Parasitized/`` and ``Uninfected/`` folders anywhere under ``data_dir``.

    Handles both the official NIH zip and the Kaggle mirror (which ships a duplicated nested copy):
    files are de-duplicated per class by file name.
    """
    root = Path(data_dir)
    if not root.exists():
        raise FileNotFoundError(f"Data directory not found: {root}. Run `python scripts/download_data.py` first.")
    samples: list[Sample] = []
    for cls, label in CLASS_DIRS.items():
        dirs = sorted(p for p in root.rglob(cls) if p.is_dir())
        if root.name == cls:
            dirs.insert(0, root)
        if not dirs:
            raise FileNotFoundError(f"No '{cls}' folder found under {root}")
        seen: set[str] = set()
        for d in dirs:
            for f in sorted(d.iterdir()):
                if f.suffix.lower() in IMG_EXTS and f.name not in seen:
                    seen.add(f.name)
                    samples.append(Sample(str(f), label, patient_id_from_filename(f.name)))
    if not samples:
        raise FileNotFoundError(f"No images found under {root}")
    return samples


def subsample(samples: list[Sample], n: int, seed: int) -> list[Sample]:
    if n is None or n >= len(samples):
        return samples
    rng = np.random.default_rng(seed)
    idx = np.sort(rng.choice(len(samples), size=n, replace=False))
    return [samples[i] for i in idx]


def split_dataset(
    samples: list[Sample],
    mode: str = "patient",
    n_folds: int = 7,
    val_fold: int = 1,
    test_fold: int = 0,
    seed: int = 42,
) -> dict[str, list[Sample]]:
    """Split into train/val/test. ``mode="patient"`` guarantees no patient appears in two splits."""
    from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold

    y = np.array([s.label for s in samples])
    x = np.zeros(len(samples))
    if mode == "patient":
        groups = np.array([s.patient for s in samples])
        n_groups = len(set(groups))
        if n_groups < n_folds:
            raise ValueError(f"Only {n_groups} patient groups found; need at least n_folds={n_folds}.")
        folds = list(StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=seed).split(x, y, groups))
    elif mode == "cell":
        folds = list(StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed).split(x, y))
    else:
        raise ValueError(f"Unknown split mode {mode!r}")

    test_idx = set(folds[test_fold][1].tolist())
    val_idx = set(folds[val_fold][1].tolist())
    splits: dict[str, list[Sample]] = {"train": [], "val": [], "test": []}
    for i, s in enumerate(samples):
        if i in test_idx:
            splits["test"].append(s)
        elif i in val_idx:
            splits["val"].append(s)
        else:
            splits["train"].append(s)

    if mode == "patient":
        overlap = patient_overlap(splits)
        if any(overlap.values()):
            raise AssertionError(f"Patient leakage detected between splits: {overlap}")
    return splits


def patient_overlap(splits: dict[str, list[Sample]]) -> dict[str, int]:
    pats = {k: {s.patient for s in v} for k, v in splits.items()}
    return {
        "train_val": len(pats["train"] & pats["val"]),
        "train_test": len(pats["train"] & pats["test"]),
        "val_test": len(pats["val"] & pats["test"]),
    }


def summarize_splits(splits: dict[str, list[Sample]]) -> dict[str, dict[str, int]]:
    out = {}
    for name, items in splits.items():
        c = Counter(s.label for s in items)
        out[name] = {
            "cells": len(items),
            "parasitized": c.get(1, 0),
            "uninfected": c.get(0, 0),
            "patients": len({s.patient for s in items}),
        }
    out["leakage"] = patient_overlap(splits)
    return out


def write_split_csv(splits: dict[str, list[Sample]], path: str | Path) -> None:
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["split", "path", "label", "patient"])
        for name, items in splits.items():
            for s in items:
                w.writerow([name, s.path, s.label, s.patient])


def class_weights(samples: list[Sample]) -> dict[int, float]:
    c = Counter(s.label for s in samples)
    total = sum(c.values())
    return {k: total / (len(c) * v) for k, v in c.items()}


def patients_by_split(splits: dict[str, list[Sample]]) -> dict[str, list[str]]:
    d: dict[str, set[str]] = defaultdict(set)
    for k, v in splits.items():
        for s in v:
            d[k].add(s.patient)
    return {k: sorted(v) for k, v in d.items()}


# ----------------------------------------------------------------------------------------------------------
# Image preprocessing — ONE function used by training, evaluation and the API (no train/serve skew).
# ----------------------------------------------------------------------------------------------------------


def preprocess_tensor(image: tf.Tensor, img_size: int) -> tf.Tensor:
    """uint8/float HxWx3 -> float32 img_size x img_size x 3 in [0, 255].

    Aspect ratio is preserved with black padding, which matches the black background of the NIH crops.
    Pixel scaling is done *inside* each model, so the served model always takes raw 0-255 pixels.
    """
    image = tf.cast(image, tf.float32)
    image = tf.image.resize_with_pad(image, img_size, img_size, method="bilinear", antialias=True)
    return tf.clip_by_value(image, 0.0, 255.0)


def decode_image_bytes(data: tf.Tensor) -> tf.Tensor:
    img = tf.io.decode_image(data, channels=3, expand_animations=False)
    img.set_shape([None, None, 3])
    return img


def preprocess_bytes(data: bytes, img_size: int) -> np.ndarray:
    """Used by the API: raw file bytes -> (img_size, img_size, 3) float32 array."""
    return preprocess_tensor(decode_image_bytes(tf.constant(data)), img_size).numpy()


def _load(path: tf.Tensor, img_size: int) -> tf.Tensor:
    img = preprocess_tensor(decode_image_bytes(tf.io.read_file(path)), img_size)
    return tf.cast(tf.round(img), tf.uint8)  # uint8 keeps the cache 4x smaller


def stain_jitter(image: tf.Tensor) -> tf.Tensor:
    """Simulate Giemsa stain / white-balance variation between slides and microscopes."""
    x = image / 255.0
    x = tf.image.random_hue(x, 0.03)
    x = tf.image.random_saturation(x, 0.8, 1.2)
    x = tf.image.random_brightness(x, 0.06)
    x = tf.image.random_contrast(x, 0.85, 1.15)
    return tf.clip_by_value(x, 0.0, 1.0) * 255.0


def geometric_augmenter(seed: int) -> tf.keras.Sequential:
    # Cells have no canonical orientation -> full rotations and flips are label-preserving.
    from tensorflow import keras

    return keras.Sequential(
        [
            keras.layers.RandomFlip("horizontal_and_vertical", seed=seed),
            keras.layers.RandomRotation(0.5, fill_mode="constant", fill_value=0.0, seed=seed),
            keras.layers.RandomZoom((-0.1, 0.1), fill_mode="constant", fill_value=0.0, seed=seed),
        ],
        name="augment",
    )


def make_dataset(
    samples: list[Sample],
    img_size: int,
    batch_size: int,
    training: bool,
    seed: int = 42,
    augment: bool = True,
    cache: bool = True,
) -> tf.data.Dataset:
    paths = [s.path for s in samples]
    labels = np.array([s.label for s in samples], dtype=np.float32)
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    ds = ds.map(lambda p, y: (_load(p, img_size), y), num_parallel_calls=tf.data.AUTOTUNE)
    if cache:
        ds = ds.cache()
    if training:
        ds = ds.shuffle(min(len(samples), 10_000), seed=seed, reshuffle_each_iteration=True)
    ds = ds.map(lambda x, y: (tf.cast(x, tf.float32), y), num_parallel_calls=tf.data.AUTOTUNE)
    if training and augment:
        ds = ds.map(lambda x, y: (stain_jitter(x), y), num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size)
    if training and augment:
        aug = geometric_augmenter(seed)
        ds = ds.map(lambda x, y: (aug(x, training=True), y), num_parallel_calls=tf.data.AUTOTUNE)
    return ds.prefetch(tf.data.AUTOTUNE)


def load_images(paths: list[str], img_size: int) -> np.ndarray:
    return np.stack([_load(tf.constant(p), img_size).numpy() for p in paths]).astype(np.float32)
