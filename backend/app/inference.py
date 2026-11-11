"""Model loading, input validation and prediction (with calibrated probabilities and Grad-CAM)."""

from __future__ import annotations

import base64
import io
import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import tensorflow as tf
from PIL import Image, ImageOps, UnidentifiedImageError
from tensorflow import keras

from malariascope.calibration import sigmoid
from malariascope.data import preprocess_tensor
from malariascope.gradcam import gradcam, make_gradcam_model, overlay

log = logging.getLogger("malariascope.api")

ALLOWED_FORMATS = {"PNG", "JPEG", "BMP", "TIFF", "WEBP"}
DISPLAY_SIZE = 256


class InvalidImage(ValueError):
    """Raised for files that are not decodable images (-> HTTP 415/422)."""


@dataclass
class DecodedImage:
    filename: str
    pixels: np.ndarray  # float32 HxWx3 model input (img_size)
    warnings: list[str]


def decode_upload(filename: str, data: bytes, img_size: int) -> DecodedImage:
    if not data:
        raise InvalidImage("Empty file")
    try:
        with Image.open(io.BytesIO(data)) as probe:
            fmt = probe.format
            probe.verify()
        img = Image.open(io.BytesIO(data))
        img = ImageOps.exif_transpose(img).convert("RGB")
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as e:
        raise InvalidImage(f"Not a readable image ({e.__class__.__name__})") from e
    if fmt not in ALLOWED_FORMATS:
        raise InvalidImage(f"Unsupported format {fmt}; use PNG, JPEG, BMP, TIFF or WEBP")
    arr = np.asarray(img, dtype=np.uint8)
    pixels = preprocess_tensor(tf.constant(arr), img_size).numpy()
    return DecodedImage(filename, pixels, input_warnings(arr))


def input_warnings(arr: np.ndarray) -> list[str]:
    """Cheap out-of-distribution checks. The model was trained ONLY on single segmented RBC crops on a black
    background; anything else will still get a (meaningless) score, so we say so explicitly."""
    h, w = arr.shape[:2]
    warnings: list[str] = []
    if min(h, w) < 32:
        warnings.append("Very small image (<32 px); prediction may be unreliable.")
    if max(h, w) > 600:
        warnings.append("Large image: this model classifies single segmented cells, not whole fields of view.")
    b = max(1, int(0.06 * min(h, w)))
    border = np.concatenate([arr[:b].reshape(-1, 3), arr[-b:].reshape(-1, 3), arr[:, :b].reshape(-1, 3), arr[:, -b:].reshape(-1, 3)])
    if (border.max(axis=1) < 40).mean() < 0.35:
        warnings.append("Doesn't look like a segmented cell crop (expected a dark background around one cell).")
    return warnings


def _png_data_url(img: np.ndarray) -> str:
    buf = io.BytesIO()
    Image.fromarray(img).resize((DISPLAY_SIZE, DISPLAY_SIZE), Image.BILINEAR).save(buf, format="PNG", optimize=True)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


class Engine:
    def __init__(self, model_dir: Path):
        model_dir = Path(model_dir)
        self.model_dir = model_dir
        self.metadata = json.loads((model_dir / "metadata.json").read_text())
        mpath = model_dir / "metrics.json"
        self.metrics = json.loads(mpath.read_text()) if mpath.exists() else None
        self.model = keras.models.load_model(model_dir / "model.keras", compile=False)
        self.grad_model = make_gradcam_model(self.model)
        self.img_size = int(self.metadata["img_size"])
        self.temperature = float(self.metadata.get("temperature", 1.0))
        self.threshold = float(self.metadata.get("threshold", 0.5))
        self.review_margin = float(self.metadata.get("review_margin", 0.15))
        self._predict = tf.function(lambda x: self.model(x, training=False), reduce_retracing=True)
        self._predict(tf.zeros((1, self.img_size, self.img_size, 3)))  # warm-up / trace
        log.info("Loaded %s (%s) from %s", self.metadata.get("model"), self.metadata.get("run"), model_dir)

    def predict(self, images: list[DecodedImage], with_gradcam: bool = True) -> list[dict]:
        if not images:
            return []
        t0 = time.perf_counter()
        x = np.stack([im.pixels for im in images]).astype(np.float32)
        if with_gradcam:
            heat, logits = gradcam(self.grad_model, x)
        else:
            heat, logits = None, self._predict(tf.constant(x)).numpy()[:, 0]
        probs = sigmoid(np.asarray(logits, dtype=np.float64) / self.temperature)
        per_ms = (time.perf_counter() - t0) * 1000 / len(images)

        out = []
        for i, (im, p) in enumerate(zip(images, probs)):
            positive = bool(p >= self.threshold)
            out.append(
                {
                    "filename": im.filename,
                    "label": "Parasitized" if positive else "Uninfected",
                    "probability_parasitized": float(p),
                    "confidence": float(p if positive else 1 - p),
                    "threshold": self.threshold,
                    "needs_review": bool(abs(p - self.threshold) < self.review_margin),
                    "warnings": im.warnings,
                    "gradcam_png": _png_data_url(overlay(im.pixels, heat[i], strength=max(float(p), 0.15)))
                    if heat is not None
                    else None,
                    "latency_ms": round(per_ms, 2),
                }
            )
        return out

    def public_info(self) -> dict:
        return {"metadata": self.metadata, "metrics": self.metrics}
