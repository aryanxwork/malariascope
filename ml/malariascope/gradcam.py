"""Grad-CAM for the "Parasitized" logit + a dependency-free heatmap overlay (no matplotlib/OpenCV needed,
so the API container stays slim)."""

from __future__ import annotations

import numpy as np
import tensorflow as tf
from tensorflow import keras

from .models import FEATURE_LAYER


def make_gradcam_model(model: keras.Model) -> keras.Model:
    return keras.Model(model.inputs, [model.get_layer(FEATURE_LAYER).output, model.output])


def gradcam(grad_model: keras.Model, images: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (heatmaps [B, h, w] in [0, 1], logits [B]).

    Explains the evidence *for* "Parasitized" (Selvaraju et al., 2017): channel weights are the spatially
    averaged gradients of the logit w.r.t. the last feature map.
    """
    x = tf.convert_to_tensor(images, dtype=tf.float32)
    with tf.GradientTape() as tape:
        fmap, logits = grad_model(x, training=False)
        target = logits[:, 0]
    grads = tape.gradient(target, fmap)
    weights = tf.reduce_mean(grads, axis=(1, 2))  # [B, C]
    cam = tf.nn.relu(tf.reduce_sum(fmap * weights[:, None, None, :], axis=-1))  # [B, h, w]
    cam_max = tf.reduce_max(cam, axis=(1, 2), keepdims=True)
    cam = cam / tf.maximum(cam_max, 1e-12)
    return cam.numpy(), logits.numpy()[:, 0]


def turbo(x: np.ndarray) -> np.ndarray:
    """Polynomial approximation of Google's Turbo colormap. x in [0, 1] -> RGB float in [0, 1]."""
    x = np.clip(x, 0.0, 1.0)[..., None]
    r = 0.13572138 + x * (4.61539260 + x * (-42.66032258 + x * (132.13108234 + x * (-152.94239396 + x * 59.28637943))))
    g = 0.09140261 + x * (2.19418839 + x * (4.84296658 + x * (-14.18503333 + x * (4.27729857 + x * 2.82956604))))
    b = 0.10667330 + x * (12.64194608 + x * (-60.58204836 + x * (110.36276771 + x * (-89.90310912 + x * 27.34824973))))
    return np.clip(np.concatenate([r, g, b], axis=-1), 0.0, 1.0)


def overlay(image: np.ndarray, heatmap: np.ndarray, strength: float = 1.0, alpha: float = 0.55) -> np.ndarray:
    """Blend a heatmap onto an image. Blend weight follows heat, so cold regions keep the original pixels.

    ``strength`` (e.g. the predicted probability) fades the overlay for confidently-negative cells, whose
    per-image-normalised heatmap would otherwise exaggerate weak, meaningless activations.
    """
    img = np.clip(image, 0, 255).astype(np.float32) / 255.0
    h, w = img.shape[:2]
    hm = tf.image.resize(heatmap[..., None], (h, w), method="bilinear").numpy()[..., 0]
    hm = np.clip(hm, 0.0, 1.0)
    weight = (alpha * float(np.clip(strength, 0.0, 1.0)) * hm)[..., None]
    out = img * (1.0 - weight) + turbo(hm) * weight
    return (np.clip(out, 0.0, 1.0) * 255).round().astype(np.uint8)
