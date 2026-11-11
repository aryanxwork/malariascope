"""Export to TensorFlow Lite (fp32 / fp16 / dynamic-range int8) and benchmark size, latency and agreement
with the Keras model — evidence for "can this run on a cheap phone attached to a microscope?".

    malariascope-export --model-dir artifacts/latest --data-dir data/cell_images --n 500
"""

from __future__ import annotations

import argparse
import json
import tempfile
import time
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow import keras

from .calibration import sigmoid


def _interpreter(path: Path):
    try:  # LiteRT is the new home of the TFLite runtime
        from ai_edge_litert.interpreter import Interpreter  # type: ignore
    except ImportError:
        Interpreter = tf.lite.Interpreter
    interp = Interpreter(model_path=str(path), num_threads=1)
    interp.allocate_tensors()
    return interp


def _run_tflite(interp, images: np.ndarray) -> tuple[np.ndarray, float]:
    inp = interp.get_input_details()[0]
    out = interp.get_output_details()[0]
    logits, times = [], []
    for img in images:
        interp.set_tensor(inp["index"], img[None].astype(np.float32))
        t0 = time.perf_counter()
        interp.invoke()
        times.append(time.perf_counter() - t0)
        logits.append(float(interp.get_tensor(out["index"])[0, 0]))
    return np.array(logits), float(np.median(times) * 1000)


def convert(model: keras.Model, out_dir: Path) -> dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        saved = Path(tmp) / "saved_model"
        model.export(str(saved), format="tf_saved_model", verbose=False)
        variants = {}
        for name in ("fp32", "fp16", "int8_dynamic"):
            conv = tf.lite.TFLiteConverter.from_saved_model(str(saved))
            if name != "fp32":
                conv.optimizations = [tf.lite.Optimize.DEFAULT]
            if name == "fp16":
                conv.target_spec.supported_types = [tf.float16]
            path = out_dir / f"model_{name}.tflite"
            path.write_bytes(conv.convert())
            variants[name] = path
    return variants


def benchmark(model_dir: Path, images: np.ndarray, labels: np.ndarray | None) -> dict:
    model = keras.models.load_model(model_dir / "model.keras")
    meta = json.loads((model_dir / "metadata.json").read_text())
    T, thr = meta["temperature"], meta["threshold"]

    t0 = time.perf_counter()
    for img in images[:50]:
        model(img[None], training=False)
    keras_ms = (time.perf_counter() - t0) / min(50, len(images)) * 1000
    ref = model.predict(images, verbose=0)[:, 0]
    ref_pred = sigmoid(ref / T) >= thr

    keras_size = (model_dir / "model.keras").stat().st_size
    report = {"n_images": int(len(images)), "keras": {"size_mb": keras_size / 1e6, "ms_per_image_batch1": keras_ms}}
    if labels is not None:
        report["keras"]["accuracy"] = float((ref_pred == labels).mean())
    for name, path in convert(model, model_dir / "tflite").items():
        logits, ms = _run_tflite(_interpreter(path), images)
        pred = sigmoid(logits / T) >= thr
        entry = {
            "path": str(path.name),
            "size_mb": path.stat().st_size / 1e6,
            "ms_per_image_batch1_1thread": ms,
            "label_agreement_with_keras": float((pred == ref_pred).mean()),
            "max_abs_prob_diff": float(np.max(np.abs(sigmoid(logits / T) - sigmoid(ref / T)))),
        }
        if labels is not None:
            entry["accuracy"] = float((pred == labels).mean())
        report[name] = entry
    (model_dir / "export_report.json").write_text(json.dumps(report, indent=2))
    return report


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default="artifacts/latest")
    ap.add_argument("--run-dir", help="use the test split of this run for the benchmark (recommended)")
    ap.add_argument("--n", type=int, default=300)
    a = ap.parse_args(argv)

    model_dir = Path(a.model_dir)
    meta = json.loads((model_dir / "metadata.json").read_text())
    labels = None
    if a.run_dir:
        import csv

        from .data import load_images

        rows = [r for r in csv.DictReader(open(Path(a.run_dir) / "splits.csv", encoding="utf-8")) if r["split"] == "test"]
        rows = rows[: a.n]
        images = load_images([r["path"] for r in rows], meta["img_size"])
        labels = np.array([int(r["label"]) for r in rows]).astype(bool)
    else:
        rng = np.random.default_rng(0)
        images = rng.uniform(0, 255, (a.n, meta["img_size"], meta["img_size"], 3)).astype(np.float32)
    print(json.dumps(benchmark(model_dir, images, labels), indent=2))


if __name__ == "__main__":
    main()
