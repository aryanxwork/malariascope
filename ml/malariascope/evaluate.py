"""Full evaluation of a trained model: calibration -> threshold selection (on validation) -> test metrics
with patient-clustered bootstrap CIs -> figures -> Grad-CAM error gallery -> deployable metadata.

CLI (re-evaluate a finished run):  malariascope-evaluate --run-dir artifacts/runs/<run>
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import time
from pathlib import Path

import numpy as np

from . import CLASS_NAMES, __version__
from . import calibration as cal
from . import metrics as M
from . import plots
from .config import TrainConfig
from .data import Sample, load_images, make_dataset, summarize_splits
from .gradcam import gradcam, make_gradcam_model, overlay

DEPLOY_FILES = ("model.keras", "metadata.json", "metrics.json")


def _json_default(o):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(type(o))


def write_json(obj, path: Path):
    path.write_text(json.dumps(obj, indent=2, default=_json_default), encoding="utf-8")


def predict_logits(model, samples: list[Sample], cfg: TrainConfig) -> np.ndarray:
    ds = make_dataset(samples, cfg.img_size, cfg.batch_size, training=False, cache=False)
    return model.predict(ds, verbose=0)[:, 0].astype(np.float64)


def _pick_gallery(y, pred, p, k: int, seed: int) -> list[tuple[int, str]]:
    """Pick a mix of correct and incorrect test cases (errors first — that's where the insight is)."""
    rng = np.random.default_rng(seed)
    buckets = {
        "FN": np.where((y == 1) & (pred == 0))[0],
        "FP": np.where((y == 0) & (pred == 1))[0],
        "TP": np.where((y == 1) & (pred == 1))[0],
        "TN": np.where((y == 0) & (pred == 0))[0],
    }
    per = max(1, k // 4)
    chosen: list[tuple[int, str]] = []
    for name in ("FN", "FP", "TP", "TN"):
        idx = buckets[name]
        if len(idx):
            for i in rng.choice(idx, size=min(per, len(idx)), replace=False):
                chosen.append((int(i), name))
    for name in ("TP", "TN"):  # top up if there were few errors
        extra = [i for i in buckets[name] if (int(i), name) not in chosen]
        while len(chosen) < k and extra:
            chosen.append((int(extra.pop()), name))
    return chosen[:k]


def run_evaluation(
    model,
    splits: dict[str, list[Sample]],
    cfg: TrainConfig,
    run_dir: Path,
    history: dict | None = None,
    finetune_start: int | None = None,
    params: dict | None = None,
    train_seconds: float | None = None,
) -> dict:
    run_dir = Path(run_dir)
    fig_dir = run_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    val, test = splits["val"], splits["test"]
    y_val = np.array([s.label for s in val])
    y_test = np.array([s.label for s in test])
    g_test = np.array([s.patient for s in test])

    t0 = time.perf_counter()
    z_val = predict_logits(model, val, cfg)
    z_test = predict_logits(model, test, cfg)
    infer_ms = (time.perf_counter() - t0) / (len(val) + len(test)) * 1000

    # 1) Calibrate on validation only.
    temperature = cal.fit_temperature(z_val, y_val)
    p_val = cal.sigmoid(z_val / temperature)
    p_test_raw = cal.sigmoid(z_test)
    p_test = cal.sigmoid(z_test / temperature)

    # 2) Choose the operating threshold on validation only (never peek at test).
    thr = M.threshold_for_sensitivity(y_val, p_val, cfg.target_sensitivity)
    val_at_thr = M.binary_metrics(y_val, p_val, thr)

    # 3) Test metrics.
    test_at_thr = M.binary_metrics(y_test, p_test, thr)
    test_at_half = M.binary_metrics(y_test, p_test, 0.5)
    ranking = M.ranking_metrics(y_test, p_test)
    ci = M.cluster_bootstrap_ci(y_test, p_test, g_test, thr, n_iter=cfg.bootstrap_iterations, seed=cfg.seed)
    rel_before = cal.reliability(y_test, p_test_raw)
    rel_after = cal.reliability(y_test, p_test)
    crv = M.curves(y_test, p_test)
    review = np.abs(p_test - thr) < cfg.review_margin
    not_review = ~review
    auto_acc = float(((p_test[not_review] >= thr).astype(int) == y_test[not_review]).mean()) if not_review.any() else None

    metrics = {
        "schema_version": 1,
        "model": cfg.model,
        "split_mode": cfg.split_mode,
        "params": params or {},
        "img_size": cfg.img_size,
        "train_seconds": train_seconds,
        "eval_ms_per_image_batched": infer_ms,
        "splits": summarize_splits(splits),
        "calibration": {
            "temperature": temperature,
            "ece_before": rel_before["ece"],
            "ece_after": rel_after["ece"],
            "brier_before": cal.brier(y_test, p_test_raw),
            "brier_after": cal.brier(y_test, p_test),
            "reliability_before": rel_before["bins"],
            "reliability_after": rel_after["bins"],
        },
        "threshold": {
            "policy": "max specificity subject to validation sensitivity >= target",
            "target_sensitivity": cfg.target_sensitivity,
            "value": thr,
            "youden_value": M.youden_threshold(y_val, p_val),
            "validation": val_at_thr,
            "review_margin": cfg.review_margin,
        },
        "validation": {**M.ranking_metrics(y_val, p_val), "n": len(y_val)},
        "test": {
            **ranking,
            "n": len(y_test),
            "at_threshold": test_at_thr,
            "at_0_5": test_at_half,
            "ci95_patient_bootstrap": ci,
            "review_rate": float(review.mean()),
            "accuracy_excluding_review": auto_acc,
            "roc_curve": crv["roc"],
            "pr_curve": crv["pr"],
        },
        "history": history or {},
        "finetune_start_epoch": finetune_start,
    }

    # 4) Figures.
    plots.confusion_matrix_plot(test_at_thr["confusion"], fig_dir / "confusion_matrix.png")
    op = (1 - test_at_thr["specificity"], test_at_thr["sensitivity"])
    plots.roc_plot(crv["roc"], ranking["roc_auc"], op, fig_dir / "roc_curve.png")
    plots.pr_plot(crv["pr"], ranking["pr_auc"], fig_dir / "pr_curve.png")
    plots.reliability_plot(rel_before, rel_after, fig_dir / "reliability.png")
    if history:
        plots.history_plot(history, fig_dir / "training_curves.png", finetune_start)

    # 5) Grad-CAM gallery of errors and successes.
    pred = (p_test >= thr).astype(int)
    picks = _pick_gallery(y_test, pred, p_test, cfg.gradcam_examples, cfg.seed)
    if picks:
        imgs = load_images([test[i].path for i, _ in picks], cfg.img_size)
        heat, _ = gradcam(make_gradcam_model(model), imgs)
        rows = [
            {
                "image": imgs[j].astype(np.uint8),
                "overlay": overlay(imgs[j], heat[j], strength=max(float(p_test[i]), 0.35)),
                "title": f"{kind} · p={p_test[i]:.2f}",
            }
            for j, (i, kind) in enumerate(picks)
        ]
        plots.gradcam_gallery(rows, fig_dir / "gradcam_gallery.png")

    # 6) Per-image predictions for error analysis.
    with open(run_dir / "test_predictions.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["path", "patient", "label", "prob_parasitized", "pred", "needs_review"])
        for s, p_, r_ in zip(test, p_test, review):
            w.writerow([s.path, s.patient, s.label, f"{p_:.6f}", int(p_ >= thr), int(r_)])

    metadata = {
        "name": "MalariaScope",
        "package_version": __version__,
        "model": cfg.model,
        "run": run_dir.name,
        "trained_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "img_size": cfg.img_size,
        "class_names": list(CLASS_NAMES),
        "positive_class": CLASS_NAMES[1],
        "temperature": temperature,
        "threshold": thr,
        "review_margin": cfg.review_margin,
        "target_sensitivity": cfg.target_sensitivity,
        "split_mode": cfg.split_mode,
        "params": params or {},
        "dataset": "NIH/NLM Malaria thin-smear cell images (27,558 cells, Rajaraman et al. 2018)",
    }
    write_json(metrics, run_dir / "metrics.json")
    write_json(metadata, run_dir / "metadata.json")
    return metrics


def promote(run_dir: Path, latest_dir: Path) -> None:
    """Copy the deployable files of a run to artifacts/latest (what the API serves)."""
    latest_dir.mkdir(parents=True, exist_ok=True)
    for f in DEPLOY_FILES:
        shutil.copy2(run_dir / f, latest_dir / f)
    figs = run_dir / "figures"
    if figs.exists():
        shutil.copytree(figs, latest_dir / "figures", dirs_exist_ok=True)


def print_summary(m: dict) -> None:
    t = m["test"]
    a = t["at_threshold"]
    ci = t.get("ci95_patient_bootstrap", {})

    def fmt(k, v):
        lo_hi = ci.get(k)
        return f"{v:.4f}" + (f"  [95% CI {lo_hi[0]:.4f}–{lo_hi[1]:.4f}]" if lo_hi else "")

    print("\n================ TEST RESULTS (held-out patients) ================")
    print(f"split_mode          : {m['split_mode']}   n={t['n']}")
    print(f"threshold           : {m['threshold']['value']:.4f} (target sensitivity {m['threshold']['target_sensitivity']})")
    print(f"accuracy            : {fmt('accuracy', a['accuracy'])}")
    print(f"sensitivity         : {fmt('sensitivity', a['sensitivity'])}")
    print(f"specificity         : {fmt('specificity', a['specificity'])}")
    print(f"F1                  : {fmt('f1', a['f1'])}")
    print(f"ROC-AUC             : {fmt('roc_auc', t['roc_auc'])}")
    print(f"PR-AUC              : {t['pr_auc']:.4f}")
    print(f"MCC                 : {a['mcc']:.4f}")
    print(f"ECE raw -> scaled   : {m['calibration']['ece_before']:.4f} -> {m['calibration']['ece_after']:.4f} (T={m['calibration']['temperature']:.3f})")
    print(f"accuracy @0.5       : {t['at_0_5']['accuracy']:.4f}")
    print("==================================================================\n")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Re-evaluate a finished training run.")
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--promote", action="store_true", help="copy to artifacts/latest afterwards")
    args = ap.parse_args(argv)

    from tensorflow import keras

    run_dir = Path(args.run_dir)
    cfg = TrainConfig(**json.loads((run_dir / "config.json").read_text())).validate()
    splits: dict[str, list[Sample]] = {"train": [], "val": [], "test": []}
    with open(run_dir / "splits.csv", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            splits[row["split"]].append(Sample(row["path"], int(row["label"]), row["patient"]))
    model = keras.models.load_model(run_dir / "model.keras")
    old = json.loads((run_dir / "metrics.json").read_text()) if (run_dir / "metrics.json").exists() else {}
    m = run_evaluation(model, splits, cfg, run_dir, history=old.get("history"),
                       finetune_start=old.get("finetune_start_epoch"), params=old.get("params"),
                       train_seconds=old.get("train_seconds"))
    print_summary(m)
    if args.promote:
        promote(run_dir, Path(cfg.output_dir) / "latest")


if __name__ == "__main__":
    main()
