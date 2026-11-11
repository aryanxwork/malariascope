import numpy as np

from malariascope.calibration import fit_temperature, reliability, sigmoid
from malariascope.gradcam import gradcam, make_gradcam_model, overlay
from malariascope.metrics import binary_metrics, cluster_bootstrap_ci, threshold_for_sensitivity
from malariascope.models import build_baseline_cnn


def test_baseline_outputs_logits_and_gradcam_works():
    model = build_baseline_cnn(64)
    x = np.random.default_rng(0).uniform(0, 255, (2, 64, 64, 3)).astype("float32")
    assert model(x).shape == (2, 1)
    heat, logits = gradcam(make_gradcam_model(model), x)
    assert heat.shape[0] == 2 and heat.min() >= 0 and heat.max() <= 1 + 1e-6
    ov = overlay(x[0], heat[0], strength=0.9)
    assert ov.shape == (64, 64, 3) and ov.dtype == np.uint8


def test_binary_metrics_known_values():
    y = np.array([1, 1, 0, 0])
    p = np.array([0.9, 0.2, 0.8, 0.1])
    m = binary_metrics(y, p, 0.5)
    assert m["confusion"] == {"tp": 1, "fp": 1, "tn": 1, "fn": 1}
    assert m["accuracy"] == 0.5 and m["sensitivity"] == 0.5 and m["specificity"] == 0.5


def test_threshold_meets_target_sensitivity():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 2000)
    p = np.clip(y * 0.6 + rng.normal(0.2, 0.2, 2000), 0, 1)
    t = threshold_for_sensitivity(y, p, 0.95)
    assert binary_metrics(y, p, t)["sensitivity"] >= 0.95


def test_temperature_scaling_fixes_overconfidence():
    rng = np.random.default_rng(0)
    true_logits = rng.normal(0, 2, 5000)
    y = (rng.uniform(size=5000) < sigmoid(true_logits)).astype(float)
    overconfident = true_logits * 3
    T = fit_temperature(overconfident, y)
    assert 2.4 < T < 3.6
    assert reliability(y, sigmoid(overconfident / T))["ece"] < reliability(y, sigmoid(overconfident))["ece"]


def test_cluster_bootstrap_returns_ordered_intervals():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 600)
    p = np.clip(y * 0.5 + rng.uniform(0, 0.5, 600), 0, 1)
    groups = np.repeat(np.arange(30), 20)
    ci = cluster_bootstrap_ci(y, p, groups, 0.5, n_iter=100)
    for lo, hi in ci.values():
        assert lo <= hi


def test_threshold_is_not_brittle_on_separable_data():
    y = np.array([0] * 50 + [1] * 50)
    p = np.concatenate([np.linspace(0.0, 0.2, 50), np.linspace(0.8, 1.0, 50)])
    t = threshold_for_sensitivity(y, p, 0.98)
    assert 0.2 < t < 0.8  # inside the gap, not glued to the edge of the positive scores
    m = binary_metrics(y, p, t)
    assert m["sensitivity"] == 1.0 and m["specificity"] == 1.0
