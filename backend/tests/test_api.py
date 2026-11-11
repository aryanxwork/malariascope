import io
import json

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image, ImageDraw

from app.main import create_app
from app.settings import Settings
from malariascope.models import build_baseline_cnn


def cell_png(size=120, dark_border=True) -> bytes:
    img = Image.new("RGB", (size, size), (0, 0, 0) if dark_border else (230, 230, 230))
    ImageDraw.Draw(img).ellipse([10, 10, size - 10, size - 10], fill=(210, 160, 160))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


@pytest.fixture(scope="session")
def model_dir(tmp_path_factory):
    d = tmp_path_factory.mktemp("model")
    build_baseline_cnn(32).save(d / "model.keras")
    (d / "metadata.json").write_text(
        json.dumps({"model": "baseline_cnn", "run": "test-run", "img_size": 32, "temperature": 1.5,
                    "threshold": 0.4, "review_margin": 0.15})
    )
    (d / "metrics.json").write_text(json.dumps({"test": {"roc_auc": 0.5}}))
    return d


@pytest.fixture(scope="session")
def client(model_dir):
    app = create_app(Settings(model_dir=model_dir, max_upload_mb=0.5, max_batch=4, cors_origins=["*"]))
    with TestClient(app) as c:
        yield c


def test_health(client):
    r = client.get("/api/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok" and body["model_loaded"] and body["run"] == "test-run"
    assert "X-Process-Time-ms" in r.headers


def test_model_info(client):
    body = client.get("/api/model").json()
    assert body["metadata"]["img_size"] == 32 and "disclaimer" in body


def test_predict_single(client):
    r = client.post("/api/predict", files={"file": ("cell.png", cell_png(), "image/png")})
    assert r.status_code == 200, r.text
    p = r.json()
    assert p["label"] in {"Parasitized", "Uninfected"}
    assert 0 <= p["probability_parasitized"] <= 1
    assert p["confidence"] >= 0.5
    assert (p["label"] == "Parasitized") == (p["probability_parasitized"] >= 0.4)
    assert p["gradcam_png"].startswith("data:image/png;base64,")
    assert p["warnings"] == []


def test_predict_without_gradcam(client):
    r = client.post("/api/predict?gradcam=false", files={"file": ("cell.png", cell_png(), "image/png")})
    assert r.status_code == 200 and r.json()["gradcam_png"] is None


def test_gradcam_and_plain_paths_agree(client):
    a = client.post("/api/predict", files={"file": ("c.png", cell_png(), "image/png")}).json()
    b = client.post("/api/predict?gradcam=false", files={"file": ("c.png", cell_png(), "image/png")}).json()
    assert abs(a["probability_parasitized"] - b["probability_parasitized"]) < 1e-4


def test_ood_warning_for_non_cell_image(client):
    r = client.post("/api/predict", files={"file": ("photo.png", cell_png(dark_border=False), "image/png")})
    assert r.status_code == 200 and r.json()["warnings"]


def test_rejects_non_image(client):
    r = client.post("/api/predict", files={"file": ("notes.txt", b"hello world", "text/plain")})
    assert r.status_code == 415


def test_rejects_oversized_upload(client):
    noise = np.random.default_rng(0).integers(0, 255, (700, 700, 3), dtype=np.uint8)
    buf = io.BytesIO()
    Image.fromarray(noise).save(buf, format="PNG")
    r = client.post("/api/predict", files={"file": ("big.png", buf.getvalue(), "image/png")})
    assert r.status_code == 413


def test_batch_mixed_valid_and_invalid(client):
    files = [
        ("files", ("a.png", cell_png(), "image/png")),
        ("files", ("b.png", cell_png(100), "image/png")),
        ("files", ("bad.txt", b"nope", "text/plain")),
    ]
    r = client.post("/api/predict/batch", files=files)
    assert r.status_code == 200, r.text
    body = r.json()
    s = body["summary"]
    assert s["total"] == 3 and s["analyzed"] == 2 and s["failed"] == 1
    assert s["parasitized"] + s["uninfected"] == 2
    assert body["errors"][0]["filename"] == "bad.txt"


def test_flagged_inputs_excluded_from_parasitemia(client):
    files = [
        ("files", ("cell.png", cell_png(), "image/png")),
        ("files", ("photo.png", cell_png(dark_border=False), "image/png")),
    ]
    s = client.post("/api/predict/batch", files=files).json()["summary"]
    assert s["flagged_input"] == 1
    assert s["estimated_parasitemia_pct"] in (0.0, 100.0)  # computed from the single reliable cell only


def test_batch_limit(client):
    files = [("files", (f"{i}.png", cell_png(), "image/png")) for i in range(5)]
    assert client.post("/api/predict/batch", files=files).status_code == 413


def test_missing_model_is_503_not_crash(tmp_path):
    app = create_app(Settings(model_dir=tmp_path / "nope"))
    with TestClient(app) as c:
        h = c.get("/api/health").json()
        assert h["status"] == "degraded" and not h["model_loaded"]
        r = c.post("/api/predict", files={"file": ("c.png", cell_png(), "image/png")})
        assert r.status_code == 503
