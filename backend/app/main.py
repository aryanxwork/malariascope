"""MalariaScope inference API.

    uvicorn app.main:app --reload --port 8000        (from backend/)
    docs: http://localhost:8000/docs
"""

from __future__ import annotations

import logging
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

from fastapi import FastAPI, File, HTTPException, Query, Request, UploadFile  # noqa: E402
from fastapi.middleware.cors import CORSMiddleware  # noqa: E402
from fastapi.responses import FileResponse  # noqa: E402

from .inference import Engine, InvalidImage, decode_upload  # noqa: E402
from .schemas import BatchResponse, Health, Prediction  # noqa: E402
from .settings import Settings  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("malariascope.api")

DISCLAIMER = (
    "Research/education prototype. Not a medical device and not validated for clinical diagnosis. "
    "Trained on single segmented red-blood-cell crops from Giemsa-stained thin smears (NIH/NLM dataset)."
)


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or Settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.started = time.time()
        app.state.engine = None
        app.state.load_error = None
        try:
            app.state.engine = Engine(settings.model_dir)
        except Exception as e:  # keep serving /health so the failure is visible
            app.state.load_error = f"{e.__class__.__name__}: {e}"
            log.error("Model failed to load from %s: %s", settings.model_dir, app.state.load_error)
        yield

    app = FastAPI(
        title="MalariaScope API",
        version="1.0.0",
        description="Malaria parasite detection in thin blood-smear cell images. " + DISCLAIMER,
        lifespan=lifespan,
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
    )

    @app.middleware("http")
    async def timing(request: Request, call_next):
        t0 = time.perf_counter()
        response = await call_next(request)
        response.headers["X-Process-Time-ms"] = f"{(time.perf_counter() - t0) * 1000:.1f}"
        return response

    def engine() -> Engine:
        e = request_engine()
        if e is None:
            raise HTTPException(503, detail=f"Model not loaded: {app.state.load_error}. Train a model first.")
        return e

    def request_engine() -> Engine | None:
        return getattr(app.state, "engine", None)

    def read_upload(f: UploadFile) -> bytes:
        data = f.file.read(settings.max_upload_bytes + 1)
        if len(data) > settings.max_upload_bytes:
            raise HTTPException(413, detail=f"{f.filename}: file exceeds {settings.max_upload_mb:g} MB")
        return data

    @app.get("/api/health", response_model=Health, tags=["meta"])
    def health():
        e = request_engine()
        return Health(
            status="ok" if e else "degraded",
            model_loaded=e is not None,
            model=e.metadata.get("model") if e else None,
            run=e.metadata.get("run") if e else None,
            uptime_s=round(time.time() - app.state.started, 1),
            detail=None if e else app.state.load_error,
        )

    @app.get("/api/model", tags=["meta"])
    def model_info():
        return {**engine().public_info(), "disclaimer": DISCLAIMER}

    @app.get("/api/figures/{name}", tags=["meta"], include_in_schema=False)
    def figure(name: str):
        if "/" in name or ".." in name or not name.endswith(".png"):
            raise HTTPException(404)
        path = Path(engine().model_dir) / "figures" / name
        if not path.exists():
            raise HTTPException(404)
        return FileResponse(path, media_type="image/png")

    @app.post("/api/predict", response_model=Prediction, tags=["inference"])
    def predict(file: UploadFile = File(...), gradcam: bool = Query(True, description="Return a Grad-CAM overlay")):
        e = engine()
        data = read_upload(file)
        try:
            img = decode_upload(file.filename or "upload", data, e.img_size)
        except InvalidImage as err:
            raise HTTPException(415, detail=str(err)) from err
        return e.predict([img], gradcam)[0]

    @app.post("/api/predict/batch", response_model=BatchResponse, tags=["inference"])
    def predict_batch(files: list[UploadFile] = File(...), gradcam: bool = Query(True)):
        e = engine()
        if len(files) > settings.max_batch:
            raise HTTPException(413, detail=f"At most {settings.max_batch} images per batch")
        t0 = time.perf_counter()
        decoded, errors = [], []
        for f in files:
            name = f.filename or "upload"
            try:
                decoded.append(decode_upload(name, read_upload(f), e.img_size))
            except InvalidImage as err:
                errors.append({"filename": name, "error": str(err)})
            except HTTPException as err:
                errors.append({"filename": name, "error": str(err.detail)})
        results = []
        for i in range(0, len(decoded), 32):
            results.extend(e.predict(decoded[i : i + 32], gradcam))
        n_par = sum(r["label"] == "Parasitized" for r in results)
        # Images that failed the input checks still get a score, but it is not meaningful -> keep them out of
        # the slide-level estimate.
        reliable = [r for r in results if not r["warnings"]]
        n_par_reliable = sum(r["label"] == "Parasitized" for r in reliable)
        summary = {
            "total": len(files),
            "analyzed": len(results),
            "failed": len(errors),
            "parasitized": n_par,
            "uninfected": len(results) - n_par,
            "needs_review": sum(r["needs_review"] for r in results),
            "flagged_input": len(results) - len(reliable),
            "estimated_parasitemia_pct": round(100 * n_par_reliable / len(reliable), 2) if reliable else 0.0,
            "latency_ms": round((time.perf_counter() - t0) * 1000, 1),
        }
        return {"results": results, "errors": errors, "summary": summary}

    return app


app = create_app()
