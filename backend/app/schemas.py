from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class Prediction(BaseModel):
    filename: str
    label: Literal["Parasitized", "Uninfected"]
    probability_parasitized: float = Field(..., ge=0, le=1, description="Temperature-calibrated probability")
    confidence: float = Field(..., ge=0, le=1, description="Probability of the predicted label")
    threshold: float = Field(..., description="Operating threshold chosen on validation for the target sensitivity")
    needs_review: bool = Field(..., description="Probability is close to the threshold -> send to a human")
    warnings: list[str] = Field(default_factory=list, description="Input-quality / out-of-distribution warnings")
    gradcam_png: str | None = Field(None, description="data:image/png;base64 Grad-CAM overlay")
    latency_ms: float


class PredictionError(BaseModel):
    filename: str
    error: str


class BatchSummary(BaseModel):
    total: int
    analyzed: int
    failed: int
    parasitized: int
    uninfected: int
    needs_review: int
    flagged_input: int = Field(..., description="Images that failed input-quality checks (excluded from parasitemia)")
    estimated_parasitemia_pct: float = Field(..., description="Parasitized / analyzed cells (excluding flagged inputs), in %")
    latency_ms: float


class BatchResponse(BaseModel):
    results: list[Prediction]
    errors: list[PredictionError]
    summary: BatchSummary


class Health(BaseModel):
    status: Literal["ok", "degraded"]
    model_loaded: bool
    model: str | None = None
    run: str | None = None
    uptime_s: float
    detail: str | None = None
