from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def _list(v: str) -> list[str]:
    return [x.strip() for x in v.split(",") if x.strip()]


@dataclass(frozen=True)
class Settings:
    model_dir: Path = field(default_factory=lambda: Path(os.getenv("MODEL_DIR", REPO_ROOT / "artifacts" / "latest")))
    max_upload_mb: float = float(os.getenv("MAX_UPLOAD_MB", "5"))
    max_batch: int = int(os.getenv("MAX_BATCH", "64"))
    cors_origins: list[str] = field(
        default_factory=lambda: _list(os.getenv("CORS_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173"))
    )

    @property
    def max_upload_bytes(self) -> int:
        return int(self.max_upload_mb * 1024 * 1024)
