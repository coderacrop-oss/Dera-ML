from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    data_dir: Path
    api_key: str
    calendar_token: str | None = None
    models_dir: Path = Path("models")
    public_url: str = "http://localhost:8000"

    @classmethod
    def from_env(cls) -> "Settings":
        key = os.environ.get("CRM_API_KEY")
        if not key:
            raise RuntimeError("CRM_API_KEY is not set; refusing to start an unauthenticated API")
        return cls(data_dir=Path(os.environ.get("CRM_API_DATA_DIR", "data/processed")),
                   api_key=key, calendar_token=os.environ.get("CRM_API_CALENDAR_TOKEN") or None,
                   models_dir=Path(os.environ.get("CRM_API_MODELS_DIR", "models")),
                   public_url=os.environ.get("CRM_API_PUBLIC_URL", "http://localhost:8000"))
