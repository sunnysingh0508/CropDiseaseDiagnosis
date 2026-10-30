"""Environment-based settings for the CropCare AI backend."""

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

BACKEND_DIRECTORY = Path(__file__).resolve().parent
load_dotenv(BACKEND_DIRECTORY / ".env")


def _read_positive_integer(name: str, default: int) -> int:
    """Read a positive integer setting and fail clearly if it is invalid."""
    value = os.getenv(name, str(default))
    try:
        parsed_value = int(value)
    except ValueError as exc:
        raise ValueError(f"{name} must be a positive integer.") from exc

    if parsed_value <= 0:
        raise ValueError(f"{name} must be a positive integer.")

    return parsed_value


@dataclass(frozen=True)
class Settings:
    """Application settings loaded from the backend .env file."""

    cors_origins: tuple[str, ...]
    max_upload_size_bytes: int
    allowed_image_formats: frozenset[str]


def _load_settings() -> Settings:
    cors_origins = tuple(
        origin.strip()
        for origin in os.getenv(
            "CORS_ORIGINS",
            "http://127.0.0.1:5500,http://localhost:5500",
        ).split(",")
        if origin.strip()
    )
    if not cors_origins:
        raise ValueError("CORS_ORIGINS must contain at least one frontend origin.")

    allowed_formats = frozenset(
        image_format.strip().upper()
        for image_format in os.getenv(
            "ALLOWED_IMAGE_FORMATS",
            "JPEG,PNG,WEBP",
        ).split(",")
        if image_format.strip()
    )
    if not allowed_formats:
        raise ValueError("ALLOWED_IMAGE_FORMATS must contain at least one format.")

    max_upload_size_mb = _read_positive_integer("MAX_UPLOAD_SIZE_MB", 10)

    return Settings(
        cors_origins=cors_origins,
        max_upload_size_bytes=max_upload_size_mb * 1024 * 1024,
        allowed_image_formats=allowed_formats,
    )


settings = _load_settings()
