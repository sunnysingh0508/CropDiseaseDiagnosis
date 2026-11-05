"""Application entry point for the CropCare AI API."""

import logging
from contextlib import asynccontextmanager
from collections.abc import AsyncGenerator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from config import settings
from routes.diagnosis import router as diagnosis_router
from services.model_service import load_model_bundle

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Load and validate the trained classifier before serving requests."""
    try:
        app.state.model_bundle = load_model_bundle()
    except Exception as exc:
        logger.exception("Failed to load the CropCare diagnosis model.")
        raise RuntimeError(
            "CropCare model initialization failed; diagnosis requests are disabled."
        ) from exc
    yield


app = FastAPI(
    title="CropCare AI",
    description="API for the CropCare AI crop disease diagnosis project.",
    version="0.1.0",
    lifespan=lifespan,
)

# Allow the local static frontend to call the API during development.
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

app.include_router(diagnosis_router, prefix="/api")


@app.get("/")
async def read_root() -> dict[str, str]:
    """Confirm that the backend is running."""
    return {"message": "CropCare AI Backend is Running"}
