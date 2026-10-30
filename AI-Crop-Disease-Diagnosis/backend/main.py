"""Application entry point for the CropCare AI API."""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from config import settings
from routes.diagnosis import router as diagnosis_router

app = FastAPI(
    title="CropCare AI",
    description="API for the CropCare AI crop disease diagnosis project.",
    version="0.1.0",
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
