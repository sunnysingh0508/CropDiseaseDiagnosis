"""Health and crop image diagnosis endpoints."""

import logging
import warnings
from io import BytesIO

from fastapi import APIRouter, File, HTTPException, Request, UploadFile
from PIL import Image, UnidentifiedImageError
from starlette.concurrency import run_in_threadpool

from config import settings
from services.model_service import ModelBundle, predict_image

router = APIRouter()
logger = logging.getLogger(__name__)
SUPPORTED_IMAGE_FORMATS = frozenset({"JPEG", "PNG"})
MAX_IMAGE_SIZE_BYTES = 10 * 1024 * 1024


@router.get("/health")
async def read_health() -> dict[str, str]:
    """Report that the diagnosis API is available."""
    return {"status": "ok", "message": "CropCare AI diagnosis API is ready"}


@router.post("/diagnose")
async def diagnose_crop(
    request: Request,
    image: UploadFile = File(...),
) -> dict[str, object]:
    """Run the trained classifier on a validated crop image."""
    max_upload_size_bytes = min(
        settings.max_upload_size_bytes,
        MAX_IMAGE_SIZE_BYTES,
    )
    try:
        if not image.filename:
            raise HTTPException(status_code=400, detail="Choose an image file to continue.")

        image_data = await image.read(max_upload_size_bytes + 1)
        if not image_data:
            raise HTTPException(status_code=400, detail="The uploaded image is empty.")
        if len(image_data) > max_upload_size_bytes:
            max_size_mb = max_upload_size_bytes // (1024 * 1024)
            raise HTTPException(
                status_code=413,
                detail=f"The image exceeds the maximum size of {max_size_mb} MB.",
            )
    finally:
        await image.close()

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(BytesIO(image_data)) as uploaded_image:
                image_format = (uploaded_image.format or "").upper()
                allowed_formats = SUPPORTED_IMAGE_FORMATS & settings.allowed_image_formats
                if image_format not in allowed_formats:
                    allowed_format_names = ", ".join(sorted(allowed_formats))
                    raise HTTPException(
                        status_code=415,
                        detail=(
                            "Unsupported image format. "
                            f"Allowed formats: {allowed_format_names}."
                        ),
                    )
                uploaded_image.verify()
            with Image.open(BytesIO(image_data)) as uploaded_image:
                uploaded_image.load()
                rgb_image = uploaded_image.convert("RGB")
    except HTTPException:
        raise
    except (
        Image.DecompressionBombError,
        Image.DecompressionBombWarning,
        OSError,
        UnidentifiedImageError,
        ValueError,
    ) as exc:
        raise HTTPException(
            status_code=400,
            detail="The uploaded file is not a valid, readable image.",
        ) from exc

    model_bundle: ModelBundle | None = getattr(request.app.state, "model_bundle", None)
    if model_bundle is None:
        raise HTTPException(
            status_code=503,
            detail="The diagnosis model is not available.",
        )

    try:
        return await run_in_threadpool(predict_image, model_bundle, rgb_image)
    except Exception as exc:
        logger.exception("Crop disease prediction failed.")
        raise HTTPException(
            status_code=500,
            detail="The diagnosis model failed to process the image.",
        ) from exc
