"""Health and crop image diagnosis endpoints."""

from io import BytesIO

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from PIL import Image, UnidentifiedImageError

from config import settings

router = APIRouter()

SUPPORTED_CROPS = {
    "apple",
    "bell pepper",
    "corn",
    "grape",
    "potato",
    "tomato",
}


@router.get("/health")
async def read_health() -> dict[str, str]:
    """Report that the diagnosis API is available."""
    return {"status": "ok", "message": "CropCare AI diagnosis API is ready"}


@router.post("/diagnose")
async def diagnose_crop(
    crop_name: str = Form(...),
    image: UploadFile = File(...),
) -> dict[str, str]:
    """Validate an uploaded crop image without making a disease prediction."""
    normalized_crop_name = crop_name.strip().casefold()
    if normalized_crop_name not in SUPPORTED_CROPS:
        raise HTTPException(
            status_code=400,
            detail=(
                "Select a supported crop: Tomato, Potato, Apple, Grape, "
                "Corn, or Bell Pepper."
            ),
        )

    if not image.filename:
        raise HTTPException(status_code=400, detail="Choose an image file to continue.")

    try:
        # Read at most one byte past the limit so oversized files are rejected safely.
        image_data = await image.read(settings.max_upload_size_bytes + 1)
        if not image_data:
            raise HTTPException(status_code=400, detail="The uploaded image is empty.")
        if len(image_data) > settings.max_upload_size_bytes:
            max_size_mb = settings.max_upload_size_bytes // (1024 * 1024)
            raise HTTPException(
                status_code=413,
                detail=f"The image exceeds the maximum size of {max_size_mb} MB.",
            )

        try:
            with Image.open(BytesIO(image_data)) as uploaded_image:
                image_format = (uploaded_image.format or "").upper()
                if image_format not in settings.allowed_image_formats:
                    allowed_formats = ", ".join(sorted(settings.allowed_image_formats))
                    raise HTTPException(
                        status_code=415,
                        detail=(
                            "Unsupported image format. "
                            f"Allowed formats: {allowed_formats}."
                        ),
                    )

                uploaded_image.verify()
        except HTTPException:
            raise
        except (
            UnidentifiedImageError,
            OSError,
            Image.DecompressionBombError,
        ) as exc:
            raise HTTPException(
                status_code=400,
                detail="The uploaded file is not a valid, readable image.",
            ) from exc
    finally:
        await image.close()

    return {
        "status": "pending",
        "crop": crop_name.strip(),
        "message": "AI model integration is pending. No disease prediction was made.",
    }
