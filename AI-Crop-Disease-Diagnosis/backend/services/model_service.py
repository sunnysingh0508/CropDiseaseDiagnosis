"""Model loading, preprocessing, and inference for CropCare AI."""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import tensorflow as tf
from PIL import Image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

PROJECT_DIRECTORY = Path(__file__).resolve().parents[2]
MODEL_PATH = PROJECT_DIRECTORY / "model" / "training" / "best_model.keras"
CLASS_MAPPING_PATH = (
    PROJECT_DIRECTORY / "dataset" / "processed" / "class_mapping.json"
)
IMAGE_SIZE = (224, 224)
EXPECTED_CLASS_COUNT = 15
PROBABILITY_TOLERANCE = 1e-4


@dataclass(frozen=True)
class ModelBundle:
    """The loaded classifier and its output-index-ordered class names."""

    model: tf.keras.Model
    class_names: tuple[str, ...]


def _load_class_names() -> tuple[str, ...]:
    """Load and validate the training class mapping in output-index order."""
    with CLASS_MAPPING_PATH.open("r", encoding="utf-8") as mapping_file:
        mapping_data: Any = json.load(mapping_file)

    if not isinstance(mapping_data, dict):
        raise ValueError("class_mapping.json must contain a JSON object.")

    class_to_index = mapping_data.get("class_to_index")
    if not isinstance(class_to_index, dict) or len(class_to_index) != EXPECTED_CLASS_COUNT:
        raise ValueError(
            f"class_to_index must define exactly {EXPECTED_CLASS_COUNT} classes."
        )

    if any(
        not isinstance(class_name, str)
        or not class_name
        or type(class_index) is not int
        for class_name, class_index in class_to_index.items()
    ):
        raise ValueError("class_to_index must map non-empty class names to integer indices.")

    indices = set(class_to_index.values())
    if indices != set(range(EXPECTED_CLASS_COUNT)):
        raise ValueError("class_to_index indices must be unique and contiguous from 0 to 14.")

    class_names_by_index = [""] * EXPECTED_CLASS_COUNT
    for class_name, class_index in class_to_index.items():
        class_names_by_index[class_index] = class_name
    return tuple(class_names_by_index)


def load_model_bundle() -> ModelBundle:
    """Load the Keras model and refuse startup if its class contract is invalid."""
    class_names = _load_class_names()
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)

    input_shape = model.input_shape
    if (
        isinstance(input_shape, list)
        or len(input_shape) != 4
        or tuple(input_shape[1:]) != (*IMAGE_SIZE, 3)
    ):
        raise ValueError(
            f"The model must accept 224x224 RGB images; found {input_shape!r}."
        )

    output_shape = model.output_shape
    if (
        isinstance(output_shape, list)
        or len(output_shape) != 2
        or output_shape[-1] != len(class_names)
    ):
        raise ValueError(
            "The model output class count must match all 15 class_to_index entries; "
            f"found output shape {output_shape!r}."
        )

    return ModelBundle(model=model, class_names=class_names)


def predict_image(
    model_bundle: ModelBundle,
    image: Image.Image,
) -> dict[str, object]:
    """Apply training-identical MobileNetV2 preprocessing and return real scores."""
    image_array = np.asarray(image.convert("RGB"), dtype=np.float32)
    image_tensor = tf.image.resize(image_array, IMAGE_SIZE)
    model_input = tf.expand_dims(preprocess_input(image_tensor), axis=0)
    raw_scores = model_bundle.model(model_input, training=False)
    scores = np.asarray(raw_scores, dtype=np.float32)

    expected_shape = (1, len(model_bundle.class_names))
    if scores.shape != expected_shape:
        raise RuntimeError(
            f"Expected prediction scores with shape {expected_shape}, got {scores.shape}."
        )

    probabilities = scores[0]
    if (
        not np.isfinite(probabilities).all()
        or np.any(probabilities < 0)
        or np.any(probabilities > 1)
        or not np.isclose(
            probabilities.sum(),
            1.0,
            rtol=0.0,
            atol=PROBABILITY_TOLERANCE,
        )
    ):
        raise RuntimeError("The model returned invalid class probabilities.")

    top_indices = np.argsort(probabilities)[-3:][::-1]
    predicted_class = model_bundle.class_names[int(top_indices[0])]
    is_healthy = predicted_class.casefold().endswith("healthy")

    return {
        "predicted_class": predicted_class,
        "confidence_percentage": round(
            float(probabilities[int(top_indices[0])]) * 100,
            2,
        ),
        "top_3_predictions": [
            {
                "class_name": model_bundle.class_names[int(index)],
                "confidence_percentage": round(float(probabilities[index]) * 100, 2),
            }
            for index in top_indices
        ],
        "status": "healthy" if is_healthy else "diseased",
    }
