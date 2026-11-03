"""Prepare and optionally train CropCare AI's MobileNetV2 classifier.

Running this script without --train only checks configuration and manifests.
The model is never fitted unless the user explicitly passes --train.
"""

import argparse
import csv
import json
import math
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import tensorflow as tf
from tensorflow.keras import Model, layers
from tensorflow.keras.applications.mobilenet_v2 import (
    MobileNetV2,
    preprocess_input,
)
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    ReduceLROnPlateau,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = PROJECT_ROOT / "dataset" / "processed"
CONFIG_PATH = PROJECT_ROOT / "scripts" / "training_config.json"
TRAINING_OUTPUT_DIR = PROJECT_ROOT / "model"
CHECKPOINT_DIR = TRAINING_OUTPUT_DIR / "checkpoints"
HISTORY_DIR = TRAINING_OUTPUT_DIR / "history"
EVALUATION_DIR = TRAINING_OUTPUT_DIR / "evaluation"
EXPECTED_MANIFEST_COLUMNS = {
    "image_path",
    "class_name",
    "class_index",
    "split",
    "image_hash",
}
EXPECTED_SPLITS = {"train": "train_manifest.csv", "validation": "validation_manifest.csv"}


def load_config(config_path: Path = CONFIG_PATH) -> dict[str, Any]:
    """Load and validate the small JSON training configuration."""
    with config_path.open("r", encoding="utf-8") as config_file:
        config = json.load(config_file)

    required = {
        "seed",
        "epochs",
        "batch_size",
        "learning_rate",
        "image_size",
        "class_count",
        "dropout_rate",
        "pretrained_weights",
        "early_stopping_patience",
        "reduce_lr_patience",
        "minimum_learning_rate",
    }
    missing = required - set(config)
    if missing:
        raise ValueError(f"Training configuration is missing: {', '.join(sorted(missing))}")
    if config["class_count"] != 15:
        raise ValueError("This verified dataset must have exactly 15 classes.")
    if config["image_size"] != [224, 224]:
        raise ValueError("MobileNetV2 input size must be [224, 224].")
    if not 0 <= config["dropout_rate"] < 1:
        raise ValueError("dropout_rate must be between 0 and 1.")
    if config["pretrained_weights"] != "imagenet":
        raise ValueError("The configured MobileNetV2 weights must be 'imagenet'.")
    for key in ("seed", "epochs", "batch_size", "early_stopping_patience", "reduce_lr_patience"):
        if not isinstance(config[key], int) or config[key] <= 0:
            raise ValueError(f"{key} must be a positive integer.")
    for key in ("learning_rate", "minimum_learning_rate"):
        if not isinstance(config[key], (int, float)) or not math.isfinite(config[key]) or config[key] <= 0:
            raise ValueError(f"{key} must be a finite positive number.")

    return config


def load_class_mapping(mapping_path: Path) -> tuple[dict[str, int], list[str]]:
    """Load the saved mapping and return classes in their numeric index order."""
    with mapping_path.open("r", encoding="utf-8") as mapping_file:
        mapping_data = json.load(mapping_file)

    class_to_index = mapping_data.get("class_to_index")
    index_to_class = mapping_data.get("index_to_class")
    if not isinstance(class_to_index, dict) or not isinstance(index_to_class, dict):
        raise ValueError("class_mapping.json must contain both class mappings.")
    if len(class_to_index) != 15 or len(index_to_class) != 15:
        raise ValueError("class_mapping.json must define exactly 15 classes.")

    classes = []
    for index in range(len(class_to_index)):
        class_name = index_to_class.get(str(index))
        if class_name is None or class_to_index.get(class_name) != index:
            raise ValueError("class_mapping.json indices must be contiguous and reversible.")
        classes.append(class_name)

    return class_to_index, classes


def load_manifest(
    manifest_path: Path,
    split_name: str,
    class_to_index: dict[str, int],
) -> tuple[list[str], np.ndarray]:
    """Read paths and integer labels, checking each row against the class mapping."""
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    paths: list[str] = []
    labels: list[int] = []
    with manifest_path.open("r", newline="", encoding="utf-8-sig") as manifest_file:
        reader = csv.DictReader(manifest_file)
        if reader.fieldnames is None or not EXPECTED_MANIFEST_COLUMNS.issubset(reader.fieldnames):
            raise ValueError(
                f"{manifest_path.name} must contain columns: "
                f"{', '.join(sorted(EXPECTED_MANIFEST_COLUMNS))}"
            )

        for line_number, row in enumerate(reader, start=2):
            class_name = row["class_name"]
            if class_name not in class_to_index:
                raise ValueError(
                    f"{manifest_path.name}:{line_number} has unknown class {class_name!r}."
                )
            try:
                class_index = int(row["class_index"])
            except ValueError as error:
                raise ValueError(
                    f"{manifest_path.name}:{line_number} has an invalid class index."
                ) from error
            if class_index != class_to_index[class_name]:
                raise ValueError(
                    f"{manifest_path.name}:{line_number} class index does not match "
                    f"class {class_name!r}."
                )
            if row["split"] != split_name:
                raise ValueError(
                    f"{manifest_path.name}:{line_number} declares split "
                    f"{row['split']!r}, expected {split_name!r}."
                )
            if not row["image_hash"]:
                raise ValueError(f"{manifest_path.name}:{line_number} has no image hash.")

            relative_path = Path(row["image_path"])
            image_path = (PROJECT_ROOT / relative_path).resolve()
            if not image_path.is_relative_to(PROJECT_ROOT) or not image_path.is_file():
                raise FileNotFoundError(
                    f"{manifest_path.name}:{line_number} image path is missing or "
                    f"outside the project: {row['image_path']}"
                )

            paths.append(str(image_path))
            labels.append(class_index)

    present_classes = set(labels)
    expected_indices = set(class_to_index.values())
    if present_classes != expected_indices:
        missing_indices = sorted(expected_indices - present_classes)
        raise ValueError(
            f"{manifest_path.name} is missing class indices: {missing_indices}"
        )
    if not paths:
        raise ValueError(f"{manifest_path.name} contains no images.")

    return paths, np.asarray(labels, dtype=np.int32)


def load_class_weights(
    weights_path: Path,
    train_labels: np.ndarray,
    classes: list[str],
    class_to_index: dict[str, int],
) -> dict[int, float]:
    """Load finite class weights and confirm they were calculated from train rows."""
    with weights_path.open("r", encoding="utf-8") as weights_file:
        data = json.load(weights_file)

    if data.get("method") != "sklearn.utils.class_weight.compute_class_weight":
        raise ValueError("Class weights were not generated by the required sklearn method.")
    if data.get("class_weight") != "balanced":
        raise ValueError("Expected balanced training-only class weights.")

    weights = data.get("weights", {})
    counts = data.get("training_counts", {})
    actual_counts = np.bincount(train_labels, minlength=len(classes))
    expected_counts = {
        class_name: int(actual_counts[class_to_index[class_name]])
        for class_name in classes
    }
    if counts != expected_counts:
        raise ValueError("Saved class-weight counts do not match the training manifest.")
    if set(weights) != set(classes):
        raise ValueError("Saved class weights do not match the class mapping.")

    class_weight_by_index: dict[int, float] = {}
    for class_name in classes:
        weight = weights[class_name]
        if not isinstance(weight, (int, float)) or not math.isfinite(weight) or weight <= 0:
            raise ValueError(f"Invalid class weight for {class_name!r}.")
        class_weight_by_index[class_to_index[class_name]] = float(weight)

    return class_weight_by_index


def make_dataset(
    image_paths: list[str],
    labels: np.ndarray,
    batch_size: int,
    training: bool,
    seed: int,
) -> tf.data.Dataset:
    """Build an RGB, resized, MobileNetV2-preprocessed tf.data pipeline."""
    dataset = tf.data.Dataset.from_tensor_slices((image_paths, labels))
    if training:
        dataset = dataset.shuffle(
            buffer_size=len(image_paths),
            seed=seed,
            reshuffle_each_iteration=True,
        )

    def load_image(image_path: tf.Tensor, label: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
        image_bytes = tf.io.read_file(image_path)
        image = tf.io.decode_image(
            image_bytes,
            channels=3,
            expand_animations=False,
        )
        image.set_shape([None, None, 3])
        image = tf.image.resize(image, [224, 224])
        image = preprocess_input(image)
        image.set_shape([224, 224, 3])
        return image, label

    dataset = dataset.map(
        load_image,
        num_parallel_calls=tf.data.AUTOTUNE,
        deterministic=True,
    )
    dataset = dataset.batch(batch_size)
    return dataset.prefetch(tf.data.AUTOTUNE)


def build_model(
    class_count: int,
    learning_rate: float,
    dropout_rate: float,
    pretrained_weights: str | None = "imagenet",
) -> tuple[Model, Model]:
    """Build a frozen MobileNetV2 transfer-learning classifier."""
    if class_count != 15:
        raise ValueError("The output layer must contain exactly 15 classes.")

    base_model = MobileNetV2(
        input_shape=(224, 224, 3),
        include_top=False,
        weights=pretrained_weights,
    )
    base_model.trainable = False

    inputs = layers.Input(shape=(224, 224, 3), name="image")
    features = base_model(inputs, training=False)
    features = layers.GlobalAveragePooling2D(name="global_average_pooling")(features)
    features = layers.Dropout(dropout_rate, name="dropout")(features)
    outputs = layers.Dense(class_count, activation="softmax", name="predictions")(features)
    model = Model(inputs=inputs, outputs=outputs, name="cropcare_mobilenetv2")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model, base_model


def prepare_output_directories() -> None:
    """Create destinations for checkpoints, training history, and later evaluation."""
    for directory in (CHECKPOINT_DIR, HISTORY_DIR, EVALUATION_DIR):
        directory.mkdir(parents=True, exist_ok=True)


def create_callbacks(config: dict[str, Any]) -> list[tf.keras.callbacks.Callback]:
    """Create callbacks that save the best validation checkpoint and stop safely."""
    prepare_output_directories()
    return [
        EarlyStopping(
            monitor="val_loss",
            patience=config["early_stopping_patience"],
            restore_best_weights=True,
        ),
        ModelCheckpoint(
            filepath=str(CHECKPOINT_DIR / "best_model.keras"),
            monitor="val_loss",
            save_best_only=True,
            mode="min",
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.2,
            patience=config["reduce_lr_patience"],
            min_lr=config["minimum_learning_rate"],
            mode="min",
        ),
    ]


def train(config: dict[str, Any]) -> None:
    """Train using only train and validation manifests; never read the test split."""
    random.seed(config["seed"])
    np.random.seed(config["seed"])
    tf.keras.utils.set_random_seed(config["seed"])

    class_to_index, classes = load_class_mapping(
        PROCESSED_DIR / "class_mapping.json"
    )
    train_paths, train_labels = load_manifest(
        PROCESSED_DIR / EXPECTED_SPLITS["train"],
        "train",
        class_to_index,
    )
    validation_paths, validation_labels = load_manifest(
        PROCESSED_DIR / EXPECTED_SPLITS["validation"],
        "validation",
        class_to_index,
    )
    class_weights = load_class_weights(
        PROCESSED_DIR / "class_weights.json",
        train_labels,
        classes,
        class_to_index,
    )

    train_dataset = make_dataset(
        train_paths,
        train_labels,
        config["batch_size"],
        training=True,
        seed=config["seed"],
    )
    validation_dataset = make_dataset(
        validation_paths,
        validation_labels,
        config["batch_size"],
        training=False,
        seed=config["seed"],
    )

    # ImageNet weights may be downloaded the first time explicit training is started.
    model, base_model = build_model(
        class_count=len(classes),
        learning_rate=config["learning_rate"],
        dropout_rate=config["dropout_rate"],
        pretrained_weights=config["pretrained_weights"],
    )
    print(f"Frozen MobileNetV2 base: {base_model.name}")
    print(f"Train/validation samples: {len(train_paths)}/{len(validation_paths)}")
    print("The test manifest is intentionally not loaded or evaluated.")

    history = model.fit(
        train_dataset,
        validation_data=validation_dataset,
        epochs=config["epochs"],
        class_weight=class_weights,
        callbacks=create_callbacks(config),
    )

    history_data = {
        "config": config,
        "classes": classes,
        "epochs_completed": len(history.epoch),
        "history": {
            key: [float(value) for value in values]
            for key, values in history.history.items()
        },
        "test_evaluation": "not performed",
    }
    history_path = HISTORY_DIR / "training_history.json"
    history_path.write_text(
        json.dumps(history_data, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"Training history saved to: {history_path}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Inspect the training setup; pass --train to start fitting."
    )
    parser.add_argument(
        "--train",
        action="store_true",
        help="Explicitly start model.fit using train and validation manifests.",
    )
    args = parser.parse_args()

    try:
        config = load_config()
        _, classes = load_class_mapping(PROCESSED_DIR / "class_mapping.json")
        train_paths, train_labels = load_manifest(
            PROCESSED_DIR / EXPECTED_SPLITS["train"],
            "train",
            {name: index for index, name in enumerate(classes)},
        )
        validation_paths, validation_labels = load_manifest(
            PROCESSED_DIR / EXPECTED_SPLITS["validation"],
            "validation",
            {name: index for index, name in enumerate(classes)},
        )
        load_class_weights(
            PROCESSED_DIR / "class_weights.json",
            train_labels,
            classes,
            {name: index for index, name in enumerate(classes)},
        )
        prepare_output_directories()
    except (OSError, ValueError, json.JSONDecodeError) as error:
        parser.error(str(error))

    print(f"Python: {sys.version.split()[0]}")
    print(f"TensorFlow: {tf.__version__}")
    print(f"Keras: {tf.keras.__version__}")
    print(f"Classes: {len(classes)}")
    print(f"Train images: {len(train_paths)}; validation images: {len(validation_paths)}")
    print("Test images: intentionally not loaded.")
    print("Architecture: frozen ImageNet MobileNetV2 -> GlobalAveragePooling2D")
    print("              -> Dropout -> 15-unit softmax Dense.")
    print("Callbacks: EarlyStopping, ModelCheckpoint, ReduceLROnPlateau.")
    print(f"Configuration: {CONFIG_PATH}")
    print(f"Output directories: {CHECKPOINT_DIR}, {HISTORY_DIR}, {EVALUATION_DIR}")

    if not args.train:
        print("Preparation check only. No model was built and no training was started.")
        return 0

    train(config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
