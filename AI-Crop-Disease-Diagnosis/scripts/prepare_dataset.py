"""Create reproducible, deduplicated PlantVillage split manifests."""

import argparse
import csv
import hashlib
import json
import warnings
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, UnidentifiedImageError
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATASET_DIR = PROJECT_ROOT / "dataset" / "raw" / "PlantVillage"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "dataset" / "processed"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
RANDOM_SEED = 42
TRAIN_FRACTION = 0.70
VALIDATION_FRACTION = 0.15
TEST_FRACTION = 0.15
MANIFEST_FIELDS = (
    "image_path",
    "class_name",
    "class_index",
    "split",
    "image_hash",
)


@dataclass(frozen=True)
class ImageRecord:
    """Metadata for one readable image file."""

    image_path: str
    class_name: str
    image_hash: str
    width: int
    height: int
    image_format: str


@dataclass
class DatasetScan:
    """Validated image records and inspection details."""

    unique_records: list[ImageRecord]
    source_image_count: int
    unreadable_images: list[tuple[str, str]]
    non_image_files: list[str]
    nested_directories: list[str]
    root_files: list[str]
    duplicate_groups: list[list[ImageRecord]]
    source_formats: Counter[str]
    decoded_formats: Counter[str]
    dimensions: Counter[tuple[int, int]]


def sha256_file(path: Path) -> str:
    """Calculate a file's SHA-256 digest without loading it all into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as image_file:
        for chunk in iter(lambda: image_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def preprocess_image(image_path: Path):
    """Load one image for MobileNetV2 without applying random augmentation."""
    try:
        import tensorflow as tf
        from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
    except ImportError as error:
        raise RuntimeError(
            "TensorFlow is required for image preprocessing. "
            "Install scripts/requirements-preprocessing.txt first."
        ) from error

    image = tf.keras.utils.load_img(
        image_path,
        target_size=(224, 224),
        color_mode="rgb",
    )
    image_array = tf.keras.utils.img_to_array(image, dtype="float32")
    return preprocess_input(image_array)


def scan_dataset(dataset_dir: Path, project_root: Path) -> tuple[list[str], DatasetScan]:
    """Discover class folders, ignore non-images, and validate image files."""
    if not dataset_dir.is_dir():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")

    class_directories = sorted(
        (path for path in dataset_dir.iterdir() if path.is_dir()),
        key=lambda path: path.name.casefold(),
    )
    if not class_directories:
        raise ValueError(f"No class directories found in: {dataset_dir}")

    records_by_hash: dict[str, list[ImageRecord]] = defaultdict(list)
    unreadable_images: list[tuple[str, str]] = []
    non_image_files: list[str] = []
    nested_directories: list[str] = []
    source_formats: Counter[str] = Counter()
    decoded_formats: Counter[str] = Counter()
    dimensions: Counter[tuple[int, int]] = Counter()
    source_image_count = 0

    for class_directory in class_directories:
        paths = sorted(
            class_directory.rglob("*"),
            key=lambda path: path.as_posix().casefold(),
        )
        for path in paths:
            relative_path = path.relative_to(project_root).as_posix()
            if path.is_dir():
                nested_directories.append(relative_path)
                continue
            if not path.is_file():
                continue

            extension = path.suffix.lower()
            if extension not in IMAGE_EXTENSIONS:
                non_image_files.append(relative_path)
                continue

            source_image_count += 1
            source_formats[extension] += 1
            try:
                image_hash = sha256_file(path)
                # Treat decompression-bomb warnings as invalid input, not as images.
                with warnings.catch_warnings():
                    warnings.simplefilter("error", Image.DecompressionBombWarning)
                    with Image.open(path) as image:
                        image_format = (image.format or "UNKNOWN").upper()
                        width, height = image.size
                        image.load()
                if width <= 0 or height <= 0:
                    raise ValueError("Image has an invalid size.")
            except (
                Image.DecompressionBombError,
                Image.DecompressionBombWarning,
                OSError,
                UnidentifiedImageError,
                ValueError,
            ) as error:
                unreadable_images.append(
                    (relative_path, str(error) or type(error).__name__)
                )
                continue

            record = ImageRecord(
                image_path=relative_path,
                class_name=class_directory.name,
                image_hash=image_hash,
                width=width,
                height=height,
                image_format=image_format,
            )
            records_by_hash[image_hash].append(record)
            decoded_formats[image_format] += 1
            dimensions[(width, height)] += 1

    duplicate_groups = [
        sorted(group, key=lambda record: record.image_path.casefold())
        for group in records_by_hash.values()
        if len(group) > 1
    ]
    duplicate_groups.sort(key=lambda group: group[0].image_path.casefold())

    cross_class_groups = [
        group for group in duplicate_groups
        if len({record.class_name for record in group}) > 1
    ]
    if cross_class_groups:
        examples = "; ".join(
            ", ".join(
                f"{record.class_name}: {record.image_path}" for record in group
            )
            for group in cross_class_groups[:3]
        )
        raise ValueError(
            "Exact duplicate image content has conflicting class labels. "
            "Resolve the labels before splitting to prevent leakage. "
            f"Examples: {examples}"
        )

    # Keep one deterministic representative for each content hash.
    unique_records = sorted(
        (group[0] for group in records_by_hash.values()),
        key=lambda record: record.image_path.casefold(),
    )
    root_files = sorted(
        (
            path.relative_to(project_root).as_posix()
            for path in dataset_dir.iterdir()
            if path.is_file()
        ),
        key=str.casefold,
    )

    return [path.name for path in class_directories], DatasetScan(
        unique_records=unique_records,
        source_image_count=source_image_count,
        unreadable_images=unreadable_images,
        non_image_files=sorted(non_image_files, key=str.casefold),
        nested_directories=sorted(nested_directories, key=str.casefold),
        root_files=root_files,
        duplicate_groups=duplicate_groups,
        source_formats=source_formats,
        decoded_formats=decoded_formats,
        dimensions=dimensions,
    )


def create_splits(
    records: list[ImageRecord],
    classes: list[str],
    seed: int = RANDOM_SEED,
) -> dict[str, list[ImageRecord]]:
    """Create reproducible stratified 70/15/15 manifests."""
    if not records:
        raise ValueError("No readable, unique images are available for splitting.")

    class_to_index = {class_name: index for index, class_name in enumerate(classes)}
    counts = Counter(record.class_name for record in records)
    insufficient = {
        class_name: counts[class_name]
        for class_name in classes
        if counts[class_name] < 3
    }
    if insufficient:
        details = ", ".join(
            f"{class_name}={count}" for class_name, count in insufficient.items()
        )
        raise ValueError(
            "Stratified 70/15/15 splitting requires at least 3 unique readable "
            f"images per class; cannot continue: {details}"
        )

    labels = [class_to_index[record.class_name] for record in records]
    try:
        training, holdout, training_labels, holdout_labels = train_test_split(
            records,
            labels,
            test_size=VALIDATION_FRACTION + TEST_FRACTION,
            random_state=seed,
            stratify=labels,
        )
        validation, testing, validation_labels, test_labels = train_test_split(
            holdout,
            holdout_labels,
            test_size=TEST_FRACTION / (VALIDATION_FRACTION + TEST_FRACTION),
            random_state=seed,
            stratify=holdout_labels,
        )
    except ValueError as error:
        raise ValueError(
            "Stratified splitting failed. Check that every class has enough "
            f"unique images for all splits: {error}"
        ) from error

    splits = {
        "train": list(training),
        "validation": list(validation),
        "test": list(testing),
    }

    for split_name, split_records in splits.items():
        split_classes = {record.class_name for record in split_records}
        missing = sorted(set(classes) - split_classes)
        if missing:
            raise ValueError(
                f"The {split_name} split is missing classes: {', '.join(missing)}"
            )

    del training_labels, validation_labels, test_labels
    return splits


def write_manifests(
    output_dir: Path,
    splits: dict[str, list[ImageRecord]],
    class_to_index: dict[str, int],
) -> None:
    """Write CSV manifests that reference, but do not copy, source images."""
    output_dir.mkdir(parents=True, exist_ok=True)
    for split_name, records in splits.items():
        manifest_path = output_dir / f"{split_name}_manifest.csv"
        with manifest_path.open("w", newline="", encoding="utf-8") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=MANIFEST_FIELDS)
            writer.writeheader()
            for record in sorted(
                records,
                key=lambda item: (class_to_index[item.class_name], item.image_path.casefold()),
            ):
                writer.writerow(
                    {
                        "image_path": record.image_path,
                        "class_name": record.class_name,
                        "class_index": class_to_index[record.class_name],
                        "split": split_name,
                        "image_hash": record.image_hash,
                    }
                )


def _percentage(count: int, total: int) -> str:
    """Format a split percentage without dividing by zero."""
    return f"{count / total * 100:.2f}%" if total else "0.00%"


def create_class_weights(
    train_records: list[ImageRecord],
    classes: list[str],
    class_to_index: dict[str, int],
) -> tuple[dict[str, float], dict[str, int]]:
    """Compute balanced class weights using training records only."""
    import numpy as np

    training_counts = Counter(record.class_name for record in train_records)
    labels = np.asarray(
        [class_to_index[record.class_name] for record in train_records],
        dtype=np.int64,
    )
    class_indices = np.arange(len(classes), dtype=np.int64)
    weights = compute_class_weight(
        class_weight="balanced",
        classes=class_indices,
        y=labels,
    )
    return (
        {
            class_name: float(weights[class_to_index[class_name]])
            for class_name in classes
        },
        {class_name: training_counts[class_name] for class_name in classes},
    )


def write_json(path: Path, data: dict[str, object]) -> None:
    """Write stable, human-readable JSON."""
    path.write_text(
        json.dumps(data, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def build_report(
    dataset_dir: Path,
    classes: list[str],
    scan: DatasetScan,
    splits: dict[str, list[ImageRecord]],
    weights: dict[str, float],
    training_counts: dict[str, int],
) -> str:
    """Build the preprocessing report from actual scan and split results."""
    total_unique = len(scan.unique_records)
    unique_class_counts = Counter(
        record.class_name for record in scan.unique_records
    )
    smallest_class_count = min(unique_class_counts.values(), default=0)
    largest_class_count = max(unique_class_counts.values(), default=0)
    class_imbalance_ratio = (
        largest_class_count / smallest_class_count
        if smallest_class_count
        else 0
    )
    split_names = ("train", "validation", "test")
    lines = [
        "CropCare AI - MobileNetV2 Dataset Preprocessing Report",
        "=" * 58,
        f"Source dataset: {dataset_dir.resolve()}",
        "Source images were read only; no source files were copied, deleted, or modified.",
        f"Random seed: {RANDOM_SEED}",
        "Requested split ratios: train=70%, validation=15%, test=15%",
        "Image input pipeline: RGB, resized to 224 x 224, MobileNetV2 preprocess_input.",
        "Random augmentation: none.",
        "",
        "Dataset inspection",
        f"  Discovered classes: {len(classes)}",
        f"  Source image files: {scan.source_image_count}",
        f"  Readable unique images: {total_unique}",
        f"  Exact duplicate groups: {len(scan.duplicate_groups)}",
        f"  Duplicate copies excluded: {scan.source_image_count - total_unique - len(scan.unreadable_images)}",
        f"  Corrupted/unreadable images excluded: {len(scan.unreadable_images)}",
        f"  Non-image files ignored: {len(scan.non_image_files)}",
        f"  Nested directories inside classes: {len(scan.nested_directories)}",
        f"  Files directly under source root: {len(scan.root_files)}",
        f"  Unique-image largest/smallest class ratio: {class_imbalance_ratio:.2f}:1",
        "",
        "Source filename formats:",
    ]
    lines.extend(
        f"  {extension}: {count}"
        for extension, count in sorted(scan.source_formats.items())
    )
    lines.append("Decoded image formats:")
    lines.extend(
        f"  {image_format}: {count}"
        for image_format, count in sorted(scan.decoded_formats.items())
    )
    lines.append("Source dimensions:")
    lines.extend(
        f"  {width} x {height}: {count}"
        for (width, height), count in sorted(scan.dimensions.items())
    )

    if scan.unreadable_images:
        lines.extend(["", "Unreadable images excluded:"])
        lines.extend(
            f"  {path}: {message}" for path, message in scan.unreadable_images
        )
    if scan.non_image_files:
        lines.extend(["", "Non-image files ignored:"])
        lines.extend(f"  {path}" for path in scan.non_image_files)
    if scan.nested_directories:
        lines.extend(["", "Nested directories found:"])
        lines.extend(f"  {path}" for path in scan.nested_directories)
    if scan.root_files:
        lines.extend(["", "Files directly under source root:"])
        lines.extend(f"  {path}" for path in scan.root_files)

    lines.extend(["", "Exact duplicates excluded (representative retained):"])
    if not scan.duplicate_groups:
        lines.append("  None.")
    for group in scan.duplicate_groups:
        representative = group[0]
        lines.append(
            f"  SHA-256 {representative.image_hash}: retained {representative.image_path}"
        )
        lines.extend(f"    excluded {record.image_path}" for record in group[1:])

    lines.extend(
        [
            "",
            "Split totals:",
            f"  Total unique images: {total_unique}",
        ]
    )
    for split_name in split_names:
        count = len(splits[split_name])
        target = {
            "train": TRAIN_FRACTION,
            "validation": VALIDATION_FRACTION,
            "test": TEST_FRACTION,
        }[split_name]
        lines.append(
            f"  {split_name}: {count} ({_percentage(count, total_unique)}; target {target:.0%})"
        )
    lines.append(
        "  Rounding note: stratified split sizes use integer sample counts; "
        "scikit-learn rounds holdout sizes up, so reported percentages may differ "
        "slightly from the target ratios."
    )

    for split_name in split_names:
        split_records = splits[split_name]
        split_counts = Counter(record.class_name for record in split_records)
        lines.extend(
            [
                "",
                f"{split_name.title()} class distribution:",
            ]
        )
        for class_name in classes:
            lines.append(
                f"  {class_name}: {split_counts[class_name]} "
                f"({_percentage(split_counts[class_name], len(split_records))})"
            )

    lines.extend(["", "Training-only balanced class weights (scikit-learn):"])
    for class_name in classes:
        lines.append(
            f"  {class_name}: {weights[class_name]:.8f} "
            f"(training samples: {training_counts[class_name]})"
        )

    warnings_found = []
    if scan.unreadable_images:
        warnings_found.append(
            f"{len(scan.unreadable_images)} unreadable image(s) were excluded."
        )
    if scan.non_image_files:
        warnings_found.append(
            f"{len(scan.non_image_files)} non-image file(s) were ignored."
        )
    if scan.duplicate_groups:
        warnings_found.append(
            f"{len(scan.duplicate_groups)} exact duplicate group(s) were deduplicated."
        )
    if class_imbalance_ratio > 5:
        warnings_found.append(
            f"Class imbalance remains substantial ({class_imbalance_ratio:.2f}:1 "
            f"between {largest_class_count} and {smallest_class_count} images); "
            "training-only class weights are provided."
        )
    if scan.nested_directories:
        warnings_found.append(
            "Nested directories exist below class folders; their readable images "
            "were assigned to the parent class."
        )
    if scan.root_files:
        warnings_found.append("Files directly under the source dataset root were ignored.")
    lines.extend(["", "Warnings:"])
    lines.extend(f"  - {warning}" for warning in warnings_found)
    if not warnings_found:
        lines.append("  None.")

    lines.extend(
        [
            "",
            "Training status: preprocessing only; no model was trained.",
            "Run scripts/verify_preprocessing.py to verify manifests and MobileNetV2 preprocessing.",
        ]
    )
    return "\n".join(lines) + "\n"


def prepare_dataset(
    dataset_dir: Path = DEFAULT_DATASET_DIR,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    project_root: Path = PROJECT_ROOT,
) -> str:
    """Scan, deduplicate, split, weight, and report without copying images."""
    classes, scan = scan_dataset(dataset_dir, project_root)
    counts = Counter(record.class_name for record in scan.unique_records)
    insufficient = {
        class_name: counts[class_name]
        for class_name in classes
        if counts[class_name] < 3
    }
    if insufficient:
        details = ", ".join(
            f"{class_name}={count}" for class_name, count in insufficient.items()
        )
        raise ValueError(
            "Cannot stratify the requested three splits: every class needs at least "
            f"3 readable unique images. Insufficient classes: {details}"
        )

    class_to_index = {class_name: index for index, class_name in enumerate(classes)}
    splits = create_splits(scan.unique_records, classes)
    weights, training_counts = create_class_weights(
        splits["train"],
        classes,
        class_to_index,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    write_manifests(output_dir, splits, class_to_index)
    write_json(
        output_dir / "class_mapping.json",
        {
            "class_to_index": class_to_index,
            "index_to_class": {
                str(index): class_name for class_name, index in class_to_index.items()
            },
        },
    )
    write_json(
        output_dir / "class_weights.json",
        {
            "method": "sklearn.utils.class_weight.compute_class_weight",
            "class_weight": "balanced",
            "training_counts": training_counts,
            "weights": weights,
        },
    )

    report = build_report(
        dataset_dir,
        classes,
        scan,
        splits,
        weights,
        training_counts,
    )
    (output_dir / "preprocessing_report.txt").write_text(
        report,
        encoding="utf-8",
    )
    return report


def main() -> int:
    """Run preprocessing using command-line paths or project defaults."""
    parser = argparse.ArgumentParser(
        description="Build deduplicated, stratified image manifests for training."
    )
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    try:
        report = prepare_dataset(args.dataset_dir, args.output_dir)
    except (FileNotFoundError, ValueError, OSError) as error:
        parser.error(str(error))

    print(report, end="")
    print(f"\nProcessed manifests and reports saved to: {args.output_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
