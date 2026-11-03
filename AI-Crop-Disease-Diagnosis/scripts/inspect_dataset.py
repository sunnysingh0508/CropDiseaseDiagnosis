"""Validate the PlantVillage dataset and optionally save a summary report."""

import argparse
import hashlib
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image, UnidentifiedImageError

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
EXPECTED_CLASSES = {
    "Pepper__bell___Bacterial_spot",
    "Pepper__bell___healthy",
    "Potato___Early_blight",
    "Potato___Late_blight",
    "Potato___healthy",
    "Tomato_Bacterial_spot",
    "Tomato_Early_blight",
    "Tomato_Late_blight",
    "Tomato_Leaf_Mold",
    "Tomato_Septoria_leaf_spot",
    "Tomato_Spider_mites_Two_spotted_spider_mite",
    "Tomato__Target_Spot",
    "Tomato__Tomato_mosaic_virus",
    "Tomato__Tomato_YellowLeaf__Curl_Virus",
    "Tomato_healthy",
}
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATASET_DIR = PROJECT_ROOT / "dataset" / "raw" / "PlantVillage"
DEFAULT_SUMMARY_FILE = PROJECT_ROOT / "dataset" / "dataset_summary.txt"


def _format_percent(count: int, total: int) -> str:
    """Return a percentage string, handling an empty dataset safely."""
    if not total:
        return "0.00%"
    return f"{count / total * 100:.2f}%"


def _sha256(file_path: Path) -> str:
    """Hash a file in small chunks so large files do not fill memory."""
    digest = hashlib.sha256()
    with file_path.open("rb") as image_file:
        for chunk in iter(lambda: image_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def inspect_dataset(dataset_dir: Path) -> tuple[str, int]:
    """Validate image files and return a text report and total image count."""
    if not dataset_dir.is_dir():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")

    class_directories = sorted(
        (path for path in dataset_dir.iterdir() if path.is_dir()),
        key=lambda path: path.name.casefold(),
    )
    if not class_directories:
        raise ValueError(f"No class directories found in: {dataset_dir}")

    class_counts: Counter[str] = Counter()
    extension_counts: Counter[str] = Counter()
    decoded_format_counts: Counter[str] = Counter()
    dimension_counts: Counter[tuple[int, int]] = Counter()
    class_hashes: dict[str, list[tuple[str, str]]] = defaultdict(list)
    errors: list[tuple[str, str]] = []
    non_image_files: list[str] = []
    nested_directories: list[str] = []
    extension_mismatches: list[tuple[str, str, str]] = []
    unexpected_root_files = sorted(
        (path.name for path in dataset_dir.iterdir() if path.is_file()),
        key=str.casefold,
    )
    total_images = 0
    valid_images = 0

    for class_directory in class_directories:
        for path in sorted(class_directory.rglob("*"), key=lambda item: str(item).casefold()):
            relative_path = path.relative_to(dataset_dir).as_posix()
            if path.is_dir():
                nested_directories.append(relative_path)
                continue
            if not path.is_file():
                continue

            extension = path.suffix.lower()
            if extension not in IMAGE_EXTENSIONS:
                non_image_files.append(relative_path)
                continue

            total_images += 1
            class_counts[class_directory.name] += 1
            extension_counts[extension] += 1

            try:
                file_hash = _sha256(path)
            except OSError as error:
                errors.append((relative_path, f"Could not read file for hashing: {error}"))
                continue
            class_hashes[file_hash].append((class_directory.name, relative_path))

            try:
                # Load the pixels as well as opening the header to detect damaged data.
                with Image.open(path) as image:
                    image_format = (image.format or "UNKNOWN").upper()
                    image.load()
                    dimensions = image.size
                valid_images += 1
                decoded_format_counts[image_format] += 1
                dimension_counts[dimensions] += 1

                expected_extensions = {
                    "JPEG": {".jpg", ".jpeg"},
                    "PNG": {".png"},
                    "WEBP": {".webp"},
                    "BMP": {".bmp"},
                }
                if extension not in expected_extensions.get(image_format, set()):
                    extension_mismatches.append(
                        (relative_path, extension, image_format)
                    )
            except (
                Image.DecompressionBombError,
                Image.DecompressionBombWarning,
                OSError,
                UnidentifiedImageError,
                ValueError,
            ) as error:
                errors.append((relative_path, str(error) or type(error).__name__))

    actual_classes = {path.name for path in class_directories}
    missing_classes = sorted(EXPECTED_CLASSES - actual_classes)
    unexpected_classes = sorted(actual_classes - EXPECTED_CLASSES)
    total_class_entries = sum(class_counts.values())
    largest_class = max(class_counts.values(), default=0)
    smallest_class = min(class_counts.values(), default=0)
    imbalance_ratio = (
        largest_class / smallest_class
        if smallest_class
        else 0
    )

    duplicate_groups = [
        entries for entries in class_hashes.values() if len(entries) > 1
    ]
    duplicate_images = sum(len(entries) - 1 for entries in duplicate_groups)
    duplicate_cross_class_groups = [
        entries
        for entries in duplicate_groups
        if len({class_name for class_name, _ in entries}) > 1
    ]

    report_lines = [
        "CropCare AI - PlantVillage Dataset Validation Report",
        "=" * 58,
        f"Dataset directory: {dataset_dir.resolve()}",
        "Inspection: read-only; source images were not changed.",
        "",
        "Dataset structure",
        f"  Expected classes: {len(EXPECTED_CLASSES)}",
        f"  Found class directories: {len(class_directories)}",
        f"  Missing expected classes: {len(missing_classes)}",
    ]
    if missing_classes:
        report_lines.extend(f"    - {name}" for name in missing_classes)
    report_lines.append(f"  Unexpected class directories: {len(unexpected_classes)}")
    if unexpected_classes:
        report_lines.extend(f"    - {name}" for name in unexpected_classes)
    report_lines.append(f"  Nested directories inside class folders: {len(nested_directories)}")
    if nested_directories:
        report_lines.extend(f"    - {name}" for name in nested_directories)
    report_lines.append(f"  Files directly under dataset root: {len(unexpected_root_files)}")
    if unexpected_root_files:
        report_lines.extend(f"    - {name}" for name in unexpected_root_files)

    report_lines.extend(
        [
            "",
            "Image validation",
            f"  Image files counted: {total_images}",
            f"  Images decoded successfully: {valid_images}",
            f"  Corrupted/unreadable images: {len(errors)}",
        ]
    )
    if errors:
        report_lines.extend(f"    - {path}: {message}" for path, message in errors)
    report_lines.append(f"  Non-image files inside class folders: {len(non_image_files)}")
    if non_image_files:
        report_lines.extend(f"    - {path}" for path in non_image_files)

    report_lines.extend(["", "Image formats by filename extension:"])
    if extension_counts:
        report_lines.extend(
            f"  {extension}: {count}"
            for extension, count in sorted(extension_counts.items())
        )
    else:
        report_lines.append("  No supported image files found.")

    report_lines.append("Decoded image formats:")
    if decoded_format_counts:
        report_lines.extend(
            f"  {image_format}: {count}"
            for image_format, count in sorted(decoded_format_counts.items())
        )
    else:
        report_lines.append("  No readable images.")
    report_lines.append(f"  Extension/format mismatches: {len(extension_mismatches)}")
    if extension_mismatches:
        report_lines.extend(
            f"    - {path}: {extension} extension, decoded as {image_format}"
            for path, extension, image_format in extension_mismatches
        )

    report_lines.extend(
        [
            "",
            "Image dimensions (width x height; successfully decoded images):",
        ]
    )
    if dimension_counts:
        report_lines.extend(
            f"  {width} x {height}: {count}"
            for (width, height), count in sorted(dimension_counts.items())
        )
    else:
        report_lines.append("  No readable images.")

    report_lines.extend(
        [
            "",
            f"Class counts and dataset percentages (denominator: {total_class_entries} image files):",
        ]
    )
    for class_directory in class_directories:
        count = class_counts[class_directory.name]
        report_lines.append(
            f"  {class_directory.name}: {count} ({_format_percent(count, total_class_entries)})"
        )

    report_lines.extend(
        [
            "",
            f"Exact duplicate SHA-256 hash groups: {len(duplicate_groups)}",
            f"Extra duplicate image copies: {duplicate_images}",
            f"Duplicate groups spanning multiple classes: {len(duplicate_cross_class_groups)}",
        ]
    )
    for entries in duplicate_groups:
        report_lines.append("  Duplicate SHA-256 group:")
        report_lines.extend(
            f"    - [{class_name}] {path}" for class_name, path in entries
        )

    report_lines.extend(
        [
            "",
            "Class imbalance",
            f"  Largest class: {largest_class} images",
            f"  Smallest class: {smallest_class} images",
            f"  Largest/smallest ratio: {imbalance_ratio:.2f}:1",
        ]
    )

    structure_ok = (
        len(class_directories) == len(EXPECTED_CLASSES)
        and not missing_classes
        and not unexpected_classes
        and not nested_directories
        and not unexpected_root_files
    )
    integrity_ok = (
        total_images == valid_images
        and not errors
        and not extension_mismatches
        and not non_image_files
    )
    duplicates_ok = not duplicate_groups
    ready = structure_ok and integrity_ok and duplicates_ok and imbalance_ratio <= 5

    report_lines.extend(
        [
            "",
            "MobileNetV2 training readiness",
            f"  Structure check: {'PASS' if structure_ok else 'ISSUE'}",
            f"  Image integrity/format check: {'PASS' if integrity_ok else 'ISSUE'}",
            f"  Duplicate check: {'PASS' if duplicates_ok else 'ISSUE'}",
            f"  Balance check (5:1 project screening heuristic): {'PASS' if imbalance_ratio <= 5 else 'ISSUE'}",
            f"  Overall: {'READY for training preparation' if ready else 'NOT READY for training as-is'}",
            "  Note: MobileNetV2 still requires a documented preprocessing pipeline, "
            "stratified train/validation/test splits, and evaluation.",
        ]
    )
    report = "\n".join(report_lines) + "\n"
    return report, total_images


def main() -> int:
    """Parse options, print the report, and optionally save it."""
    parser = argparse.ArgumentParser(
        description="Validate image integrity, class balance, dimensions, and duplicates."
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=DEFAULT_DATASET_DIR,
        help=f"Dataset directory to inspect (default: {DEFAULT_DATASET_DIR})",
    )
    parser.add_argument(
        "--write-summary",
        nargs="?",
        const=DEFAULT_SUMMARY_FILE,
        type=Path,
        metavar="PATH",
        help=(
            "Write the report to dataset/dataset_summary.txt, "
            "or to PATH if provided."
        ),
    )
    args = parser.parse_args()

    try:
        report, _ = inspect_dataset(args.dataset_dir)
    except (FileNotFoundError, ValueError) as error:
        parser.error(str(error))

    print(report, end="")
    if args.write_summary:
        args.write_summary.parent.mkdir(parents=True, exist_ok=True)
        args.write_summary.write_text(report, encoding="utf-8")
        print(f"\nSaved validation report to: {args.write_summary.resolve()}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
