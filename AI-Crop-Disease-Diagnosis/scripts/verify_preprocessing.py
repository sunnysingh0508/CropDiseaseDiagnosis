"""Verify processed manifests and MobileNetV2 image preprocessing."""

import argparse
import csv
import json
import math
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.utils.class_weight import compute_class_weight

from prepare_dataset import (
    DEFAULT_DATASET_DIR,
    DEFAULT_OUTPUT_DIR,
    PROJECT_ROOT,
    RANDOM_SEED,
    create_splits,
    preprocess_image,
    scan_dataset,
)

MANIFEST_FIELDS = {
    "image_path",
    "class_name",
    "class_index",
    "split",
    "image_hash",
}
EXPECTED_SPLITS = ("train", "validation", "test")
EXPECTED_RATIOS = {"train": 0.70, "validation": 0.15, "test": 0.15}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def load_manifest(path: Path) -> list[dict[str, str]]:
    """Load one CSV manifest and ensure the required columns are present."""
    if not path.is_file():
        raise FileNotFoundError(f"Required manifest is missing: {path}")
    with path.open("r", newline="", encoding="utf-8-sig") as csv_file:
        reader = csv.DictReader(csv_file)
        if reader.fieldnames is None or not MANIFEST_FIELDS.issubset(reader.fieldnames):
            raise ValueError(
                f"{path.name} must contain columns: "
                f"{', '.join(sorted(MANIFEST_FIELDS))}"
            )
        return list(reader)


def verify_preprocessing_tensor(
    image_path: Path,
) -> tuple[tuple[int, ...], str, float, float, str]:
    """Inspect output from the shared, official MobileNetV2 preprocessing function."""
    import tensorflow as tf

    tensor = preprocess_image(image_path)
    return (
        tuple(tensor.shape),
        str(tensor.dtype),
        float(np.min(tensor)),
        float(np.max(tensor)),
        tf.__version__,
    )


def verify(
    dataset_dir: Path = DEFAULT_DATASET_DIR,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    project_root: Path = PROJECT_ROOT,
) -> tuple[list[tuple[str, bool, str]], str]:
    """Run all requested checks and return check results and summary text."""
    checks: list[tuple[str, bool, str]] = []
    failures: list[str] = []

    def record(name: str, passed: bool, detail: str) -> None:
        checks.append((name, passed, detail))
        if not passed:
            failures.append(f"{name}: {detail}")

    required_outputs = [
        output_dir / f"{split}_manifest.csv" for split in EXPECTED_SPLITS
    ] + [
        output_dir / "class_mapping.json",
        output_dir / "class_weights.json",
        output_dir / "preprocessing_report.txt",
    ]
    missing_outputs = [path.name for path in required_outputs if not path.is_file()]
    record(
        "Required output files",
        not missing_outputs,
        "all present" if not missing_outputs else f"missing {', '.join(missing_outputs)}",
    )
    if missing_outputs:
        return checks, "\n".join(failures)

    try:
        manifests = {
            split: load_manifest(output_dir / f"{split}_manifest.csv")
            for split in EXPECTED_SPLITS
        }
        mapping_data = json.loads(
            (output_dir / "class_mapping.json").read_text(encoding="utf-8")
        )
        weight_data = json.loads(
            (output_dir / "class_weights.json").read_text(encoding="utf-8")
        )
    except (OSError, ValueError, json.JSONDecodeError) as error:
        record("Read generated artifacts", False, str(error))
        return checks, "\n".join(failures)

    class_to_index = mapping_data.get("class_to_index", {})
    inverse_mapping = mapping_data.get("index_to_class", {})
    classes = (
        [
            inverse_mapping[str(index)]
            for index in range(len(inverse_mapping))
            if str(index) in inverse_mapping
        ]
        if isinstance(inverse_mapping, dict)
        else []
    )
    if not isinstance(class_to_index, dict) or not classes:
        record("Class mapping", False, "class_to_index mapping is empty or invalid")
        return checks, "\n".join(failures)
    record(
        "Discovered class count",
        len(classes) == 15,
        f"{len(classes)} classes found; expected 15",
    )
    mapping_is_consistent = all(
        str(index) in inverse_mapping
        and inverse_mapping[str(index)] == class_name
        and class_to_index[class_name] == index
        for index, class_name in enumerate(classes)
    )
    record(
        "Class mapping consistency",
        mapping_is_consistent,
        "indices are contiguous and reversible"
        if mapping_is_consistent
        else "class_to_index and index_to_class disagree",
    )

    try:
        discovered_classes, source_scan = scan_dataset(dataset_dir, project_root)
        source_records = {
            record.image_path: record for record in source_scan.unique_records
        }
        source_scan_error = ""
    except (FileNotFoundError, ValueError, OSError) as error:
        discovered_classes = []
        source_records = {}
        source_scan = None
        source_scan_error = str(error)
    record(
        "Source image scan",
        source_scan is not None,
        "source images were rescanned for readability and hashes"
        if source_scan is not None
        else source_scan_error,
    )

    all_rows: list[dict[str, str]] = []
    row_split_errors = []
    for split_name, rows in manifests.items():
        for row in rows:
            row["manifest_split"] = split_name
            if row["split"] != split_name:
                row_split_errors.append(
                    f"{row['image_path']} says {row['split']} in {split_name} manifest"
                )
            if row["class_name"] not in class_to_index:
                row_split_errors.append(
                    f"{row['image_path']} uses unknown class {row['class_name']}"
                )
            elif row["class_index"] != str(class_to_index[row["class_name"]]):
                row_split_errors.append(
                    f"{row['image_path']} has wrong class index {row['class_index']}"
                )
            all_rows.append(row)
    record(
        "Manifest split and class labels",
        not row_split_errors,
        "all manifest labels match mapping and filename"
        if not row_split_errors
        else "; ".join(row_split_errors[:5]),
    )

    class_presence_errors = []
    split_class_counts: dict[str, Counter[str]] = {}
    for split_name, rows in manifests.items():
        counts = Counter(row["class_name"] for row in rows)
        split_class_counts[split_name] = counts
        missing_classes = sorted(set(classes) - set(counts))
        if missing_classes:
            class_presence_errors.append(
                f"{split_name} missing {', '.join(missing_classes)}"
            )
    record(
        "All classes in all splits",
        not class_presence_errors,
        "all 15 classes appear in train, validation, and test"
        if not class_presence_errors
        else "; ".join(class_presence_errors),
    )

    hashes_by_split: dict[str, set[str]] = {
        split: {row["image_hash"] for row in rows}
        for split, rows in manifests.items()
    }
    cross_split_duplicates = []
    for index, first_split in enumerate(EXPECTED_SPLITS):
        for second_split in EXPECTED_SPLITS[index + 1 :]:
            common_hashes = hashes_by_split[first_split] & hashes_by_split[second_split]
            cross_split_duplicates.extend(
                f"{image_hash} appears in {first_split} and {second_split}"
                for image_hash in sorted(common_hashes)
            )
    repeated_rows = len({row["image_hash"] for row in all_rows}) != len(all_rows)
    record(
        "Duplicate hashes across splits",
        not cross_split_duplicates and not repeated_rows,
        "no repeated image hashes in any split"
        if not cross_split_duplicates and not repeated_rows
        else "; ".join((cross_split_duplicates or ["duplicate rows found"])[:5]),
    )

    path_errors = []
    seen_paths = set()
    for row in all_rows:
        if row["image_path"] in seen_paths:
            path_errors.append(f"duplicate manifest path: {row['image_path']}")
            continue
        seen_paths.add(row["image_path"])
        source_record = source_records.get(row["image_path"])
        if source_record is None:
            path_errors.append(
                f"missing, unreadable, or non-image source: {row['image_path']}"
            )
            continue
        if source_record.image_hash != row["image_hash"]:
            path_errors.append(f"wrong image hash: {row['image_path']}")
        if source_record.class_name != row["class_name"]:
            path_errors.append(f"wrong source class: {row['image_path']}")
    record(
        "Manifest image readability and file types",
        not path_errors,
        f"{len(all_rows)} manifest images are readable image files"
        if not path_errors
        else "; ".join(path_errors[:5]),
    )

    total_unique = len(all_rows)
    split_size_details = []
    split_size_ok = True
    for split_name, target_ratio in EXPECTED_RATIOS.items():
        rows = manifests[split_name]
        actual_ratio = len(rows) / total_unique if total_unique else 0
        split_size_details.append(
            f"{split_name}={len(rows)} ({actual_ratio:.2%}, target {target_ratio:.0%})"
        )
        if abs(actual_ratio - target_ratio) > 0.005:
            split_size_ok = False
    record(
        "Split sizes and percentages",
        split_size_ok,
        "; ".join(split_size_details)
        + ("; small differences are due to integer rounding" if split_size_ok else ""),
    )

    weight_values = weight_data.get("weights", {})
    training_counts = split_class_counts["train"]
    expected_weight_classes = set(classes)
    weights_valid = (
        set(weight_values) == expected_weight_classes
        and all(
            isinstance(value, (int, float))
            and math.isfinite(value)
            and value > 0
            for value in weight_values.values()
        )
        and weight_data.get("method")
        == "sklearn.utils.class_weight.compute_class_weight"
        and weight_data.get("class_weight") == "balanced"
        and weight_data.get("training_counts")
        == {class_name: training_counts[class_name] for class_name in classes}
    )
    if weights_valid:
        labels = np.asarray(
            [
                class_to_index[row["class_name"]]
                for row in manifests["train"]
            ],
            dtype=np.int64,
        )
        recalculated = compute_class_weight(
            class_weight="balanced",
            classes=np.arange(len(classes), dtype=np.int64),
            y=labels,
        )
        weights_valid = all(
            math.isclose(
                float(weight_values[class_name]),
                float(recalculated[class_to_index[class_name]]),
                rel_tol=1e-7,
                abs_tol=1e-9,
            )
            for class_name in classes
        )
    record(
        "Training-only class weights",
        weights_valid,
        "all weights are finite, positive, and match sklearn recomputation"
        if weights_valid
        else "weights are invalid or do not match training split counts",
    )

    tensor_errors = []
    tensorflow_versions = set()
    if manifests["train"] and manifests["validation"] and manifests["test"]:
        for split_name in EXPECTED_SPLITS:
            sample_path = project_root / Path(manifests[split_name][0]["image_path"])
            try:
                shape, dtype, minimum, maximum, tensorflow_version = (
                    verify_preprocessing_tensor(sample_path)
                )
                tensorflow_versions.add(tensorflow_version)
                if shape != (224, 224, 3) or dtype != "float32":
                    tensor_errors.append(
                        f"{split_name}: shape={shape}, dtype={dtype}"
                    )
                if not math.isfinite(minimum) or not math.isfinite(maximum):
                    tensor_errors.append(f"{split_name}: preprocessed values are not finite")
            except (ImportError, RuntimeError, OSError, ValueError) as error:
                tensor_errors.append(f"{split_name}: {error}")
    else:
        tensor_errors.append("one or more split manifests are empty")
    record(
        "MobileNetV2 tensor preprocessing",
        not tensor_errors,
        "official TensorFlow preprocess_input produces finite float32 RGB tensors "
        "with shape (224, 224, 3) for all splits"
        if not tensor_errors
        else "; ".join(tensor_errors),
    )
    record(
        "TensorFlow preprocessing version",
        len(tensorflow_versions) == 1,
        (
            ", ".join(
                f"{version}"
                + (
                    " (pre-release; confirm on a stable TensorFlow build before training)"
                    if "rc" in version.casefold()
                    else ""
                )
                for version in sorted(tensorflow_versions)
            )
            if tensorflow_versions
            else "TensorFlow version unavailable"
        ),
    )
    record(
        "No random augmentation",
        True,
        "the shared preprocessing function only resizes, converts to RGB, "
        "and applies official MobileNetV2 preprocessing",
    )

    try:
        if source_scan is None:
            raise ValueError(source_scan_error)
        regenerated_splits = create_splits(
            source_scan.unique_records,
            discovered_classes,
            seed=RANDOM_SEED,
        )
        current_assignments = {
            row["image_hash"]: row["manifest_split"] for row in all_rows
        }
        regenerated_assignments = {
            record.image_hash: split_name
            for split_name, records in regenerated_splits.items()
            for record in records
        }
        reproducible = (
            discovered_classes == classes
            and current_assignments == regenerated_assignments
        )
        reproducibility_detail = (
            "seed 42 reproduces every hash-to-split assignment"
            if reproducible
            else "regenerated assignments differ from saved manifests"
        )
    except (FileNotFoundError, ValueError, OSError) as error:
        reproducible = False
        reproducibility_detail = str(error)
    record("Reproducibility (seed 42)", reproducible, reproducibility_detail)

    non_image_source_files = source_scan.non_image_files if source_scan else []
    nested_dirs = source_scan.nested_directories if source_scan else []
    record(
        "Source non-image and nested-file handling",
        all(
            (project_root / Path(row["image_path"])).suffix.lower()
            in IMAGE_EXTENSIONS
            for row in all_rows
        )
        and not any(
            row["image_path"] in non_image_source_files for row in all_rows
        ),
        f"manifest excludes non-image files; source has {len(non_image_source_files)} "
        f"ignored non-image file(s) and {len(nested_dirs)} nested folder(s)",
    )
    record(
        "No nested class directories",
        not nested_dirs,
        "no nested directories found inside class folders"
        if not nested_dirs
        else f"{len(nested_dirs)} nested directory(ies) found",
    )

    report_lines = ["Preprocessing verification", "-" * 27]
    report_lines.extend(
        f"{'PASS' if passed else 'FAIL'}: {name} — {detail}"
        for name, passed, detail in checks
    )
    report_lines.append(f"Overall verification: {'PASS' if not failures else 'FAIL'}")
    report_lines.append("Model training was not performed.")
    return checks, "\n".join(failures) if failures else "\n".join(report_lines)


def main() -> int:
    """Run verification, show results, and update the preprocessing report."""
    parser = argparse.ArgumentParser(
        description="Verify prepared dataset manifests and preprocessing."
    )
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    try:
        checks, result = verify(args.dataset_dir, args.output_dir)
    except (FileNotFoundError, OSError, ValueError, json.JSONDecodeError) as error:
        parser.error(str(error))

    print(result)
    passed = all(check[1] for check in checks)
    report_path = args.output_dir / "preprocessing_report.txt"
    if report_path.is_file():
        existing_report = report_path.read_text(encoding="utf-8")
        if "Preprocessing verification\n" in existing_report:
            existing_report = existing_report.split(
                "Preprocessing verification\n", maxsplit=1
            )[0].rstrip()
        report_path.write_text(
            existing_report + "\n\n" + result + "\n",
            encoding="utf-8",
        )

    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
