from __future__ import annotations

import random
import shutil
import sys
from pathlib import Path

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]

SOURCE_ROOT = (
    PROJECT_ROOT
    / "training"
    / "new_upload"
)

DESTINATION_ROOT = (
    PROJECT_ROOT
    / "training"
    / "custom_data"
)

SUPPORTED_IMAGES = {
    ".jpg",
    ".jpeg",
    ".png",
    ".webp",
}

TARGET_CLASSES = {
    "ball": 0,
    "car": 1,
    "manner": 2,
}


def load_source_class_names() -> dict[int, str]:
    yaml_files = [
        *SOURCE_ROOT.rglob("data.yaml"),
        *SOURCE_ROOT.rglob("data.yml"),
    ]

    if yaml_files:
        yaml_path = yaml_files[0]

        with yaml_path.open(
            "r",
            encoding="utf-8",
        ) as file:
            data = yaml.safe_load(file)

        names = data.get("names")

        if isinstance(names, list):
            return {
                index: str(name)
                for index, name in enumerate(names)
            }

        if isinstance(names, dict):
            return {
                int(index): str(name)
                for index, name in names.items()
            }

    classes_files = list(
        SOURCE_ROOT.rglob("classes.txt")
    )

    if classes_files:
        lines = classes_files[0].read_text(
            encoding="utf-8",
        ).splitlines()

        return {
            index: line.strip()
            for index, line in enumerate(lines)
            if line.strip()
        }

    print(
        "WARNING: No classes.txt or data.yaml found."
    )

    print(
        "The script assumes:"
        "\n0 = Ball"
        "\n1 = Car"
        "\n2 = Manner"
    )

    return {
        0: "Ball",
        1: "Car",
        2: "Manner",
    }


def create_class_map() -> dict[int, int]:
    source_names = load_source_class_names()

    class_map: dict[int, int] = {}

    print("\nClass mapping:")

    for source_id, source_name in sorted(
        source_names.items()
    ):
        normalized_name = source_name.strip().lower()

        if normalized_name not in TARGET_CLASSES:
            raise ValueError(
                f"Unknown class name: {source_name!r}"
            )

        target_id = TARGET_CLASSES[normalized_name]

        class_map[source_id] = target_id

        print(
            f"  {source_id}: {source_name}"
            f" -> {target_id}"
        )

    required_classes = set(TARGET_CLASSES.values())
    mapped_classes = set(class_map.values())

    if mapped_classes != required_classes:
        raise ValueError(
            "The export must contain Ball, Car and Manner."
        )

    return class_map


def find_label_for_image(
    image_path: Path,
) -> Path | None:
    candidates: list[Path] = []

    if image_path.parent.name.lower() == "images":
        candidates.append(
            image_path.parent.parent
            / "labels"
            / f"{image_path.stem}.txt"
        )

    candidates.append(
        image_path.with_suffix(".txt")
    )

    for candidate in candidates:
        if candidate.exists():
            return candidate

    return None


def find_image_label_pairs() -> list[
    tuple[Path, Path]
]:
    pairs: list[tuple[Path, Path]] = []

    image_paths = sorted(
        path
        for path in SOURCE_ROOT.rglob("*")
        if (
            path.is_file()
            and path.suffix.lower()
            in SUPPORTED_IMAGES
        )
    )

    missing_labels: list[Path] = []

    for image_path in image_paths:
        label_path = find_label_for_image(
            image_path
        )

        if label_path is None:
            missing_labels.append(image_path)
            continue

        pairs.append(
            (
                image_path,
                label_path,
            )
        )

    if missing_labels:
        print("\nImages without label files:")

        for image_path in missing_labels:
            print(f"  {image_path.name}")

        raise RuntimeError(
            f"{len(missing_labels)} images have no label."
        )

    return pairs


def convert_label(
    label_path: Path,
    class_map: dict[int, int],
) -> str:
    converted_lines: list[str] = []

    lines = label_path.read_text(
        encoding="utf-8",
    ).splitlines()

    for line_number, line in enumerate(
        lines,
        start=1,
    ):
        line = line.strip()

        if not line:
            continue

        parts = line.split()

        if len(parts) != 5:
            raise ValueError(
                f"{label_path.name}, line {line_number}: "
                f"expected 5 values, got {len(parts)}"
            )

        try:
            source_class_id = int(parts[0])

            x_center = float(parts[1])
            y_center = float(parts[2])
            width = float(parts[3])
            height = float(parts[4])
        except ValueError as error:
            raise ValueError(
                f"{label_path.name}, line {line_number}: "
                "invalid number"
            ) from error

        if source_class_id not in class_map:
            raise ValueError(
                f"{label_path.name}, line {line_number}: "
                f"unknown class ID {source_class_id}"
            )

        values = [
            x_center,
            y_center,
            width,
            height,
        ]

        if not all(
            0.0 <= value <= 1.0
            for value in values
        ):
            raise ValueError(
                f"{label_path.name}, line {line_number}: "
                "coordinates must be between 0 and 1"
            )

        if width <= 0 or height <= 0:
            raise ValueError(
                f"{label_path.name}, line {line_number}: "
                "width and height must be above 0"
            )

        target_class_id = class_map[
            source_class_id
        ]

        converted_lines.append(
            f"{target_class_id} "
            f"{x_center:.8f} "
            f"{y_center:.8f} "
            f"{width:.8f} "
            f"{height:.8f}"
        )

    return "\n".join(converted_lines) + (
        "\n" if converted_lines else ""
    )


def create_split(
    records: list[tuple[Path, Path]],
) -> dict[str, list[tuple[Path, Path]]]:
    shuffled = records.copy()

    random.Random(42).shuffle(shuffled)

    total = len(shuffled)

    validation_count = max(
        1,
        round(total * 0.10),
    )

    test_count = max(
        1,
        round(total * 0.10),
    )

    train_count = (
        total
        - validation_count
        - test_count
    )

    return {
        "train": shuffled[:train_count],

        "val": shuffled[
            train_count:
            train_count + validation_count
        ],

        "test": shuffled[
            train_count + validation_count:
        ],
    }


def main() -> None:
    if not SOURCE_ROOT.exists():
        print(
            f"ERROR: Source folder not found:"
            f"\n{SOURCE_ROOT}"
        )
        sys.exit(1)

    class_map = create_class_map()

    records = find_image_label_pairs()

    if not records:
        print("ERROR: No image and label pairs found.")
        sys.exit(1)

    print(
        f"\nFound {len(records)} image/label pairs."
    )

    validated_records: list[
        tuple[Path, Path, str]
    ] = []

    errors: list[str] = []

    for image_path, label_path in records:
        try:
            converted_label = convert_label(
                label_path,
                class_map,
            )

            validated_records.append(
                (
                    image_path,
                    label_path,
                    converted_label,
                )
            )
        except ValueError as error:
            errors.append(str(error))

    if errors:
        print("\nInvalid labels found:")

        for error in errors:
            print(f"  {error}")

        print(
            "\nNothing was changed because the "
            "dataset contains invalid labels."
        )

        sys.exit(1)

    simple_records = [
        (image_path, label_path)
        for image_path, label_path, _
        in validated_records
    ]

    converted_by_image = {
        image_path: converted_label
        for image_path, _, converted_label
        in validated_records
    }

    splits = create_split(simple_records)

    temporary_destination = (
        PROJECT_ROOT
        / "training"
        / "custom_data_importing"
    )

    if temporary_destination.exists():
        shutil.rmtree(temporary_destination)

    for split_name in ("train", "val", "test"):
        (
            temporary_destination
            / "images"
            / split_name
        ).mkdir(
            parents=True,
            exist_ok=True,
        )

        (
            temporary_destination
            / "labels"
            / split_name
        ).mkdir(
            parents=True,
            exist_ok=True,
        )

    image_number = 1

    for split_name in ("train", "val", "test"):
        for image_path, _ in splits[split_name]:
            new_stem = f"image_{image_number:06d}"

            destination_image = (
                temporary_destination
                / "images"
                / split_name
                / f"{new_stem}{image_path.suffix.lower()}"
            )

            destination_label = (
                temporary_destination
                / "labels"
                / split_name
                / f"{new_stem}.txt"
            )

            shutil.copy2(
                image_path,
                destination_image,
            )

            destination_label.write_text(
                converted_by_image[image_path],
                encoding="utf-8",
            )

            image_number += 1

    if DESTINATION_ROOT.exists():
        shutil.rmtree(DESTINATION_ROOT)

    temporary_destination.rename(
        DESTINATION_ROOT
    )

    print("\nCorrected dataset imported:")

    for split_name in ("train", "val", "test"):
        print(
            f"  {split_name}: "
            f"{len(splits[split_name])}"
        )

    print(
        f"  total: {len(records)}"
    )

    print(
        f"\nDataset path:\n{DESTINATION_ROOT}"
    )


if __name__ == "__main__":
    main()
