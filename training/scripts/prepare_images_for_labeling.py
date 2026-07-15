from __future__ import annotations

import re
import shutil
import sys
import tempfile
import zipfile
from datetime import datetime
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]

ZIP_PATH = Path.home() / "Downloads" / "data.zip"

EXISTING_DATASET = (
    PROJECT_ROOT
    / "training"
    / "custom_data"
    / "images"
)

RAW_IMAGE_ROOT = (
    PROJECT_ROOT
    / "training"
    / "raw_images"
)

SUPPORTED_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".webp",
}


def find_highest_existing_number() -> int:
    highest_number = 0

    if not EXISTING_DATASET.exists():
        return highest_number

    for image_path in EXISTING_DATASET.rglob("*"):
        if not image_path.is_file():
            continue

        if image_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue

        numbers = re.findall(
            r"\d+",
            image_path.stem,
        )

        if not numbers:
            continue

        highest_number = max(
            highest_number,
            int(numbers[-1]),
        )

    return highest_number


def find_images(folder: Path) -> list[Path]:
    return sorted(
        path
        for path in folder.rglob("*")
        if (
            path.is_file()
            and path.suffix.lower()
            in SUPPORTED_EXTENSIONS
        )
    )


def main() -> None:
    if not ZIP_PATH.exists():
        print(
            f"ERROR: ZIP file not found:\n{ZIP_PATH}"
        )
        sys.exit(1)

    timestamp = datetime.now().strftime(
        "%Y%m%d_%H%M%S"
    )

    output_folder = (
        RAW_IMAGE_ROOT
        / f"to_label_{timestamp}"
    )

    output_folder.mkdir(
        parents=True,
        exist_ok=False,
    )

    highest_number = find_highest_existing_number()
    next_number = highest_number + 1

    with tempfile.TemporaryDirectory() as temp_directory:
        temp_path = Path(temp_directory)

        with zipfile.ZipFile(ZIP_PATH, "r") as archive:
            archive.extractall(temp_path)

        source_images = find_images(temp_path)

        if not source_images:
            print("ERROR: No images found in the ZIP.")
            shutil.rmtree(output_folder)
            sys.exit(1)

        print(
            f"Found {len(source_images)} images."
        )

        print(
            f"Current highest dataset number: "
            f"{highest_number}"
        )

        print(
            f"New numbering starts at: "
            f"{next_number}"
        )

        for source_image in source_images:
            new_name = (
                f"{next_number:04d}"
                f"{source_image.suffix.lower()}"
            )

            destination = (
                output_folder
                / new_name
            )

            shutil.copy2(
                source_image,
                destination,
            )

            next_number += 1

    prepared_images = find_images(output_folder)

    print()
    print("=" * 60)
    print("PREPARATION COMPLETE")
    print("=" * 60)

    print(
        f"Images prepared: {len(prepared_images)}"
    )

    print(
        f"First image: {prepared_images[0].name}"
    )

    print(
        f"Last image:  {prepared_images[-1].name}"
    )

    print(
        f"Output folder:\n{output_folder}"
    )

    if len(prepared_images) != 95:
        print()
        print(
            "WARNING: You expected 95 images, "
            f"but {len(prepared_images)} were found."
        )


if __name__ == "__main__":
    main()
