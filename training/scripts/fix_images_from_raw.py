import re
import shutil
from pathlib import Path


TRAINING_DIR = Path(__file__).resolve().parent.parent

RAW_IMAGES_DIR = TRAINING_DIR / "raw_images"
CUSTOM_DATA_DIR = TRAINING_DIR / "custom_data"
EXPORT_IMAGES_DIR = CUSTOM_DATA_DIR / "images"
LABELS_DIR = CUSTOM_DATA_DIR / "labels"

SUPPORTED_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".webp",
}


def build_raw_image_index() -> dict[str, Path]:
    index: dict[str, Path] = {}

    for image_path in RAW_IMAGES_DIR.iterdir():
        if not image_path.is_file():
            continue

        if image_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue

        index[image_path.stem] = image_path

        if image_path.stem.isdigit():
            index[str(int(image_path.stem))] = image_path

    return index


def find_raw_image(label_stem: str, image_index: dict[str, Path]) -> Path | None:
    direct_match = image_index.get(label_stem)

    if direct_match:
        return direct_match

    numbers = re.findall(r"\d+", label_stem)

    if not numbers:
        return None

    final_number = numbers[-1]

    normalized_number = str(int(final_number))

    return (
        image_index.get(final_number)
        or image_index.get(normalized_number)
        or image_index.get(final_number.zfill(4))
    )


def main() -> None:
    if not RAW_IMAGES_DIR.exists():
        raise FileNotFoundError(
            f"Raw image folder does not exist: {RAW_IMAGES_DIR}"
        )

    if not LABELS_DIR.exists():
        raise FileNotFoundError(
            f"Label folder does not exist: {LABELS_DIR}"
        )

    EXPORT_IMAGES_DIR.mkdir(parents=True, exist_ok=True)

    image_index = build_raw_image_index()

    copied = 0
    missing: list[str] = []

    for label_path in sorted(LABELS_DIR.glob("*.txt")):
        if label_path.name.lower() == "classes.txt":
            continue

        source_image = find_raw_image(label_path.stem, image_index)

        if source_image is None:
            missing.append(label_path.name)
            continue

        destination = EXPORT_IMAGES_DIR / (
            label_path.stem + source_image.suffix.lower()
        )

        shutil.copy2(source_image, destination)
        copied += 1

    print(f"Copied images: {copied}")
    print(f"Missing images: {len(missing)}")

    if missing:
        print("\nLabels without matching raw images:")

        for filename in missing:
            print(f"  - {filename}")


if __name__ == "__main__":
    main()