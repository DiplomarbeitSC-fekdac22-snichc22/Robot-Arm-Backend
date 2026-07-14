import random
import shutil
from pathlib import Path


TRAINING_DIR = Path(__file__).resolve().parent.parent
CUSTOM_DATA_DIR = TRAINING_DIR / "custom_data"

SOURCE_IMAGES_DIR = CUSTOM_DATA_DIR / "images"
SOURCE_LABELS_DIR = CUSTOM_DATA_DIR / "labels"

TRAIN_IMAGES_DIR = CUSTOM_DATA_DIR / "train" / "images"
TRAIN_LABELS_DIR = CUSTOM_DATA_DIR / "train" / "labels"

VALIDATION_IMAGES_DIR = CUSTOM_DATA_DIR / "validation" / "images"
VALIDATION_LABELS_DIR = CUSTOM_DATA_DIR / "validation" / "labels"

VALIDATION_RATIO = 0.10
RANDOM_SEED = 42

SUPPORTED_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".webp",
}


def recreate_directory(directory: Path) -> None:
    if directory.exists():
        shutil.rmtree(directory)

    directory.mkdir(parents=True, exist_ok=True)


def copy_pair(
    image_path: Path,
    label_path: Path,
    target_images_dir: Path,
    target_labels_dir: Path,
) -> None:
    shutil.copy2(image_path, target_images_dir / image_path.name)
    shutil.copy2(label_path, target_labels_dir / label_path.name)


def main() -> None:
    for directory in [
        TRAIN_IMAGES_DIR,
        TRAIN_LABELS_DIR,
        VALIDATION_IMAGES_DIR,
        VALIDATION_LABELS_DIR,
    ]:
        recreate_directory(directory)

    pairs: list[tuple[Path, Path]] = []

    for image_path in sorted(SOURCE_IMAGES_DIR.iterdir()):
        if not image_path.is_file():
            continue

        if image_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue

        label_path = SOURCE_LABELS_DIR / f"{image_path.stem}.txt"

        if not label_path.exists():
            print(f"Skipping image without label: {image_path.name}")
            continue

        pairs.append((image_path, label_path))

    if not pairs:
        raise RuntimeError("No matching image-label pairs were found.")

    random.seed(RANDOM_SEED)
    random.shuffle(pairs)

    if len(pairs) > 1:
        validation_count = max(
            1,
            round(len(pairs) * VALIDATION_RATIO),
        )
    else:
        validation_count = 0

    validation_pairs = pairs[:validation_count]
    train_pairs = pairs[validation_count:]

    for image_path, label_path in train_pairs:
        copy_pair(
            image_path,
            label_path,
            TRAIN_IMAGES_DIR,
            TRAIN_LABELS_DIR,
        )

    for image_path, label_path in validation_pairs:
        copy_pair(
            image_path,
            label_path,
            VALIDATION_IMAGES_DIR,
            VALIDATION_LABELS_DIR,
        )

    print(f"Total pairs: {len(pairs)}")
    print(f"Training images: {len(train_pairs)}")
    print(f"Validation images: {len(validation_pairs)}")


if __name__ == "__main__":
    main()