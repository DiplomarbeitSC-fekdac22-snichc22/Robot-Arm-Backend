import json
from pathlib import Path


TRAINING_DIR = Path(__file__).resolve().parent.parent
CUSTOM_DATA_DIR = TRAINING_DIR / "custom_data"
CLASSES_FILE = CUSTOM_DATA_DIR / "classes.txt"
DATA_YAML_FILE = TRAINING_DIR / "data.yaml"

EXPECTED_CLASSES = [
    "Ball",
    "Car",
    "Manner",
]


def load_classes() -> list[str]:
    if not CLASSES_FILE.exists():
        print("classes.txt not found. Using expected classes.")
        return EXPECTED_CLASSES

    classes = [
        line.strip()
        for line in CLASSES_FILE.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    if not classes:
        raise RuntimeError("classes.txt is empty.")

    if classes != EXPECTED_CLASSES:
        raise RuntimeError(
            "Unexpected class order.\n"
            f"Expected: {EXPECTED_CLASSES}\n"
            f"Found:    {classes}\n\n"
            "Label Studio must contain Ball, Car and Manner "
            "in exactly this order."
        )

    return classes


def main() -> None:
    classes = load_classes()

    lines = [
        f"path: {CUSTOM_DATA_DIR.resolve()}",
        "train: train/images",
        "val: validation/images",
        f"nc: {len(classes)}",
        "names:",
    ]

    for class_id, class_name in enumerate(classes):
        lines.append(f"  {class_id}: {json.dumps(class_name)}")

    DATA_YAML_FILE.write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )

    print(f"Created: {DATA_YAML_FILE}")
    print()
    print(DATA_YAML_FILE.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()