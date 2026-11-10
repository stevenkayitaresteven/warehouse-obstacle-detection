"""
Convert the LOCO COCO-format annotations into YOLO detection format.

Expected input:
    datasets/raw/LOCO/
    datasets/raw/annotations/

Output:
    datasets/processed/yolo/
        images/train/
        images/val/
        labels/train/
        labels/val/

The script does not download data or model weights.
"""

import argparse
import json
import shutil
from collections import defaultdict
from pathlib import Path, PurePosixPath


# Original LOCO COCO category IDs mapped to contiguous YOLO IDs.
CATEGORY_MAP = {
    3: 0,  # small_load_carrier
    5: 1,  # forklift
    7: 2,  # pallet
    10: 3, # stillage
    11: 4, # pallet_truck
}

CLASS_NAMES = {
    0: "small_load_carrier",
    1: "forklift",
    2: "pallet",
    3: "stillage",
    4: "pallet_truck",
}

TRAIN_ANNOTATIONS = [
    "loco-sub2-v1-train.json",
    "loco-sub3-v1-train.json",
    "loco-sub5-v1-train.json",
]

VAL_ANNOTATIONS = [
    "loco-sub1-v1-val.json",
    "loco-sub4-v1-val.json",
]


def build_image_index(image_root):
    """Index image files by basename for COCO filename resolution."""
    extensions = {".jpg", ".jpeg", ".png"}
    index = defaultdict(list)

    for path in image_root.rglob("*"):
        if path.is_file() and path.suffix.lower() in extensions:
            index[path.name].append(path)

    return index


def resolve_image(file_name, image_root, image_index):
    """Resolve a COCO file_name to an actual image path."""
    normalized = str(file_name).replace("\\", "/")
    direct_path = image_root / normalized

    if direct_path.is_file():
        return direct_path

    basename = PurePosixPath(normalized).name
    candidates = image_index.get(basename, [])

    if len(candidates) == 1:
        return candidates[0]

    if len(candidates) > 1:
        suffix = PurePosixPath(normalized).parts

        matching = [
            candidate for candidate in candidates
            if len(candidate.parts) >= len(suffix)
            and tuple(candidate.parts[-len(suffix):]) == suffix
        ]

        if len(matching) == 1:
            return matching[0]

        raise RuntimeError(
            f"Ambiguous image filename: {file_name}. "
            f"Candidates: {candidates}"
        )

    raise FileNotFoundError(
        f"Image not found: {file_name} under {image_root}"
    )


def load_annotations(annotation_dir, filenames):
    """Load images and annotations from the specified COCO JSON files."""
    all_images = []
    all_annotations = []
    seen_image_ids = set()

    for filename in filenames:
        json_path = annotation_dir / filename

        if not json_path.is_file():
            raise FileNotFoundError(json_path)

        with json_path.open("r", encoding="utf-8") as file:
            data = json.load(file)

        categories = {
            category["id"]: category["name"]
            for category in data.get("categories", [])
        }

        for category_id in CATEGORY_MAP:
            if category_id not in categories:
                raise ValueError(
                    f"Category ID {category_id} missing in {filename}"
                )

        for image in data.get("images", []):
            image_id = image["id"]

            if image_id in seen_image_ids:
                raise ValueError(
                    f"Duplicate image ID {image_id} in {filename}"
                )

            seen_image_ids.add(image_id)
            all_images.append(image)

        all_annotations.extend(data.get("annotations", []))

    return all_images, all_annotations


def convert_split(
    split_name,
    annotation_files,
    image_root,
    annotation_dir,
    output_root,
    image_index,
):
    """Convert one dataset split and copy its corresponding images."""
    images, annotations = load_annotations(
        annotation_dir, annotation_files
    )

    annotations_by_image = defaultdict(list)

    for annotation in annotations:
        annotations_by_image[annotation["image_id"]].append(annotation)

    image_output = output_root / "images" / split_name
    label_output = output_root / "labels" / split_name

    image_output.mkdir(parents=True, exist_ok=True)
    label_output.mkdir(parents=True, exist_ok=True)

    copied = 0
    annotation_count = 0
    used_names = set()

    for image in images:
        image_id = image["id"]
        width = int(image["width"])
        height = int(image["height"])

        if width <= 0 or height <= 0:
            raise ValueError(f"Invalid image dimensions: {image}")

        source = resolve_image(
            image["file_name"], image_root, image_index
        )

        output_name = source.name

        if output_name in used_names:
            raise ValueError(
                f"Duplicate output image filename: {output_name}"
            )

        used_names.add(output_name)

        destination = image_output / output_name
        label_path = label_output / f"{source.stem}.txt"

        if destination.exists() or label_path.exists():
            raise FileExistsError(
                f"Output already exists: {output_name}. "
                "Use a new output directory or --overwrite."
            )

        lines = []

        for annotation in annotations_by_image.get(image_id, []):
            category_id = annotation["category_id"]

            if category_id not in CATEGORY_MAP:
                continue

            x, y, box_width, box_height = map(
                float, annotation["bbox"]
            )

            if box_width <= 0 or box_height <= 0:
                raise ValueError(
                    f"Invalid bounding box in image {image_id}"
                )

            if (
                x < 0 or y < 0
                or x + box_width > width + 1e-6
                or y + box_height > height + 1e-6
            ):
                raise ValueError(
                    f"Out-of-bounds bounding box in image {image_id}: "
                    f"{annotation['bbox']}"
                )

            x_center = (x + box_width / 2) / width
            y_center = (y + box_height / 2) / height
            normalized_width = box_width / width
            normalized_height = box_height / height

            class_id = CATEGORY_MAP[category_id]

            lines.append(
                f"{class_id} "
                f"{x_center:.6f} "
                f"{y_center:.6f} "
                f"{normalized_width:.6f} "
                f"{normalized_height:.6f}"
            )

        shutil.copy2(source, destination)
        label_path.write_text(
            "\n".join(lines) + ("\n" if lines else ""),
            encoding="utf-8",
        )

        copied += 1
        annotation_count += len(lines)

    print(f"\n{split_name.upper()}")
    print(f"Images: {copied:,}")
    print(f"Annotations: {annotation_count:,}")

    return copied, annotation_count


def main():
    parser = argparse.ArgumentParser(
        description="Convert LOCO COCO annotations to YOLO format."
    )

    parser.add_argument("--images", required=True)
    parser.add_argument("--annotations", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete the output directory before conversion.",
    )

    args = parser.parse_args()

    image_root = Path(args.images)
    annotation_dir = Path(args.annotations)
    output_root = Path(args.output)

    for directory in [image_root, annotation_dir]:
        if not directory.is_dir():
            raise NotADirectoryError(directory)

    if output_root.exists():
        if not args.overwrite:
            raise FileExistsError(
                f"Output directory already exists: {output_root}. "
                "Use --overwrite to replace it."
            )
        shutil.rmtree(output_root)

    image_index = build_image_index(image_root)

    print(f"Indexed original images: {sum(map(len, image_index.values()))}")

    train_images, train_annotations = convert_split(
        "train",
        TRAIN_ANNOTATIONS,
        image_root,
        annotation_dir,
        output_root,
        image_index,
    )

    val_images, val_annotations = convert_split(
        "val",
        VAL_ANNOTATIONS,
        image_root,
        annotation_dir,
        output_root,
        image_index,
    )

    print("\nCONVERSION SUMMARY")
    print(f"Training images: {train_images:,}")
    print(f"Validation images: {val_images:,}")
    print(f"Training annotations: {train_annotations:,}")
    print(f"Validation annotations: {val_annotations:,}")
    print(f"Total images: {train_images + val_images:,}")
    print(
        f"Total annotations: "
        f"{train_annotations + val_annotations:,}"
    )
    print(f"Output: {output_root}")


if __name__ == "__main__":
    main()
