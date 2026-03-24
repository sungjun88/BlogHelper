from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    from backend.image_classifier import (  # noqa: E402
        CATEGORY_DEFINITIONS,
        TRAINED_MODEL_PATH,
        classify_image,
        get_trained_embedding_classifier,
        list_uploaded_images,
        train_embedding_classifier,
    )
except ModuleNotFoundError:
    from image_classifier import (  # noqa: E402
        CATEGORY_DEFINITIONS,
        TRAINED_MODEL_PATH,
        classify_image,
        get_trained_embedding_classifier,
        list_uploaded_images,
        train_embedding_classifier,
    )


DEFAULT_IMAGE_DIR = REPO_ROOT / "backend" / "uploads"
DEFAULT_LABELS_FILE = REPO_ROOT / "backend" / "tuning_labels.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate and train the BlogHelper image classifier.",
    )
    parser.add_argument(
        "--image-dir",
        type=Path,
        default=DEFAULT_IMAGE_DIR,
        help="Directory containing training or evaluation images.",
    )
    parser.add_argument(
        "--labels-file",
        type=Path,
        default=DEFAULT_LABELS_FILE,
        help="JSON file containing the ground-truth labels.",
    )
    parser.add_argument(
        "--init-labels",
        action="store_true",
        help="Create or refresh the label template from the image directory.",
    )
    parser.add_argument(
        "--train",
        action="store_true",
        help="Train the CLIP embedding prototype classifier from labeled images.",
    )
    parser.add_argument(
        "--show-model",
        action="store_true",
        help="Print the currently saved trained model summary.",
    )
    return parser.parse_args()


def load_labels(labels_file: Path) -> dict[str, dict[str, str]]:
    if not labels_file.exists():
        return {}

    data = json.loads(labels_file.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("Label file must be a JSON object keyed by filename.")
    return data


def save_labels_template(image_dir: Path, labels_file: Path) -> None:
    existing = load_labels(labels_file)
    template: dict[str, dict[str, str]] = {}

    for image_path in list_uploaded_images(image_dir):
        current = existing.get(image_path.name, {})
        template[image_path.name] = {
            "label": current.get("label", ""),
            "notes": current.get("notes", ""),
        }

    labels_file.write_text(
        json.dumps(template, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def collect_labeled_paths(
    image_dir: Path,
    labels_file: Path,
) -> tuple[list[tuple[Path, str]], list[str], list[tuple[str, str]]]:
    labels = load_labels(labels_file)
    valid_categories = {category.key for category in CATEGORY_DEFINITIONS}

    labeled_paths: list[tuple[Path, str]] = []
    skipped_files: list[str] = []
    invalid_labels: list[tuple[str, str]] = []

    for image_path in list_uploaded_images(image_dir):
        entry = labels.get(image_path.name)
        if not entry or not entry.get("label"):
            skipped_files.append(image_path.name)
            continue

        label = entry["label"]
        if label not in valid_categories:
            invalid_labels.append((image_path.name, label))
            continue

        labeled_paths.append((image_path, label))

    return labeled_paths, skipped_files, invalid_labels


def evaluate_predictions(image_dir: Path, labels_file: Path) -> int:
    labeled_paths, skipped_files, invalid_labels = collect_labeled_paths(image_dir, labels_file)
    enabled_categories = [category.key for category in CATEGORY_DEFINITIONS if category.enabled]

    if invalid_labels:
        print("Invalid labels were found:")
        for filename, label in invalid_labels:
            print(f"- {filename}: {label}")
        print(f"Allowed categories: {', '.join(sorted(category.key for category in CATEGORY_DEFINITIONS))}")
        return 1

    if not labeled_paths:
        print("No labeled images were found.")
        print("Run this first to generate a label template:")
        print(r".venv\Scripts\python.exe backend\tune_classifier.py --init-labels")
        return 1

    total = 0
    correct = 0
    per_label_total = Counter()
    per_label_correct = Counter()
    confusion: dict[str, Counter[str]] = defaultdict(Counter)
    misclassified = []
    classifier_counts = Counter()

    for image_path, expected_label in labeled_paths:
        result = classify_image(image_path)
        predicted_label = result["category"]
        classifier_counts[result["features"].get("classifier", "unknown")] += 1

        total += 1
        per_label_total[expected_label] += 1
        confusion[expected_label][predicted_label] += 1

        if predicted_label == expected_label:
            correct += 1
            per_label_correct[expected_label] += 1
            continue

        ranked_scores = sorted(
            result["scores"].items(),
            key=lambda item: item[1],
            reverse=True,
        )
        misclassified.append(
            {
                "filename": image_path.name,
                "expected": expected_label,
                "predicted": predicted_label,
                "classifier": result["features"].get("classifier", "unknown"),
                "confidence": result["confidence"],
                "scores": ranked_scores[:3],
            }
        )

    accuracy = correct / total if total else 0.0

    print("Evaluation summary")
    print(f"- labeled images: {total}")
    print(f"- unlabeled images skipped: {len(skipped_files)}")
    print(f"- accuracy: {accuracy:.2%} ({correct}/{total})")
    print(f"- classifiers used: {dict(classifier_counts)}")
    print("")

    print("Per-category accuracy")
    for category in CATEGORY_DEFINITIONS:
        label_key = category.key
        if per_label_total[label_key] == 0:
            continue
        category_accuracy = per_label_correct[label_key] / per_label_total[label_key]
        suffix = " [disabled]" if label_key not in enabled_categories else ""
        print(
            f"- {label_key}: {category_accuracy:.2%} "
            f"({per_label_correct[label_key]}/{per_label_total[label_key]}){suffix}"
        )
    print("")

    print("Confusion")
    for expected_label in sorted(confusion):
        row = ", ".join(
            f"{predicted}:{count}"
            for predicted, count in confusion[expected_label].most_common()
        )
        print(f"- {expected_label} -> {row}")
    print("")

    if misclassified:
        print("Misclassified samples")
        for item in misclassified:
            score_text = ", ".join(
                f"{category}:{score:.4f}"
                for category, score in item["scores"]
            )
            print(
                f"- {item['filename']} | expected={item['expected']} predicted={item['predicted']} "
                f"classifier={item['classifier']} confidence={item['confidence']:.4f}"
            )
            print(f"  top_scores: {score_text}")
    else:
        print("No misclassifications found.")

    return 0


def train_model(image_dir: Path, labels_file: Path) -> int:
    labeled_paths, skipped_files, invalid_labels = collect_labeled_paths(image_dir, labels_file)

    if invalid_labels:
        print("Invalid labels were found:")
        for filename, label in invalid_labels:
            print(f"- {filename}: {label}")
        return 1

    if not labeled_paths:
        print("No labeled images were found.")
        return 1

    label_counts = Counter(label for _, label in labeled_paths)
    low_sample_labels = [label for label, count in label_counts.items() if count < 2]
    if low_sample_labels:
        print("Some labels have fewer than 2 samples. Training may be unstable:")
        for label in low_sample_labels:
            print(f"- {label}: {label_counts[label]}")

    summary = train_embedding_classifier(labeled_paths)

    print("Training completed")
    print(f"- model path: {summary['model_path']}")
    print(f"- model id: {summary['model_id']}")
    print(f"- labeled images used: {len(labeled_paths)}")
    print(f"- unlabeled images skipped: {len(skipped_files)}")
    print(f"- sample counts: {summary['sample_counts']}")
    print("")
    print("Run the evaluator again to check whether accuracy improved:")
    print(r".venv\Scripts\python.exe backend\tune_classifier.py")
    return 0


def show_model_summary() -> int:
    classifier = get_trained_embedding_classifier()
    if not classifier.is_available():
        print(f"No trained model found at: {TRAINED_MODEL_PATH}")
        return 1

    summary = classifier.summary()
    print("Trained model summary")
    print(f"- model path: {summary['model_path']}")
    print(f"- model id: {summary['model_id']}")
    print(f"- categories: {', '.join(summary['category_keys'])}")
    print(f"- sample counts: {summary['sample_counts']}")
    return 0


def main() -> int:
    args = parse_args()
    image_dir = args.image_dir.resolve()
    labels_file = args.labels_file.resolve()

    if not image_dir.exists():
        print(f"Image directory not found: {image_dir}")
        return 1

    if args.init_labels:
        save_labels_template(image_dir, labels_file)
        print(f"Label template saved to: {labels_file}")
        print("Fill in the label field for each image, then rerun training or evaluation.")
        print("Available categories: exterior, parking, interior, menu, food, thumbnail")
        return 0

    if args.show_model:
        return show_model_summary()

    if args.train:
        return train_model(image_dir, labels_file)

    return evaluate_predictions(image_dir, labels_file)


if __name__ == "__main__":
    raise SystemExit(main())
