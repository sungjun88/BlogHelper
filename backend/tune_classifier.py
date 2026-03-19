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
        classify_image,
        list_uploaded_images,
    )
except ModuleNotFoundError:
    from image_classifier import (  # noqa: E402
        CATEGORY_DEFINITIONS,
        classify_image,
        list_uploaded_images,
    )


DEFAULT_IMAGE_DIR = REPO_ROOT / "backend" / "uploads"
DEFAULT_LABELS_FILE = REPO_ROOT / "backend" / "tuning_labels.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="음식점 사진 분류기 튜닝용 평가 스크립트",
    )
    parser.add_argument(
        "--image-dir",
        type=Path,
        default=DEFAULT_IMAGE_DIR,
        help="평가할 이미지 폴더 경로",
    )
    parser.add_argument(
        "--labels-file",
        type=Path,
        default=DEFAULT_LABELS_FILE,
        help="정답 라벨 JSON 파일 경로",
    )
    parser.add_argument(
        "--init-labels",
        action="store_true",
        help="현재 이미지 기준으로 라벨 파일 초안을 생성하거나 갱신",
    )
    return parser.parse_args()


def load_labels(labels_file: Path) -> dict[str, dict[str, str]]:
    if not labels_file.exists():
        return {}

    data = json.loads(labels_file.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("라벨 파일 최상위 구조는 객체(JSON object)여야 합니다.")
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


def evaluate_predictions(image_dir: Path, labels_file: Path) -> int:
    labels = load_labels(labels_file)
    valid_categories = {category.key for category in CATEGORY_DEFINITIONS}
    enabled_categories = [category.key for category in CATEGORY_DEFINITIONS if category.enabled]

    labeled_paths = []
    skipped_files = []
    invalid_labels = []

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

    if invalid_labels:
        print("유효하지 않은 라벨이 있습니다.")
        for filename, label in invalid_labels:
            print(f"- {filename}: {label}")
        print(f"허용 카테고리: {', '.join(sorted(valid_categories))}")
        return 1

    if not labeled_paths:
        print("평가할 라벨이 없습니다.")
        print("먼저 다음 명령으로 라벨 파일 초안을 만드세요:")
        print("backend\\venv\\Scripts\\python.exe backend\\tune_classifier.py --init-labels")
        return 1

    total = 0
    correct = 0
    per_label_total = Counter()
    per_label_correct = Counter()
    confusion: dict[str, Counter[str]] = defaultdict(Counter)
    misclassified = []

    for image_path, expected_label in labeled_paths:
        result = classify_image(image_path)
        predicted_label = result["category"]

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
                "confidence": result["confidence"],
                "scores": ranked_scores[:3],
            }
        )

    accuracy = correct / total if total else 0.0

    print("평가 결과")
    print(f"- 이미지 수: {total}")
    print(f"- 정확도: {accuracy:.2%} ({correct}/{total})")
    print("")

    print("카테고리별 정확도")
    for category in CATEGORY_DEFINITIONS:
        label_key = category.key
        if per_label_total[label_key] == 0:
            continue
        category_accuracy = per_label_correct[label_key] / per_label_total[label_key]
        suffix = " [비활성]" if label_key not in enabled_categories else ""
        print(
            f"- {label_key}: {category_accuracy:.2%} "
            f"({per_label_correct[label_key]}/{per_label_total[label_key]}){suffix}"
        )
    print("")

    print("혼동표")
    for expected_label in sorted(confusion):
        row = ", ".join(
            f"{predicted}:{count}"
            for predicted, count in confusion[expected_label].most_common()
        )
        print(f"- {expected_label} -> {row}")
    print("")

    if misclassified:
        print("오분류 상세")
        for item in misclassified:
            score_text = ", ".join(
                f"{category}:{score:.4f}"
                for category, score in item["scores"]
            )
            print(
                f"- {item['filename']} | expected={item['expected']} "
                f"predicted={item['predicted']} confidence={item['confidence']:.4f}"
            )
            print(f"  top_scores: {score_text}")
    else:
        print("오분류 없음")

    return 0


def main() -> int:
    args = parse_args()
    image_dir = args.image_dir.resolve()
    labels_file = args.labels_file.resolve()

    if not image_dir.exists():
        print(f"이미지 폴더를 찾을 수 없습니다: {image_dir}")
        return 1

    if args.init_labels:
        save_labels_template(image_dir, labels_file)
        print(f"라벨 파일 초안을 생성했습니다: {labels_file}")
        print("각 파일의 label 값을 정답 카테고리로 채운 뒤 다시 실행하세요.")
        print("예: exterior, parking, interior, menu, food, thumbnail")
        return 0

    return evaluate_predictions(image_dir, labels_file)


if __name__ == "__main__":
    raise SystemExit(main())
