from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


ALLOWED_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
DEFAULT_CLIP_MODEL_ID = os.getenv("BLOGHELPER_CLIP_MODEL_ID", "openai/clip-vit-base-patch32")
CLASSIFIER_MODE = os.getenv("BLOGHELPER_CLASSIFIER_MODE", "auto").lower()


@dataclass(frozen=True)
class CategoryDefinition:
    key: str
    label: str
    enabled: bool
    prompts: tuple[str, ...]
    notes: str = ""


CATEGORY_DEFINITIONS: tuple[CategoryDefinition, ...] = (
    CategoryDefinition(
        key="thumbnail",
        label="썸네일용",
        enabled=False,
        prompts=(
            "a dramatic hero shot of a restaurant dish for a blog thumbnail",
            "a visually striking close-up food photo for a cover image",
            "a main featured dish photo with strong visual appeal",
        ),
        notes="기본값은 비활성화되어 있으며, 필요할 때만 활성화하세요.",
    ),
    CategoryDefinition(
        key="exterior",
        label="외부전경",
        enabled=True,
        prompts=(
            "the exterior of a restaurant building",
            "a storefront photo of a restaurant entrance",
            "an outside view of a restaurant facade",
        ),
    ),
    CategoryDefinition(
        key="parking",
        label="주차장",
        enabled=True,
        prompts=(
            "a restaurant parking lot",
            "cars parked near a restaurant building",
            "a parking area outside a restaurant",
        ),
    ),
    CategoryDefinition(
        key="interior",
        label="내부",
        enabled=True,
        prompts=(
            "the interior of a restaurant dining space",
            "tables and seats inside a restaurant",
            "an indoor photo of a restaurant interior",
        ),
    ),
    CategoryDefinition(
        key="menu",
        label="메뉴",
        enabled=True,
        prompts=(
            "a restaurant menu board",
            "a menu page with food prices and text",
            "a close-up photo of a printed restaurant menu",
        ),
    ),
    CategoryDefinition(
        key="food",
        label="음식",
        enabled=True,
        prompts=(
            "a plated dish served at a restaurant",
            "a close-up photo of restaurant food",
            "food on a table in a restaurant",
        ),
    ),
)


class LocalCLIPUnavailable(RuntimeError):
    pass


def _positive(value: float) -> float:
    return max(value, 0.0)


def get_enabled_categories() -> list[CategoryDefinition]:
    return [category for category in CATEGORY_DEFINITIONS if category.enabled]


def list_uploaded_images(upload_dir: Path) -> list[Path]:
    return sorted(
        path
        for path in upload_dir.iterdir()
        if path.is_file() and path.suffix.lower() in ALLOWED_IMAGE_EXTENSIONS
    )


def get_category_metadata() -> list[dict[str, str | bool]]:
    clip_status = get_classifier_status()
    return [
        {
            "key": category.key,
            "label": category.label,
            "enabled": category.enabled,
            "notes": category.notes,
            "classifier_mode": clip_status["mode"],
        }
        for category in CATEGORY_DEFINITIONS
    ]


class LocalCLIPClassifier:
    def __init__(self, model_id: str = DEFAULT_CLIP_MODEL_ID):
        self.model_id = model_id
        self._model = None
        self._processor = None
        self._torch = None
        self._device = None
        self._prompt_keys: list[str] = []
        self._text_features = None

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return

        try:
            import torch
            from transformers import CLIPModel, CLIPProcessor
        except ImportError as exc:
            raise LocalCLIPUnavailable(
                "로컬 CLIP 분류를 쓰려면 torch와 transformers 설치가 필요합니다."
            ) from exc

        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = CLIPModel.from_pretrained(self.model_id)
        processor = CLIPProcessor.from_pretrained(self.model_id)
        model.to(device)
        model.eval()

        self._torch = torch
        self._device = device
        self._model = model
        self._processor = processor
        self._build_text_features()

    def _build_text_features(self) -> None:
        categories = get_enabled_categories()
        prompts = [
            prompt
            for category in categories
            for prompt in category.prompts
        ]
        prompt_keys = [
            category.key
            for category in categories
            for _prompt in category.prompts
        ]

        inputs = self._processor(
            text=prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
        )
        inputs = {key: value.to(self._device) for key, value in inputs.items()}

        with self._torch.no_grad():
            text_features = self._model.get_text_features(**inputs)
            text_features = text_features / text_features.norm(p=2, dim=-1, keepdim=True)

        self._prompt_keys = prompt_keys
        self._text_features = text_features

    def classify(self, image_path: Path) -> dict:
        self._ensure_loaded()

        image = Image.open(image_path).convert("RGB")
        inputs = self._processor(images=image, return_tensors="pt")
        inputs = {key: value.to(self._device) for key, value in inputs.items()}

        with self._torch.no_grad():
            image_features = self._model.get_image_features(**inputs)
            image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)
            prompt_scores = (image_features @ self._text_features.T).squeeze(0)

        grouped_scores: dict[str, list[float]] = {}
        for key, score in zip(self._prompt_keys, prompt_scores.tolist()):
            grouped_scores.setdefault(key, []).append(score)

        averaged_scores = {
            key: float(np.mean(values))
            for key, values in grouped_scores.items()
        }
        predicted_category = max(averaged_scores, key=averaged_scores.get)

        score_values = np.array(list(averaged_scores.values()), dtype=np.float32)
        normalized_scores = _softmax(score_values)
        category_keys = list(averaged_scores.keys())
        probability_scores = {
            key: round(float(score), 4)
            for key, score in zip(category_keys, normalized_scores)
        }

        category_lookup = {category.key: category for category in CATEGORY_DEFINITIONS}

        reasons = [
            f"로컬 CLIP 모델({self.model_id})이 이 이미지를 '{category_lookup[predicted_category].label}' 문장과 가장 가깝게 판단했습니다.",
            f"장치: {self._device}",
        ]

        return {
            "filename": image_path.name,
            "category": predicted_category,
            "category_label": category_lookup[predicted_category].label,
            "confidence": probability_scores[predicted_category],
            "scores": probability_scores,
            "reasons": reasons,
            "features": {
                "classifier": "local_clip",
                "model_id": self.model_id,
                "device": self._device,
            },
        }


def _softmax(values: np.ndarray) -> np.ndarray:
    shifted = values - np.max(values)
    exp = np.exp(shifted)
    return exp / exp.sum()


def _extract_image_features(image_path: Path) -> dict[str, float]:
    image = Image.open(image_path).convert("RGB").resize((256, 256))
    image_array = np.asarray(image, dtype=np.uint8)
    normalized = image_array.astype(np.float32) / 255.0

    brightness = float(normalized.mean())
    saturation = float((normalized.max(axis=2) - normalized.min(axis=2)).mean())
    red = normalized[:, :, 0]
    green = normalized[:, :, 1]
    blue = normalized[:, :, 2]

    warm_ratio = float(((red > green + 0.03) & (green > blue - 0.02)).mean())
    blue_ratio = float(((blue > red + 0.05) & (blue > green + 0.03)).mean())
    green_ratio = float(((green > red + 0.02) & (green > blue + 0.02)).mean())
    white_ratio = float((normalized.min(axis=2) > 0.82).mean())
    dark_ratio = float((normalized.max(axis=2) < 0.25).mean())

    grayscale = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(grayscale, 80, 180)
    edge_density = float((edges > 0).mean())

    hough_lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=35,
        minLineLength=20,
        maxLineGap=6,
    )
    line_count = 0 if hough_lines is None else len(hough_lines)
    line_density = min(line_count / 120.0, 1.0)

    adaptive = cv2.adaptiveThreshold(
        grayscale,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        15,
        8,
    )
    contours, _ = cv2.findContours(adaptive, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    text_like_contours = 0
    text_like_area = 0
    for contour in contours:
        _, _, width, height = cv2.boundingRect(contour)
        area = width * height
        if area < 20 or area > 3000:
            continue
        ratio = width / max(height, 1)
        if 0.2 <= ratio <= 12:
            text_like_contours += 1
            text_like_area += area

    text_block_density = min(text_like_contours / 180.0, 1.0)
    text_area_ratio = min(text_like_area / float(256 * 256), 1.0)

    return {
        "brightness": brightness,
        "saturation": saturation,
        "warm_ratio": warm_ratio,
        "blue_ratio": blue_ratio,
        "green_ratio": green_ratio,
        "white_ratio": white_ratio,
        "dark_ratio": dark_ratio,
        "edge_density": edge_density,
        "line_density": line_density,
        "text_block_density": text_block_density,
        "text_area_ratio": text_area_ratio,
    }


def _heuristic_scores(features: dict[str, float]) -> dict[str, float]:
    return {
        "exterior": (
            1.4 * features["blue_ratio"]
            + 1.1 * features["green_ratio"]
            + 0.9 * features["line_density"]
            + 0.7 * features["edge_density"] * 4
        ),
        "parking": (
            1.7 * features["line_density"]
            + 0.9 * features["white_ratio"]
            + 0.8 * features["brightness"]
            + 0.6 * features["edge_density"] * 4
        ),
        "interior": (
            0.9 * features["line_density"]
            + 0.9 * features["brightness"]
            + 0.8 * features["warm_ratio"]
            + 0.5 * features["white_ratio"]
        ),
        "menu": (
            2.2 * features["text_block_density"]
            + 1.7 * features["text_area_ratio"]
            + 1.0 * features["line_density"]
            + 0.8 * features["white_ratio"]
        ),
        "food": (
            2.1 * features["warm_ratio"]
            + 1.8 * features["saturation"]
            + 0.7 * _positive(0.8 - features["brightness"])
            + 0.3 * features["dark_ratio"]
        ),
    }


def _classify_with_heuristics(image_path: Path) -> dict:
    features = _extract_image_features(image_path)
    raw_scores = _heuristic_scores(features)
    predicted_category = max(raw_scores, key=raw_scores.get)
    probabilities = _softmax(np.array(list(raw_scores.values()), dtype=np.float32))
    category_keys = list(raw_scores.keys())
    scores = {
        key: round(float(probability), 4)
        for key, probability in zip(category_keys, probabilities)
    }
    category_lookup = {category.key: category for category in CATEGORY_DEFINITIONS}

    return {
        "filename": image_path.name,
        "category": predicted_category,
        "category_label": category_lookup[predicted_category].label,
        "confidence": scores[predicted_category],
        "scores": scores,
        "reasons": [
            "로컬 CLIP 모델이 설치되지 않아 임시 규칙 기반 분류기를 사용했습니다.",
            "데스크톱에서 torch와 transformers를 설치하면 CLIP 분류가 자동으로 활성화됩니다.",
        ],
        "features": {
            key: round(value, 4)
            for key, value in features.items()
        } | {"classifier": "heuristic_fallback"},
    }


_LOCAL_CLIP_CLASSIFIER = LocalCLIPClassifier()


def get_classifier_status() -> dict[str, str]:
    if CLASSIFIER_MODE == "heuristic":
        return {"mode": "heuristic", "detail": "환경 변수로 휴리스틱 분류기가 강제되었습니다."}

    try:
        _LOCAL_CLIP_CLASSIFIER._ensure_loaded()
    except LocalCLIPUnavailable as exc:
        return {"mode": "heuristic", "detail": str(exc)}

    return {
        "mode": "local_clip",
        "detail": f"{_LOCAL_CLIP_CLASSIFIER.model_id} on {_LOCAL_CLIP_CLASSIFIER._device}",
    }


def classify_image(image_path: Path) -> dict:
    if CLASSIFIER_MODE == "heuristic":
        return _classify_with_heuristics(image_path)

    try:
        return _LOCAL_CLIP_CLASSIFIER.classify(image_path)
    except LocalCLIPUnavailable:
        return _classify_with_heuristics(image_path)
