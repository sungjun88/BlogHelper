from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image


ALLOWED_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
DEFAULT_CLIP_MODEL_ID = os.getenv("BLOGHELPER_CLIP_MODEL_ID", "openai/clip-vit-base-patch32")
CLASSIFIER_MODE = os.getenv("BLOGHELPER_CLASSIFIER_MODE", "auto").lower()
TRAINED_MODEL_PATH = Path(
    os.getenv(
        "BLOGHELPER_TRAINED_MODEL_PATH",
        Path(__file__).resolve().parent / "trained_classifier.npz",
    )
)


@dataclass(frozen=True)
class CategoryDefinition:
    key: str
    label: str
    enabled: bool
    prompts: tuple[str, ...]
    notes: str = ""
    trainable: bool = True


CATEGORY_DEFINITIONS: tuple[CategoryDefinition, ...] = (
    CategoryDefinition(
        key="thumbnail",
        label="Thumbnail",
        enabled=False,
        prompts=(
            "a dramatic hero shot of a restaurant dish for a blog thumbnail",
            "a visually striking close-up food photo for a cover image",
            "a main featured dish photo with strong visual appeal",
        ),
        notes="Disabled by default because it overlaps with food unless you curate labels carefully.",
    ),
    CategoryDefinition(
        key="exterior",
        label="Exterior",
        enabled=True,
        prompts=(
            "the exterior of a restaurant building",
            "a storefront photo of a restaurant entrance",
            "an outside view of a restaurant facade",
        ),
    ),
    CategoryDefinition(
        key="parking",
        label="Parking",
        enabled=True,
        prompts=(
            "a restaurant parking lot",
            "cars parked near a restaurant building",
            "a parking area outside a restaurant",
        ),
    ),
    CategoryDefinition(
        key="interior",
        label="Interior",
        enabled=True,
        prompts=(
            "the interior of a restaurant dining space",
            "tables and seats inside a restaurant",
            "an indoor photo of a restaurant interior",
        ),
    ),
    CategoryDefinition(
        key="menu",
        label="Menu",
        enabled=True,
        prompts=(
            "a restaurant menu board",
            "a menu page with food prices and text",
            "a close-up photo of a printed restaurant menu",
        ),
    ),
    CategoryDefinition(
        key="food",
        label="Food",
        enabled=True,
        prompts=(
            "a plated dish served at a restaurant",
            "a close-up photo of restaurant food",
            "food on a table in a restaurant",
        ),
    ),
    CategoryDefinition(
        key="etc",
        label="ETC",
        enabled=True,
        prompts=(),
        notes="Use for photos you want to keep out of training and evaluation.",
        trainable=False,
    ),
)


class LocalCLIPUnavailable(RuntimeError):
    pass


def _positive(value: float) -> float:
    return max(value, 0.0)


def _softmax(values: np.ndarray) -> np.ndarray:
    shifted = values - np.max(values)
    exp = np.exp(shifted)
    return exp / exp.sum()


def _extract_clip_embedding(output: Any):
    if hasattr(output, "pooler_output") and output.pooler_output is not None:
        return output.pooler_output
    if isinstance(output, tuple) and output:
        return output[0]
    return output


def get_enabled_categories() -> list[CategoryDefinition]:
    return [category for category in CATEGORY_DEFINITIONS if category.enabled]


def get_trainable_categories() -> list[CategoryDefinition]:
    return [category for category in CATEGORY_DEFINITIONS if category.trainable]


def list_uploaded_images(upload_dir: Path) -> list[Path]:
    return sorted(
        path
        for path in upload_dir.iterdir()
        if path.is_file() and path.suffix.lower() in ALLOWED_IMAGE_EXTENSIONS
    )


def get_classifier_status() -> dict[str, str]:
    if CLASSIFIER_MODE == "heuristic":
        return {"mode": "heuristic", "detail": "Forced heuristic mode by environment variable."}

    if _TRAINED_EMBEDDING_CLASSIFIER.is_available():
        return {
            "mode": "trained_clip",
            "detail": f"Trained prototype classifier loaded from {TRAINED_MODEL_PATH.name}",
        }

    try:
        _LOCAL_CLIP_CLASSIFIER.ensure_loaded()
    except Exception as exc:  # pragma: no cover - depends on local model availability
        return {"mode": "heuristic", "detail": str(exc)}

    return {
        "mode": "local_clip",
        "detail": f"{_LOCAL_CLIP_CLASSIFIER.model_id} on {_LOCAL_CLIP_CLASSIFIER.device}",
    }


def get_category_metadata() -> list[dict[str, str | bool]]:
    classifier_status = get_classifier_status()
    return [
        {
            "key": category.key,
            "label": category.label,
            "enabled": category.enabled,
            "notes": category.notes,
            "trainable": category.trainable,
            "classifier_mode": classifier_status["mode"],
        }
        for category in CATEGORY_DEFINITIONS
    ]


class LocalCLIPEncoder:
    def __init__(self, model_id: str = DEFAULT_CLIP_MODEL_ID):
        self.model_id = model_id
        self._model = None
        self._processor = None
        self._torch = None
        self._device = None

    @property
    def device(self) -> str | None:
        return self._device

    def ensure_loaded(self) -> None:
        if self._model is not None:
            return

        try:
            import torch
            from transformers import CLIPModel, CLIPProcessor
        except ImportError as exc:  # pragma: no cover - local dependency dependent
            raise LocalCLIPUnavailable(
                "Local CLIP requires torch and transformers to be installed."
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

    def encode_texts(self, texts: list[str]) -> np.ndarray:
        self.ensure_loaded()
        inputs = self._processor(
            text=texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
        )
        inputs = {key: value.to(self._device) for key, value in inputs.items()}

        with self._torch.no_grad():
            text_features = self._model.get_text_features(**inputs)
            text_features = _extract_clip_embedding(text_features)
            text_features = text_features / text_features.norm(p=2, dim=-1, keepdim=True)

        return text_features.detach().cpu().numpy().astype(np.float32)

    def encode_image(self, image_path: Path) -> np.ndarray:
        self.ensure_loaded()
        image = Image.open(image_path).convert("RGB")
        inputs = self._processor(images=image, return_tensors="pt")
        inputs = {key: value.to(self._device) for key, value in inputs.items()}

        with self._torch.no_grad():
            image_features = self._model.get_image_features(**inputs)
            image_features = _extract_clip_embedding(image_features)
            image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)

        return image_features.detach().cpu().numpy()[0].astype(np.float32)


class LocalCLIPClassifier:
    def __init__(self, encoder: LocalCLIPEncoder):
        self.encoder = encoder
        self._prompt_keys: list[str] = []
        self._text_features: np.ndarray | None = None

    @property
    def model_id(self) -> str:
        return self.encoder.model_id

    @property
    def device(self) -> str | None:
        return self.encoder.device

    def ensure_loaded(self) -> None:
        self.encoder.ensure_loaded()
        if self._text_features is None:
            self._build_text_features()

    def _build_text_features(self) -> None:
        categories = [category for category in get_enabled_categories() if category.prompts]
        prompts = [prompt for category in categories for prompt in category.prompts]
        prompt_keys = [category.key for category in categories for _ in category.prompts]
        self._text_features = self.encoder.encode_texts(prompts)
        self._prompt_keys = prompt_keys

    def classify(self, image_path: Path) -> dict:
        self.ensure_loaded()
        image_features = self.encoder.encode_image(image_path)
        prompt_scores = np.matmul(self._text_features, image_features)

        grouped_scores: dict[str, list[float]] = {}
        for key, score in zip(self._prompt_keys, prompt_scores.tolist()):
            grouped_scores.setdefault(key, []).append(score)

        averaged_scores = {
            key: float(np.mean(values))
            for key, values in grouped_scores.items()
        }
        predicted_category = max(averaged_scores, key=averaged_scores.get)
        category_lookup = {category.key: category for category in CATEGORY_DEFINITIONS}
        probability_scores = _normalize_score_dict(averaged_scores)

        return {
            "filename": image_path.name,
            "category": predicted_category,
            "category_label": category_lookup[predicted_category].label,
            "confidence": probability_scores[predicted_category],
            "scores": probability_scores,
            "reasons": [
                f"Zero-shot CLIP matched the image closest to {category_lookup[predicted_category].label}.",
                f"Device: {self.device}",
            ],
            "features": {
                "classifier": "local_clip",
                "model_id": self.model_id,
                "device": self.device,
            },
        }


class TrainedEmbeddingClassifier:
    def __init__(self, encoder: LocalCLIPEncoder, model_path: Path = TRAINED_MODEL_PATH):
        self.encoder = encoder
        self.model_path = model_path
        self._loaded = False
        self._category_keys: list[str] = []
        self._category_labels: dict[str, str] = {}
        self._centroids: np.ndarray | None = None
        self._model_id: str | None = None
        self._sample_counts: dict[str, int] = {}

    def is_available(self) -> bool:
        if not self.model_path.exists():
            return False
        self._ensure_loaded()
        return self._centroids is not None and len(self._category_keys) > 0

    def _ensure_loaded(self) -> None:
        if self._loaded:
            return
        self._loaded = True
        if not self.model_path.exists():
            return

        data = np.load(self.model_path, allow_pickle=False)
        raw_keys = data["category_keys"].tolist()
        self._category_keys = [str(key) for key in raw_keys]
        self._centroids = data["centroids"].astype(np.float32)
        self._model_id = str(data["model_id"].tolist())
        counts = data["sample_counts"].astype(np.int32).tolist()
        self._sample_counts = {
            key: int(count)
            for key, count in zip(self._category_keys, counts)
        }
        self._category_labels = {
            category.key: category.label
            for category in CATEGORY_DEFINITIONS
        }

    def classify(self, image_path: Path) -> dict:
        self._ensure_loaded()
        if self._centroids is None or not self._category_keys:
            raise LocalCLIPUnavailable("No trained embedding classifier is available.")

        image_embedding = self.encoder.encode_image(image_path)
        scores = np.matmul(self._centroids, image_embedding)
        raw_scores = {
            key: float(score)
            for key, score in zip(self._category_keys, scores.tolist())
        }
        normalized_scores = _normalize_score_dict(raw_scores)
        predicted_category = max(normalized_scores, key=normalized_scores.get)

        return {
            "filename": image_path.name,
            "category": predicted_category,
            "category_label": self._category_labels[predicted_category],
            "confidence": normalized_scores[predicted_category],
            "scores": normalized_scores,
            "reasons": [
                f"Trained CLIP embedding classifier selected {self._category_labels[predicted_category]}.",
                f"Training samples for class: {self._sample_counts.get(predicted_category, 0)}",
            ],
            "features": {
                "classifier": "trained_clip",
                "model_id": self._model_id or self.encoder.model_id,
                "training_samples": self._sample_counts.get(predicted_category, 0),
            },
        }

    def save(
        self,
        category_keys: list[str],
        centroids: np.ndarray,
        sample_counts: list[int],
        model_id: str,
    ) -> None:
        self.model_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            self.model_path,
            category_keys=np.array(category_keys, dtype="<U32"),
            centroids=centroids.astype(np.float32),
            sample_counts=np.array(sample_counts, dtype=np.int32),
            model_id=np.array(model_id),
        )
        self._loaded = False
        self._ensure_loaded()

    def summary(self) -> dict[str, Any]:
        self._ensure_loaded()
        return {
            "model_path": str(self.model_path),
            "model_id": self._model_id,
            "category_keys": self._category_keys,
            "sample_counts": self._sample_counts,
        }


def _normalize_score_dict(raw_scores: dict[str, float]) -> dict[str, float]:
    category_keys = list(raw_scores.keys())
    score_values = np.array(list(raw_scores.values()), dtype=np.float32)
    normalized_scores = _softmax(score_values)
    return {
        key: round(float(score), 4)
        for key, score in zip(category_keys, normalized_scores)
    }


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
    category_lookup = {category.key: category for category in CATEGORY_DEFINITIONS}
    scores = _normalize_score_dict(raw_scores)

    return {
        "filename": image_path.name,
        "category": predicted_category,
        "category_label": category_lookup[predicted_category].label,
        "confidence": scores[predicted_category],
        "scores": scores,
        "reasons": [
            "Fell back to heuristic rules because CLIP was unavailable.",
            "For better quality, collect labels and train the embedding classifier.",
        ],
        "features": {
            key: round(value, 4)
            for key, value in features.items()
        } | {"classifier": "heuristic_fallback"},
    }


_LOCAL_CLIP_ENCODER = LocalCLIPEncoder()
_LOCAL_CLIP_CLASSIFIER = LocalCLIPClassifier(_LOCAL_CLIP_ENCODER)
_TRAINED_EMBEDDING_CLASSIFIER = TrainedEmbeddingClassifier(_LOCAL_CLIP_ENCODER)


def get_clip_encoder() -> LocalCLIPEncoder:
    return _LOCAL_CLIP_ENCODER


def get_trained_embedding_classifier() -> TrainedEmbeddingClassifier:
    return _TRAINED_EMBEDDING_CLASSIFIER


def train_embedding_classifier(
    labeled_items: list[tuple[Path, str]],
    output_path: Path = TRAINED_MODEL_PATH,
) -> dict[str, Any]:
    if not labeled_items:
        raise ValueError("No labeled items were provided for training.")

    category_to_embeddings: dict[str, list[np.ndarray]] = {}
    encoder = get_clip_encoder()

    for image_path, label in labeled_items:
        embedding = encoder.encode_image(image_path)
        category_to_embeddings.setdefault(label, []).append(embedding)

    category_keys = sorted(category_to_embeddings)
    centroids = []
    sample_counts = []

    for key in category_keys:
        embeddings = np.stack(category_to_embeddings[key], axis=0)
        centroid = embeddings.mean(axis=0)
        centroid = centroid / np.linalg.norm(centroid)
        centroids.append(centroid.astype(np.float32))
        sample_counts.append(int(embeddings.shape[0]))

    classifier = TrainedEmbeddingClassifier(encoder, output_path)
    classifier.save(
        category_keys=category_keys,
        centroids=np.stack(centroids, axis=0),
        sample_counts=sample_counts,
        model_id=encoder.model_id,
    )

    return classifier.summary()


def classify_image(image_path: Path) -> dict:
    if CLASSIFIER_MODE == "heuristic":
        return _classify_with_heuristics(image_path)

    try:
        if _TRAINED_EMBEDDING_CLASSIFIER.is_available():
            return _TRAINED_EMBEDDING_CLASSIFIER.classify(image_path)
    except Exception:
        pass

    try:
        return _LOCAL_CLIP_CLASSIFIER.classify(image_path)
    except Exception:
        return _classify_with_heuristics(image_path)
