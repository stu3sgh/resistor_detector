#!/usr/bin/env python3

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np


LABEL_TO_ID = {"bad": -1, "good": 1}
ID_TO_LABEL = {-1: "bad", 1: "good"}


@dataclass
class HogFeatureConfig:
    width: int = 96
    height: int = 96
    block_size: int = 16
    block_stride: int = 8
    cell_size: int = 8
    bins: int = 9
    equalize_hist: bool = True
    svm_c: float = 1.0


def feature_config_from_dict(data: dict[str, Any] | None) -> HogFeatureConfig:
    if not data:
        return HogFeatureConfig()
    return HogFeatureConfig(
        width=int(data.get("width", 96)),
        height=int(data.get("height", 96)),
        block_size=int(data.get("block_size", 16)),
        block_stride=int(data.get("block_stride", 8)),
        cell_size=int(data.get("cell_size", 8)),
        bins=int(data.get("bins", 9)),
        equalize_hist=bool(data.get("equalize_hist", True)),
        svm_c=float(data.get("svm_c", 1.0)),
    )


def encode_label(label: str) -> int:
    normalized = label.strip().lower()
    if normalized not in LABEL_TO_ID:
        raise ValueError(f"Unsupported label: {label}")
    return LABEL_TO_ID[normalized]


def decode_label(label_id: int) -> str:
    normalized = int(label_id)
    if normalized not in ID_TO_LABEL:
        raise ValueError(f"Unsupported label id: {label_id}")
    return ID_TO_LABEL[normalized]


def build_hog_descriptor(config: HogFeatureConfig) -> cv2.HOGDescriptor:
    return cv2.HOGDescriptor(
        (int(config.width), int(config.height)),
        (int(config.block_size), int(config.block_size)),
        (int(config.block_stride), int(config.block_stride)),
        (int(config.cell_size), int(config.cell_size)),
        int(config.bins),
    )


def prepare_roi_for_hog(image_bgr: np.ndarray, config: HogFeatureConfig) -> np.ndarray:
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (int(config.width), int(config.height)), interpolation=cv2.INTER_AREA)
    if config.equalize_hist:
        resized = cv2.equalizeHist(resized)
    return resized


def compute_hog_features(image_bgr: np.ndarray, config: HogFeatureConfig) -> np.ndarray:
    descriptor = build_hog_descriptor(config)
    prepared = prepare_roi_for_hog(image_bgr, config)
    feature_vector = descriptor.compute(prepared)
    if feature_vector is None:
        raise ValueError("Failed to compute HOG features")
    return feature_vector.reshape(-1).astype(np.float32)


def build_feature_matrix(images: list[np.ndarray], config: HogFeatureConfig) -> np.ndarray:
    feature_rows = [compute_hog_features(image, config) for image in images]
    if not feature_rows:
        raise ValueError("No images provided for feature extraction")
    return np.vstack(feature_rows).astype(np.float32)


def create_linear_svm(c_value: float) -> Any:
    svm = cv2.ml.SVM_create()
    svm.setType(cv2.ml.SVM_C_SVC)
    svm.setKernel(cv2.ml.SVM_LINEAR)
    svm.setC(float(c_value))
    svm.setTermCriteria((cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, 5000, 1e-6))
    return svm


def train_linear_svm(feature_matrix: np.ndarray, label_ids: np.ndarray, c_value: float) -> Any:
    svm = create_linear_svm(c_value)
    labels = label_ids.reshape(-1, 1).astype(np.int32)
    ok = svm.train(feature_matrix.astype(np.float32), cv2.ml.ROW_SAMPLE, labels)
    if not ok:
        raise RuntimeError("OpenCV SVM training failed")
    return svm


def predict_label_ids(svm: Any, feature_matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    _, predicted = svm.predict(feature_matrix.astype(np.float32))
    _, raw_output = svm.predict(feature_matrix.astype(np.float32), flags=cv2.ml.StatModel_RAW_OUTPUT)
    return predicted.reshape(-1).astype(np.int32), raw_output.reshape(-1).astype(np.float32)


def save_model_bundle(
    output_dir: Path,
    svm: Any,
    feature_config: HogFeatureConfig,
    training_summary: dict[str, Any],
) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / "hog_svm.xml"
    metadata_path = output_dir / "hog_svm.json"
    svm.save(str(model_path))
    metadata = {
        "model_type": "opencv_hog_linear_svm",
        "feature_config": asdict(feature_config),
        "label_to_id": LABEL_TO_ID,
        "training_summary": training_summary,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    return model_path, metadata_path


def load_model_bundle(model_dir: Path) -> tuple[Any, HogFeatureConfig, dict[str, Any]]:
    model_path = model_dir / "hog_svm.xml"
    metadata_path = model_dir / "hog_svm.json"
    if not model_path.exists():
        raise FileNotFoundError(f"Missing model file: {model_path}")
    if not metadata_path.exists():
        raise FileNotFoundError(f"Missing metadata file: {metadata_path}")

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    feature_config = feature_config_from_dict(metadata.get("feature_config"))
    svm = cv2.ml.SVM_load(str(model_path))
    return svm, feature_config, metadata


__all__ = [
    "HogFeatureConfig",
    "build_feature_matrix",
    "compute_hog_features",
    "decode_label",
    "encode_label",
    "feature_config_from_dict",
    "load_model_bundle",
    "predict_label_ids",
    "save_model_bundle",
    "train_linear_svm",
]