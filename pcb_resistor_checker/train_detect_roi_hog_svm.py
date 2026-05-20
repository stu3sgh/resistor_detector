#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any

import cv2
import numpy as np

from detect_resistor_presence import resolve_path_argument
from roi_classifier_common import DEFAULT_IMAGE_PATTERNS, list_image_files
from roi_classifier_hog_svm import (
    HogFeatureConfig,
    decode_label,
    encode_label,
    predict_label_ids,
    save_model_bundle,
    train_linear_svm,
    build_feature_matrix,
)


@dataclass
class DatasetItem:
    image_path: Path
    label: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a HOG + linear SVM classifier on any region dataset organized as good/ and bad/."
    )
    parser.add_argument("--dataset-root", required=True, help="Dataset root containing good/ and bad/ folders")
    parser.add_argument("--output-dir", required=True, help="Directory to save trained model artifacts")
    parser.add_argument("--region-name", default="region", help="Logical region name for metadata only")
    parser.add_argument("--labels", nargs="+", default=["good", "bad"], help="Label folder names")
    parser.add_argument("--patterns", nargs="+", default=list(DEFAULT_IMAGE_PATTERNS), help="Image glob patterns")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--folds", type=int, default=5, help="Number of stratified folds for evaluation. Set 0 or 1 to disable")
    parser.add_argument("--val-ratio", type=float, default=0.3, help="Validation ratio when folds are disabled")
    parser.add_argument("--width", type=int, default=96, help="Resized ROI width for HOG features")
    parser.add_argument("--height", type=int, default=96, help="Resized ROI height for HOG features")
    parser.add_argument("--block-size", type=int, default=16, help="HOG block size")
    parser.add_argument("--block-stride", type=int, default=8, help="HOG block stride")
    parser.add_argument("--cell-size", type=int, default=8, help="HOG cell size")
    parser.add_argument("--bins", type=int, default=9, help="HOG orientation bins")
    parser.add_argument("--svm-c", type=float, default=1.0, help="Linear SVM C value")
    parser.add_argument("--no-equalize-hist", action="store_true", help="Disable histogram equalization before HOG")
    parser.add_argument(
        "--balance",
        choices=["none", "oversample"],
        default="none",
        help="Optional class balancing strategy applied to training folds and the final fitted model.",
    )
    return parser.parse_args()


def collect_dataset_items(dataset_root: Path, labels: list[str], patterns: list[str]) -> list[DatasetItem]:
    items: list[DatasetItem] = []
    for raw_label in labels:
        label = raw_label.strip().lower()
        label_dir = dataset_root / label
        if not label_dir.exists():
            raise FileNotFoundError(f"Missing label directory: {label_dir}")
        for image_path in list_image_files(label_dir, patterns):
            items.append(DatasetItem(image_path=image_path, label=label))
    if not items:
        raise ValueError(f"No training images found under {dataset_root}")
    return items


def group_items_by_label(items: list[DatasetItem]) -> dict[str, list[DatasetItem]]:
    grouped: dict[str, list[DatasetItem]] = {}
    for item in items:
        grouped.setdefault(item.label, []).append(item)
    return grouped


def balance_training_items(items: list[DatasetItem], strategy: str, seed: int) -> list[DatasetItem]:
    if strategy == "none":
        return list(items)

    grouped = group_items_by_label(items)
    if not grouped:
        return []

    max_count = max(len(group) for group in grouped.values())
    rng = random.Random(seed)
    balanced_items: list[DatasetItem] = []
    for label in sorted(grouped):
        label_items = list(grouped[label])
        if not label_items:
            continue
        expanded = list(label_items)
        while len(expanded) < max_count:
            expanded.append(rng.choice(label_items))
        balanced_items.extend(expanded)

    rng.shuffle(balanced_items)
    return balanced_items


def build_stratified_folds(items: list[DatasetItem], folds: int, seed: int) -> list[list[DatasetItem]]:
    if folds < 2:
        raise ValueError("folds must be at least 2")

    grouped = group_items_by_label(items)
    fold_buckets: list[list[DatasetItem]] = [[] for _ in range(folds)]
    rng = random.Random(seed)

    for label, label_items in grouped.items():
        if len(label_items) < folds:
            raise ValueError(f"Label '{label}' only has {len(label_items)} samples, not enough for {folds} folds")
        shuffled = list(label_items)
        rng.shuffle(shuffled)
        for index, item in enumerate(shuffled):
            fold_buckets[index % folds].append(item)
    return fold_buckets


def stratified_holdout_split(items: list[DatasetItem], val_ratio: float, seed: int) -> tuple[list[DatasetItem], list[DatasetItem]]:
    grouped = group_items_by_label(items)
    rng = random.Random(seed)
    train_items: list[DatasetItem] = []
    val_items: list[DatasetItem] = []

    for label, label_items in grouped.items():
        shuffled = list(label_items)
        rng.shuffle(shuffled)
        if len(shuffled) < 2:
            raise ValueError(f"Label '{label}' needs at least 2 samples for holdout evaluation")
        val_count = max(1, int(round(len(shuffled) * val_ratio)))
        val_count = min(val_count, len(shuffled) - 1)
        val_items.extend(shuffled[:val_count])
        train_items.extend(shuffled[val_count:])
    return train_items, val_items


def load_images(items: list[DatasetItem]) -> list[np.ndarray]:
    images: list[np.ndarray] = []
    for item in items:
        image = cv2.imread(str(item.image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f"Failed to read training image: {item.image_path}")
        images.append(image)
    return images


def compute_metrics(true_labels: list[str], pred_labels: list[str]) -> dict[str, Any]:
    confusion = {
        "good_as_good": 0,
        "good_as_bad": 0,
        "bad_as_good": 0,
        "bad_as_bad": 0,
    }
    for truth, pred in zip(true_labels, pred_labels):
        confusion[f"{truth}_as_{pred}"] += 1

    total = len(true_labels)
    correct = confusion["good_as_good"] + confusion["bad_as_bad"]

    def safe_divide(numerator: float, denominator: float) -> float:
        if denominator == 0:
            return 0.0
        return numerator / denominator

    precision_good = safe_divide(confusion["good_as_good"], confusion["good_as_good"] + confusion["bad_as_good"])
    recall_good = safe_divide(confusion["good_as_good"], confusion["good_as_good"] + confusion["good_as_bad"])
    precision_bad = safe_divide(confusion["bad_as_bad"], confusion["good_as_bad"] + confusion["bad_as_bad"])
    recall_bad = safe_divide(confusion["bad_as_bad"], confusion["bad_as_good"] + confusion["bad_as_bad"])
    f1_good = safe_divide(2 * precision_good * recall_good, precision_good + recall_good)
    f1_bad = safe_divide(2 * precision_bad * recall_bad, precision_bad + recall_bad)
    accuracy = safe_divide(correct, total)

    return {
        "sample_count": total,
        "confusion": confusion,
        "metrics": {
            "accuracy": round(accuracy, 6),
            "precision_good": round(precision_good, 6),
            "recall_good": round(recall_good, 6),
            "f1_good": round(f1_good, 6),
            "precision_bad": round(precision_bad, 6),
            "recall_bad": round(recall_bad, 6),
            "f1_bad": round(f1_bad, 6),
        },
    }


def evaluate_split(
    train_items: list[DatasetItem],
    val_items: list[DatasetItem],
    feature_config: HogFeatureConfig,
    *,
    balance_strategy: str,
    seed: int,
) -> dict[str, Any]:
    effective_train_items = balance_training_items(train_items, balance_strategy, seed)

    train_images = load_images(effective_train_items)
    val_images = load_images(val_items)

    train_features = build_feature_matrix(train_images, feature_config)
    train_labels = np.array([encode_label(item.label) for item in effective_train_items], dtype=np.int32)
    svm = train_linear_svm(train_features, train_labels, feature_config.svm_c)

    val_features = build_feature_matrix(val_images, feature_config)
    pred_ids, raw_scores = predict_label_ids(svm, val_features)
    pred_labels = [decode_label(int(label_id)) for label_id in pred_ids]
    true_labels = [item.label for item in val_items]
    evaluation = compute_metrics(true_labels, pred_labels)
    evaluation["raw_scores"] = [round(float(score), 6) for score in raw_scores.tolist()]
    evaluation["validation_items"] = [
        {
            "image": str(item.image_path),
            "truth": item.label,
            "prediction": pred_label,
        }
        for item, pred_label in zip(val_items, pred_labels)
    ]
    evaluation["train_sample_count_before_balance"] = len(train_items)
    evaluation["train_sample_count_after_balance"] = len(effective_train_items)
    return evaluation


def build_final_model(items: list[DatasetItem], feature_config: HogFeatureConfig, balance_strategy: str, seed: int) -> tuple[Any, list[DatasetItem]]:
    effective_items = balance_training_items(items, balance_strategy, seed)
    images = load_images(effective_items)
    features = build_feature_matrix(images, feature_config)
    labels = np.array([encode_label(item.label) for item in effective_items], dtype=np.int32)
    return train_linear_svm(features, labels, feature_config.svm_c), effective_items


def aggregate_fold_metrics(fold_summaries: list[dict[str, Any]]) -> dict[str, Any]:
    metric_names = [
        "accuracy",
        "precision_good",
        "recall_good",
        "f1_good",
        "precision_bad",
        "recall_bad",
        "f1_bad",
    ]
    return {
        name: round(mean(summary["metrics"][name] for summary in fold_summaries), 6)
        for name in metric_names
    }


def main() -> int:
    args = parse_args()
    dataset_root = resolve_path_argument(args.dataset_root)
    output_dir = resolve_path_argument(args.output_dir)
    feature_config = HogFeatureConfig(
        width=args.width,
        height=args.height,
        block_size=args.block_size,
        block_stride=args.block_stride,
        cell_size=args.cell_size,
        bins=args.bins,
        equalize_hist=not args.no_equalize_hist,
        svm_c=args.svm_c,
    )

    items = collect_dataset_items(dataset_root, args.labels, args.patterns)
    label_counts = {label: len(group) for label, group in group_items_by_label(items).items()}

    evaluation_summary: dict[str, Any]
    if args.folds >= 2:
        folds = build_stratified_folds(items, args.folds, args.seed)
        fold_summaries = []
        for fold_index in range(args.folds):
            val_items = list(folds[fold_index])
            train_items = [item for index, fold in enumerate(folds) if index != fold_index for item in fold]
            fold_result = evaluate_split(
                train_items,
                val_items,
                feature_config,
                balance_strategy=args.balance,
                seed=args.seed + fold_index,
            )
            fold_result["fold_index"] = fold_index
            fold_summaries.append(fold_result)
            print(
                f"fold={fold_index} accuracy={fold_result['metrics']['accuracy']:.4f} "
                f"recall_bad={fold_result['metrics']['recall_bad']:.4f} "
                f"precision_bad={fold_result['metrics']['precision_bad']:.4f}"
            )
        evaluation_summary = {
            "mode": "stratified_kfold",
            "folds": args.folds,
            "fold_summaries": fold_summaries,
            "mean_metrics": aggregate_fold_metrics(fold_summaries),
        }
    else:
        train_items, val_items = stratified_holdout_split(items, args.val_ratio, args.seed)
        holdout_result = evaluate_split(
            train_items,
            val_items,
            feature_config,
            balance_strategy=args.balance,
            seed=args.seed,
        )
        evaluation_summary = {
            "mode": "holdout",
            "val_ratio": args.val_ratio,
            "summary": holdout_result,
        }
        print(
            f"holdout accuracy={holdout_result['metrics']['accuracy']:.4f} "
            f"recall_bad={holdout_result['metrics']['recall_bad']:.4f} "
            f"precision_bad={holdout_result['metrics']['precision_bad']:.4f}"
        )

    svm, effective_items = build_final_model(items, feature_config, args.balance, args.seed)
    effective_label_counts = {label: len(group) for label, group in group_items_by_label(effective_items).items()}
    training_summary = {
        "region_name": args.region_name,
        "dataset_root": str(dataset_root),
        "label_counts": label_counts,
        "effective_label_counts": effective_label_counts,
        "balance_strategy": args.balance,
        "sample_count": len(items),
        "samples": [{"image": str(item.image_path), "label": item.label} for item in items],
        "evaluation": evaluation_summary,
    }
    model_path, metadata_path = save_model_bundle(output_dir, svm, feature_config, training_summary)
    print(f"model_path={model_path}")
    print(f"metadata_path={metadata_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())