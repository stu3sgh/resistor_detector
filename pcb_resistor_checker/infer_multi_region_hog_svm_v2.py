#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import cv2

from detect_resistor_presence import load_config, load_template_state, resolve_path_argument
from roi_classifier_common import localize_detect_roi, mask_roi_crop, roi_bbox_to_dict
from roi_classifier_hog_svm import compute_hog_features, decode_label, load_model_bundle, predict_label_ids


SCRIPT_DIR = Path(__file__).resolve().parent
MODEL_ARTIFACTS = ("hog_svm.xml", "hog_svm.json")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run multi-region inference for smd_components, main_chip, and bottom_chip with one JSON output."
    )
    parser.add_argument(
        "--config",
        default=str(SCRIPT_DIR / "config.yaml"),
        help="SMD localization config. Defaults to pcb_resistor_checker/config.yaml.",
    )
    parser.add_argument(
        "--smd-components-image",
        "--smd-components-path",
        dest="smd_components_image",
        required=True,
        help="Input patch image for the smd_components region.",
    )
    parser.add_argument(
        "--main-chip-image",
        required=True,
        help="Input patch image for the main_chip region.",
    )
    parser.add_argument(
        "--bottom-chip-image",
        required=True,
        help="Input patch image for the bottom_chip region.",
    )
    parser.add_argument(
        "--smd-components-model-dir",
        help="Optional override for the smd_components model directory. By default the highest hog_svm_v* is used.",
    )
    parser.add_argument(
        "--main-chip-model-dir",
        help="Optional override for the main_chip model directory. By default the highest main_chip_hog_svm_v* is used.",
    )
    parser.add_argument(
        "--bottom-chip-model-dir",
        help="Optional override for the bottom_chip model directory. By default the highest bottom_chip_hog_svm_v* is used.",
    )
    parser.add_argument("--output-json", help="Optional path to save the combined JSON output.")
    return parser.parse_args()


def find_latest_versioned_model_dir(models_root: Path, prefix: str) -> Path:
    if not models_root.exists():
        raise FileNotFoundError(f"Missing models directory: {models_root}")

    pattern = re.compile(rf"^{re.escape(prefix)}_v(\d+)$", re.IGNORECASE)
    candidates: list[tuple[int, str, Path]] = []
    for child in models_root.iterdir():
        if not child.is_dir():
            continue
        match = pattern.fullmatch(child.name)
        if match is None:
            continue
        if not all((child / artifact).exists() for artifact in MODEL_ARTIFACTS):
            continue
        candidates.append((int(match.group(1)), child.name.lower(), child))

    if not candidates:
        raise FileNotFoundError(
            f"No versioned model directory found for prefix '{prefix}' under {models_root}"
        )

    candidates.sort(key=lambda item: (item[0], item[1]))
    return candidates[-1][2]


def resolve_model_dir(model_dir_arg: str | None, models_root: Path, prefix: str) -> Path:
    if model_dir_arg:
        return resolve_path_argument(model_dir_arg, SCRIPT_DIR)
    return find_latest_versioned_model_dir(models_root, prefix)


def classify_direct_region(
    region_name: str,
    image_path: Path,
    svm: Any,
    feature_config: Any,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise RuntimeError(f"Failed to read image: {image_path}")

    features = compute_hog_features(image_bgr, feature_config).reshape(1, -1)
    predicted_ids, raw_scores = predict_label_ids(svm, features)
    predicted_label = decode_label(int(predicted_ids[0]))
    return {
        "region": region_name,
        "image": str(image_path),
        "result": predicted_label,
        "reason": "hog_linear_svm",
        "model_type": metadata.get("model_type"),
        "svm_label_id": int(predicted_ids[0]),
        "svm_raw_output": round(float(raw_scores[0]), 6),
        "svm_margin_abs": round(abs(float(raw_scores[0])), 6),
    }


def classify_smd_region(
    image_path: Path,
    config: dict[str, Any],
    template_state: Any,
    svm: Any,
    feature_config: Any,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    localized, payload = localize_detect_roi(image_path, config, template_state)
    if localized is None:
        result = {
            "region": "smd_components",
            "image": str(image_path),
            "result": "bad",
            "reason": "localization_failed",
            "localization_reason": payload.get("reason", "unknown"),
            "localization_ok": False,
            "model_type": metadata.get("model_type"),
        }
        for key in ("total_good_matches", "inlier_count", "anchor_stats", "transform_type", "transform_summary"):
            if key in payload:
                result[key] = payload[key]
        return result

    masked_roi = mask_roi_crop(localized.roi_crop_bgr, localized.roi_mask)
    features = compute_hog_features(masked_roi, feature_config).reshape(1, -1)
    predicted_ids, raw_scores = predict_label_ids(svm, features)
    predicted_label = decode_label(int(predicted_ids[0]))
    return {
        "region": "smd_components",
        "image": str(image_path),
        "result": predicted_label,
        "reason": "hog_linear_svm",
        "localization_ok": True,
        "model_type": metadata.get("model_type"),
        "total_good_matches": localized.total_good_matches,
        "inlier_count": localized.inlier_count,
        "anchor_stats": localized.anchor_stats,
        "transform_type": localized.transform_type,
        "transform_summary": localized.transform_summary,
        "roi_bbox": roi_bbox_to_dict(localized.roi_bbox),
        "svm_label_id": int(predicted_ids[0]),
        "svm_raw_output": round(float(raw_scores[0]), 6),
        "svm_margin_abs": round(abs(float(raw_scores[0])), 6),
    }


def build_summary(region_results: dict[str, dict[str, Any]]) -> dict[str, Any]:
    bad_regions = [name for name, payload in region_results.items() if payload.get("result") != "good"]
    localization_failures = [
        name for name, payload in region_results.items() if payload.get("reason") == "localization_failed"
    ]
    return {
        "overall_result": "good" if not bad_regions else "bad",
        "bad_regions": bad_regions,
        "localization_failures": localization_failures,
    }


def main() -> int:
    args = parse_args()

    config_path = resolve_path_argument(args.config, SCRIPT_DIR)
    smd_image_path = resolve_path_argument(args.smd_components_image)
    main_chip_image_path = resolve_path_argument(args.main_chip_image)
    bottom_chip_image_path = resolve_path_argument(args.bottom_chip_image)
    models_root = SCRIPT_DIR / "models"
    smd_model_dir = resolve_model_dir(args.smd_components_model_dir, models_root, "hog_svm")
    main_chip_model_dir = resolve_model_dir(args.main_chip_model_dir, models_root, "main_chip_hog_svm")
    bottom_chip_model_dir = resolve_model_dir(args.bottom_chip_model_dir, models_root, "bottom_chip_hog_svm")

    config = load_config(config_path)
    template_state = load_template_state(config, config_path)

    smd_svm, smd_feature_config, smd_metadata = load_model_bundle(smd_model_dir)
    main_chip_svm, main_chip_feature_config, main_chip_metadata = load_model_bundle(main_chip_model_dir)
    bottom_chip_svm, bottom_chip_feature_config, bottom_chip_metadata = load_model_bundle(bottom_chip_model_dir)

    region_results = {
        "smd_components": classify_smd_region(
            smd_image_path,
            config,
            template_state,
            smd_svm,
            smd_feature_config,
            smd_metadata,
        ),
        "main_chip": classify_direct_region(
            "main_chip",
            main_chip_image_path,
            main_chip_svm,
            main_chip_feature_config,
            main_chip_metadata,
        ),
        "bottom_chip": classify_direct_region(
            "bottom_chip",
            bottom_chip_image_path,
            bottom_chip_svm,
            bottom_chip_feature_config,
            bottom_chip_metadata,
        ),
    }

    payload = {
        **region_results,
        "summary": build_summary(region_results),
    }

    rendered = json.dumps(payload, indent=2, ensure_ascii=False)
    print(rendered)

    if args.output_json:
        output_path = resolve_path_argument(args.output_json)
        output_path.write_text(rendered + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())