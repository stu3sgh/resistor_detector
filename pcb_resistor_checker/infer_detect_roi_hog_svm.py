#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import cv2

from detect_resistor_presence import (
    load_config,
    load_template_state,
    resolve_image_paths,
    resolve_path_argument,
)
from roi_classifier_common import localize_detect_roi, mask_roi_crop, roi_bbox_to_dict
from roi_classifier_hog_svm import compute_hog_features, load_model_bundle, decode_label, predict_label_ids


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run full-image inference using detect_roi localization + HOG/SVM classifier."
    )
    parser.add_argument("--config", required=True, help="Path to detector config YAML/JSON")
    parser.add_argument("--model-dir", required=True, help="Directory containing hog_svm.xml and hog_svm.json")
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--image", help="Single full image path")
    input_group.add_argument("--glob", dest="glob_pattern", help="Glob for batch full-image inference")
    parser.add_argument("--debug-dir", help="Optional debug directory for aligned ROI crops and JSON outputs")
    parser.add_argument("--output-json", help="Optional JSON file to save results")
    parser.add_argument("--result-only", action="store_true", help="Print only good/bad in single-image mode")
    return parser.parse_args()


def format_result(result: dict[str, Any]) -> str:
    parts = [
        f"image={result['image']}",
        f"result={result['result']}",
        f"reason={result['reason']}",
    ]
    if "total_good_matches" in result:
        parts.append(f"matches={result['total_good_matches']}")
    if "inlier_count" in result:
        parts.append(f"inliers={result['inlier_count']}")
    if "svm_raw_output" in result:
        parts.append(f"svm_raw={result['svm_raw_output']}")
    return " | ".join(parts)


def save_debug_outputs(debug_root: Path, image_path: Path, result: dict[str, Any], localized, masked_roi) -> None:
    debug_dir = debug_root / image_path.stem
    debug_dir.mkdir(parents=True, exist_ok=True)
    if localized is None:
        (debug_dir / "00_result.json").write_text(
            json.dumps(result, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        return

    overlay = localized.aligned_image_bgr.copy()
    x, y, width, height = localized.roi_bbox
    cv2.rectangle(overlay, (x, y), (x + width, y + height), (0, 255, 255), 2)
    color = (0, 200, 0) if result["result"] == "good" else (0, 0, 255)
    cv2.putText(
        overlay,
        f"result={result['result']}",
        (max(0, x), max(20, y - 10)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        color,
        2,
        cv2.LINE_AA,
    )
    cv2.imwrite(str(debug_dir / "00_aligned_overlay.jpg"), overlay)
    cv2.imwrite(str(debug_dir / "01_roi_masked.png"), masked_roi)
    (debug_dir / "02_result.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def main() -> int:
    args = parse_args()
    config_path = resolve_path_argument(args.config)
    model_dir = resolve_path_argument(args.model_dir)
    image_paths = resolve_image_paths(args.image, args.glob_pattern)
    debug_root = resolve_path_argument(args.debug_dir) if args.debug_dir else None

    if args.result_only and len(image_paths) != 1:
        raise SystemExit("--result-only can only be used with --image")

    config = load_config(config_path)
    template_state = load_template_state(config, config_path)
    svm, feature_config, metadata = load_model_bundle(model_dir)

    results = []
    for image_path in image_paths:
        localized, payload = localize_detect_roi(image_path, config, template_state)
        masked_roi = None
        if localized is None:
            result = {
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
        else:
            masked_roi = mask_roi_crop(localized.roi_crop_bgr, localized.roi_mask)
            features = compute_hog_features(masked_roi, feature_config).reshape(1, -1)
            predicted_ids, raw_scores = predict_label_ids(svm, features)
            predicted_label = decode_label(int(predicted_ids[0]))
            result = {
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

        results.append(result)
        if args.result_only:
            print(result["result"])
        else:
            print(format_result(result))
        if debug_root is not None:
            save_debug_outputs(debug_root, image_path, result, localized, masked_roi)

    if args.output_json:
        output_path = resolve_path_argument(args.output_json)
        output_path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())