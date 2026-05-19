#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import cv2

from detect_resistor_presence import resolve_path_argument
from roi_classifier_common import (
    DEFAULT_IMAGE_PATTERNS,
    list_image_files,
    load_config,
    load_template_state,
    localize_detect_roi,
    mask_roi_crop,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract aligned detect_roi crops from labeled full images for classifier training."
    )
    parser.add_argument("--config", required=True, help="Path to detector config YAML/JSON")
    parser.add_argument("--input-root", required=True, help="Root directory containing label folders such as good/ and bad/")
    parser.add_argument("--output-root", required=True, help="Output root for extracted ROI crops")
    parser.add_argument(
        "--labels",
        nargs="+",
        default=["good", "bad"],
        help="Label folder names to extract from input-root",
    )
    parser.add_argument(
        "--patterns",
        nargs="+",
        default=list(DEFAULT_IMAGE_PATTERNS),
        help="Image glob patterns to collect from each label folder",
    )
    parser.add_argument(
        "--margin-px",
        type=int,
        help="Optional override for ROI crop margin in pixels",
    )
    parser.add_argument(
        "--metadata-csv",
        help="Optional metadata CSV path. Defaults to <output-root>/metadata.csv",
    )
    return parser.parse_args()


def write_metadata_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "label",
        "status",
        "reason",
        "source_image",
        "crop_path",
        "total_good_matches",
        "inlier_count",
        "roi_x",
        "roi_y",
        "roi_width",
        "roi_height",
        "transform_summary",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> int:
    args = parse_args()
    config_path = resolve_path_argument(args.config)
    input_root = resolve_path_argument(args.input_root)
    output_root = resolve_path_argument(args.output_root)
    metadata_path = resolve_path_argument(args.metadata_csv, output_root) if args.metadata_csv else output_root / "metadata.csv"

    config = load_config(config_path)
    template_state = load_template_state(config, config_path)

    output_root.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    label_summary: dict[str, dict[str, int]] = {}

    for raw_label in args.labels:
        label = raw_label.strip().lower()
        input_dir = input_root / label
        if not input_dir.exists():
            raise FileNotFoundError(f"Missing label directory: {input_dir}")

        output_dir = output_root / label
        output_dir.mkdir(parents=True, exist_ok=True)
        images = list_image_files(input_dir, args.patterns)
        label_summary[label] = {"source_count": len(images), "extracted_count": 0, "failed_count": 0}

        for image_path in images:
            localized, payload = localize_detect_roi(
                image_path,
                config,
                template_state,
                margin_px=args.margin_px,
            )
            row = {
                "label": label,
                "status": "ok" if localized is not None else "failed",
                "reason": payload.get("reason", "unknown"),
                "source_image": str(image_path),
                "crop_path": "",
                "total_good_matches": payload.get("total_good_matches", ""),
                "inlier_count": payload.get("inlier_count", ""),
                "roi_x": "",
                "roi_y": "",
                "roi_width": "",
                "roi_height": "",
                "transform_summary": json.dumps(payload.get("transform_summary", {}), ensure_ascii=False),
            }

            if localized is None:
                label_summary[label]["failed_count"] += 1
                rows.append(row)
                continue

            masked_crop = mask_roi_crop(localized.roi_crop_bgr, localized.roi_mask)
            output_path = output_dir / f"{image_path.stem}.png"
            ok = cv2.imwrite(str(output_path), masked_crop)
            if not ok:
                raise RuntimeError(f"Failed to write extracted ROI: {output_path}")

            x, y, width, height = localized.roi_bbox
            row.update(
                {
                    "crop_path": str(output_path),
                    "roi_x": int(x),
                    "roi_y": int(y),
                    "roi_width": int(width),
                    "roi_height": int(height),
                }
            )
            label_summary[label]["extracted_count"] += 1
            rows.append(row)

    write_metadata_csv(metadata_path, rows)

    summary = {
        "config": str(config_path),
        "input_root": str(input_root),
        "output_root": str(output_root),
        "metadata_csv": str(metadata_path),
        "labels": label_summary,
        "total_rows": len(rows),
    }
    (output_root / "extraction_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    for label, counts in label_summary.items():
        print(
            f"label={label} source={counts['source_count']} extracted={counts['extracted_count']} failed={counts['failed_count']}"
        )
    print(f"metadata_csv={metadata_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())