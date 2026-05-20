#!/usr/bin/env python3

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import cv2
import numpy as np

from detect_resistor_presence import (
    TemplateState,
    apply_transform,
    build_orb,
    crop_polygon_region,
    estimate_transform,
    load_config,
    load_template_state,
    match_anchor_descriptors,
    preprocess_gray,
    read_image_bgr,
    standardize_image_to_shape,
    summarize_transform,
    warp_to_template,
)


DEFAULT_IMAGE_PATTERNS = ("*.png", "*.jpg", "*.jpeg", "*.bmp", "*.webp")


@dataclass
class LocalizedRoi:
    image_path: Path
    raw_image_bgr: np.ndarray
    working_image_bgr: np.ndarray
    aligned_image_bgr: np.ndarray
    roi_crop_bgr: np.ndarray
    roi_mask: np.ndarray
    roi_bbox: tuple[int, int, int, int]
    projected_roi: np.ndarray
    transform_matrix: np.ndarray
    transform_type: str
    transform_summary: dict[str, Any]
    total_good_matches: int
    inlier_count: int
    anchor_stats: list[dict[str, Any]]


def list_image_files(root_dir: Path, patterns: Sequence[str] = DEFAULT_IMAGE_PATTERNS) -> list[Path]:
    matches: list[Path] = []
    seen: set[Path] = set()
    for pattern in patterns:
        for item in sorted(root_dir.glob(pattern)):
            resolved = item.resolve()
            if resolved in seen or not resolved.is_file():
                continue
            seen.add(resolved)
            matches.append(resolved)
    return matches


def roi_bbox_to_dict(roi_bbox: tuple[int, int, int, int]) -> dict[str, int]:
    x, y, width, height = roi_bbox
    return {
        "x": int(x),
        "y": int(y),
        "width": int(width),
        "height": int(height),
    }


def mask_roi_crop(
    roi_crop_bgr: np.ndarray,
    roi_mask: np.ndarray,
    fill_value: int | tuple[int, int, int] = 0,
) -> np.ndarray:
    masked = np.zeros_like(roi_crop_bgr)
    if isinstance(fill_value, tuple):
        masked[:] = fill_value
    else:
        masked.fill(int(fill_value))
    valid = roi_mask > 0
    masked[valid] = roi_crop_bgr[valid]
    return masked


def localize_detect_roi(
    image_path: Path,
    config: dict[str, Any],
    template_state: TemplateState,
    *,
    margin_px: int | None = None,
) -> tuple[LocalizedRoi | None, dict[str, Any]]:
    honor_exif_orientation = bool(config["preprocess"].get("honor_exif_orientation", True))
    raw_image_bgr = read_image_bgr(image_path, honor_exif_orientation)
    working_image_bgr = standardize_image_to_shape(
        raw_image_bgr,
        template_state.image_bgr.shape,
        config["preprocess"],
    )

    test_gray = preprocess_gray(working_image_bgr, config["preprocess"])
    orb = build_orb(config["orb"])
    test_keypoints, test_descriptors = orb.detectAndCompute(test_gray, None)
    base_payload = {
        "image": str(image_path),
        "input_original_size": [raw_image_bgr.shape[1], raw_image_bgr.shape[0]],
        "input_working_size": [working_image_bgr.shape[1], working_image_bgr.shape[0]],
    }
    if test_descriptors is None or len(test_keypoints) == 0:
        return None, {
            **base_payload,
            "ok": False,
            "reason": "no_features_in_test_image",
        }

    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    all_src_points = []
    all_dst_points = []
    anchor_stats = []
    for anchor in template_state.anchors:
        src_points, dst_points, stats = match_anchor_descriptors(
            matcher,
            anchor,
            test_keypoints,
            test_descriptors,
            config["matching"],
        )
        anchor_stats.append(stats)
        if len(src_points) > 0:
            all_src_points.append(src_points)
            all_dst_points.append(dst_points)

    total_good_matches = sum(item["good_matches"] for item in anchor_stats)
    if total_good_matches < int(config["matching"]["min_total_matches"]) or not all_src_points:
        return None, {
            **base_payload,
            "ok": False,
            "reason": "not_enough_anchor_matches",
            "total_good_matches": total_good_matches,
            "anchor_stats": anchor_stats,
        }

    src_points = np.concatenate(all_src_points, axis=0)
    dst_points = np.concatenate(all_dst_points, axis=0)
    transform_matrix, inliers = estimate_transform(src_points, dst_points, config["matching"])
    if transform_matrix is None or inliers is None:
        return None, {
            **base_payload,
            "ok": False,
            "reason": "transform_estimation_failed",
            "total_good_matches": total_good_matches,
            "anchor_stats": anchor_stats,
        }

    inlier_count = int(np.count_nonzero(inliers))
    transform_type = str(config["matching"]["transform_type"])
    transform_summary = summarize_transform(transform_matrix, transform_type)
    if inlier_count < int(config["matching"]["min_inliers"]):
        return None, {
            **base_payload,
            "ok": False,
            "reason": "not_enough_inliers",
            "total_good_matches": total_good_matches,
            "inlier_count": inlier_count,
            "anchor_stats": anchor_stats,
            "transform_type": transform_type,
            "transform_summary": transform_summary,
        }

    aligned_image_bgr = warp_to_template(
        working_image_bgr,
        transform_matrix,
        transform_type,
        template_state.image_bgr.shape,
    )
    roi_crop_bgr, roi_mask, roi_bbox = crop_polygon_region(
        aligned_image_bgr,
        template_state.detect_roi,
        margin_px=int(config["decision"].get("ring_margin_px", 10) if margin_px is None else margin_px),
    )
    projected_roi = apply_transform(template_state.detect_roi, transform_matrix, transform_type)

    localized = LocalizedRoi(
        image_path=image_path,
        raw_image_bgr=raw_image_bgr,
        working_image_bgr=working_image_bgr,
        aligned_image_bgr=aligned_image_bgr,
        roi_crop_bgr=roi_crop_bgr,
        roi_mask=roi_mask,
        roi_bbox=roi_bbox,
        projected_roi=projected_roi,
        transform_matrix=np.asarray(transform_matrix),
        transform_type=transform_type,
        transform_summary=transform_summary,
        total_good_matches=total_good_matches,
        inlier_count=inlier_count,
        anchor_stats=anchor_stats,
    )
    payload = {
        **base_payload,
        "ok": True,
        "reason": "localized",
        "total_good_matches": total_good_matches,
        "inlier_count": inlier_count,
        "anchor_stats": anchor_stats,
        "transform_type": transform_type,
        "transform_summary": transform_summary,
        "roi_bbox": roi_bbox_to_dict(roi_bbox),
    }
    return localized, payload


__all__ = [
    "DEFAULT_IMAGE_PATTERNS",
    "LocalizedRoi",
    "list_image_files",
    "load_config",
    "load_template_state",
    "localize_detect_roi",
    "mask_roi_crop",
    "roi_bbox_to_dict",
]