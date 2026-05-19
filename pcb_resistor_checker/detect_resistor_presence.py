#!/usr/bin/env python3

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any, Iterable, Sequence

import numpy as np
from PIL import Image, ImageOps

try:
    import yaml
except ImportError:  # pragma: no cover - optional until YAML config is used
    yaml = None

try:
    import cv2
except ImportError as exc:  # pragma: no cover - handled at runtime
    raise SystemExit(
        "Missing dependency: cv2. Install opencv-python-headless from requirements.txt first."
    ) from exc


DEFAULT_CONFIG = {
    "orb": {
        "nfeatures": 2500,
        "scaleFactor": 1.2,
        "nlevels": 8,
        "edgeThreshold": 31,
        "fastThreshold": 12,
    },
    "matching": {
        "ratio_test": 0.78,
        "min_anchor_matches": 8,
        "min_total_matches": 24,
        "min_inliers": 12,
        "ransac_reproj_threshold": 5.0,
        "transform_type": "affine_partial",
        "max_matches_per_anchor": 80,
    },
    "preprocess": {
        "honor_exif_orientation": True,
        "standardize_to_template": False,
        "standardize_mode": "fit_pad",
        "gaussian_blur_ksize": 3,
        "clahe_clip_limit": 2.0,
        "clahe_tile_grid_size": 8,
    },
    "decision": {
        "mode": "legacy",
        "fixed_black_threshold": 80,
        "min_black_threshold": 25,
        "local_dark_offset": 30,
        "ring_margin_px": 10,
        "morph_kernel": 3,
        "white_v_min": 170,
        "white_s_max": 70,
        "dark_v_max": 80,
        "green_h_min": 35,
        "green_h_max": 100,
        "green_s_min": 40,
        "white_component_min_area": 6,
        "side_split_gap_px": 2,
        "side_center_white_ratio_max": 0.34,
        "side_center_green_ratio_max": 0.28,
        "side_center_vertical_aspect_min": 1.5,
        "side_white_big_count_min": 4,
        "side_top_white_count_min": 1,
        "side_bottom_white_count_min": 1,
        "center_x_min_ratio": 0.35,
        "center_x_max_ratio": 0.65,
        "center_y_min_ratio": 0.18,
        "center_y_max_ratio": 0.82,
        "center_white_ratio_max": 0.16,
        "white_big_count_min": 12,
        "center_dark_ratio_min": 0.34,
        "black_ratio_min": 0.05,
        "largest_component_area_min": 100,
        "bbox_width_min": 8,
        "bbox_height_min": 6,
        "elongation_min": 1.1,
        "elongation_max": 6.0,
        "fill_ratio_min": 0.20,
        "contrast_component_delta_min": 20.0,
        "min_pass_count": 4,
    },
}


WINDOWS_DRIVE_PATH_RE = re.compile(r"^[A-Za-z]:[\\/]")
WSL_MOUNT_PATH_RE = re.compile(r"^/mnt/([A-Za-z])(?:/(.*))?$")


@dataclass
class AnchorFeatures:
    name: str
    polygon: np.ndarray
    keypoints: list[Any]
    descriptors: np.ndarray


@dataclass
class TemplateState:
    image_path: Path
    image_bgr: np.ndarray
    gray: np.ndarray
    detect_roi: np.ndarray
    anchors: list[AnchorFeatures]


def require_yaml_support():
    if yaml is None:
        raise SystemExit(
            "Missing dependency: PyYAML. Install requirements.txt before using YAML configs."
        )
    return yaml


def current_platform_name() -> str:
    return "windows" if os.name == "nt" else "linux"


def current_platform_path_keys() -> tuple[str, ...]:
    if os.name == "nt":
        return ("windows", "win", "win32", "nt")
    return ("linux", "wsl", "posix", "unix")


def looks_like_windows_drive_path(path_text: str) -> bool:
    return WINDOWS_DRIVE_PATH_RE.match(path_text) is not None


def windows_drive_to_wsl_path(path_text: str) -> str | None:
    if not looks_like_windows_drive_path(path_text):
        return None

    windows_path = PureWindowsPath(path_text)
    drive_letter = windows_path.drive.rstrip(":").lower()
    tail_parts = [part for part in windows_path.parts[1:] if part not in {"\\", "/"}]
    return str(PurePosixPath("/mnt", drive_letter, *tail_parts))


def wsl_mount_to_windows_path(path_text: str) -> str | None:
    match = WSL_MOUNT_PATH_RE.match(path_text)
    if match is None:
        return None

    drive_letter = match.group(1).upper()
    tail = match.group(2) or ""
    if not tail:
        return f"{drive_letter}:\\"

    windows_tail = tail.replace("/", "\\")
    return f"{drive_letter}:\\{windows_tail}"


def infer_counterpart_path_text(path_text: str) -> str | None:
    return windows_drive_to_wsl_path(path_text) or wsl_mount_to_windows_path(path_text)


def translate_foreign_absolute_path(path_text: str) -> str | None:
    if os.name == "nt":
        translated = wsl_mount_to_windows_path(path_text)
        if translated is not None:
            return translated
        if path_text.startswith("//"):
            return path_text.replace("/", "\\")
        return None

    translated = windows_drive_to_wsl_path(path_text)
    if translated is not None:
        return translated
    if path_text.startswith("\\\\"):
        return path_text.replace("\\", "/")
    return None


def select_platform_path_value(path_value: Any, field_name: str) -> str:
    if isinstance(path_value, str):
        if not path_value.strip():
            raise ValueError(f"{field_name} must not be empty")
        return path_value

    if isinstance(path_value, dict):
        normalized_entries = {
            str(key).strip().lower(): value
            for key, value in path_value.items()
            if isinstance(value, str) and value.strip()
        }
        if not normalized_entries:
            raise ValueError(f"{field_name} must contain at least one non-empty string path")

        for key in (*current_platform_path_keys(), "default", "common", "shared", "path"):
            candidate = normalized_entries.get(key)
            if candidate:
                return candidate

        return next(iter(normalized_entries.values()))

    raise ValueError(f"{field_name} must be a string path or a mapping with platform keys")


def path_text_candidates(path_text: str) -> list[str]:
    expanded = os.path.expandvars(os.path.expanduser(path_text.strip()))
    translated = translate_foreign_absolute_path(expanded)

    candidates: list[str] = []
    for candidate in (translated, expanded) if translated else (expanded,):
        if candidate and candidate not in candidates:
            candidates.append(candidate)
    return candidates


def looks_like_absolute_path_text(path_text: str) -> bool:
    return (
        Path(path_text).is_absolute()
        or looks_like_windows_drive_path(path_text)
        or WSL_MOUNT_PATH_RE.match(path_text) is not None
        or path_text.startswith(("\\\\", "//"))
    )


def resolve_path_argument(path_value: str, base_dir: Path | None = None) -> Path:
    candidates: list[Path] = []
    for candidate_text in path_text_candidates(path_value):
        candidate_path = Path(candidate_text)
        if candidate_path.is_absolute():
            candidates.append(candidate_path.resolve())
            continue

        if looks_like_absolute_path_text(candidate_text):
            candidates.append(candidate_path)
            continue

        if base_dir is not None:
            candidates.append((base_dir / candidate_path).resolve())
        else:
            candidates.append(candidate_path.resolve())

    if not candidates:
        raise ValueError("path must not be empty")

    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()

    for candidate in candidates:
        if candidate.parent.exists():
            return candidate

    return candidates[0]


def resolve_config_path_value(path_value: Any, config_path: Path, field_name: str) -> Path:
    selected_path = select_platform_path_value(path_value, field_name)
    return resolve_path_argument(selected_path, config_path.parent)


def build_platform_path_value(path: Path, existing_value: Any = None) -> str | dict[str, str]:
    serialized = str(path)
    existing_entries: dict[str, str] = {}
    if isinstance(existing_value, dict):
        existing_entries = {
            str(key): str(value)
            for key, value in existing_value.items()
            if isinstance(value, str) and value.strip()
        }

    counterpart = infer_counterpart_path_text(serialized)
    if not existing_entries and counterpart is None:
        return serialized

    merged = dict(existing_entries)
    merged[current_platform_name()] = serialized

    other_platform = "linux" if current_platform_name() == "windows" else "windows"
    if counterpart is not None:
        merged.setdefault(other_platform, counterpart)
    return merged


def load_raw_config_file(config_path: Path) -> dict[str, Any]:
    suffix = config_path.suffix.lower()
    text = config_path.read_text(encoding="utf-8")

    if suffix == ".json":
        raw_config = json.loads(text)
    elif suffix in {".yaml", ".yml"}:
        yaml_module = require_yaml_support()
        raw_config = yaml_module.safe_load(text)
    else:
        try:
            raw_config = json.loads(text)
        except json.JSONDecodeError:
            yaml_module = require_yaml_support()
            raw_config = yaml_module.safe_load(text)

    if raw_config is None:
        return {}
    if not isinstance(raw_config, dict):
        raise ValueError("Config root must be a JSON/YAML object")
    return raw_config


def render_commented_config_yaml(config: dict[str, Any]) -> str:
    yaml_module = require_yaml_support()
    dumped = yaml_module.safe_dump(config, sort_keys=False, allow_unicode=True)
    lines = dumped.splitlines()

    comment_map: list[tuple[int, str, list[str]]] = [
        (
            0,
            "orb:",
            [
                "# PCB 电阻存在性检测配置。",
                "# 主配置改成 YAML，方便直接在参数旁边写中文调参说明。",
                "# 浏览器标注工具保存时也会按同样的带注释 YAML 模板回写。",
                "",
                "# ORB 特征提取，用于锚点配准。",
            ],
        ),
        (
            2,
            "nfeatures:",
            [
                "  # 锚点经常匹配不上时增大；运行太慢时再适当减小。",
            ],
        ),
        (
            0,
            "matching:",
            [
                "",
                "# 配准阶段阈值。",
            ],
        ),
        (
            2,
            "ratio_test:",
            [
                "  # 越小越严格。好图经常报 not_enough_anchor_matches 时再略微调大。",
            ],
        ),
        (
            2,
            "min_total_matches:",
            [
                "  # 进入几何变换估计前，全部锚点累计至少要有多少个好匹配。",
            ],
        ),
        (
            2,
            "min_inliers:",
            [
                "  # 几何内点数门槛。只有配准基本正确但经常失败时再适当调低。",
            ],
        ),
        (
            0,
            "preprocess:",
            [
                "",
                "# 配准和 ROI 分析之前的图像预处理。",
            ],
        ),
        (
            2,
            "standardize_to_template:",
            [
                "  # 手机图分辨率不一致时建议保持开启。会先缩放再 pad 到模板画布。",
            ],
        ),
        (
            0,
            "decision:",
            [
                "",
                "# ROI 判定规则，以及 03_binary_mask 的黑色候选阈值。",
                "# paired_resistors_lr 表示左、右两组分别判定，只要有一组不合格就整图 NG。",
            ],
        ),
        (
            2,
            "fixed_black_threshold:",
            [
                "  # 03_binary_mask 使用的黑色阈值上限。",
                "  # 越大越容易把像素刷成白色黑块候选。",
            ],
        ),
        (
            2,
            "local_dark_offset:",
            [
                "  # dynamic_black_threshold 大致等于 ring_mean - local_dark_offset。",
                "  # 如果电阻黑体在 03_binary_mask 里太碎，就先减 5。",
                "  # 如果绿板、阴影、背景也被大量刷白，就先加 5。",
            ],
        ),
        (
            2,
            "morph_kernel:",
            [
                "  # 03_binary_mask 的形态学清理核。越大去噪越强，但也更容易吃掉细长黑块。",
            ],
        ),
        (
            2,
            "side_center_white_ratio_max:",
            [
                "  # 每一侧中间区域允许暴露的白焊锡上限。",
                "  # 良品老是因为白焊锡过多被判 NG 时，再略微调大。",
            ],
        ),
        (
            2,
            "side_center_green_ratio_max:",
            [
                "  # 每一侧中间区域允许暴露的绿板上限。",
                "  # 只有确认良品本来就会露出更多绿板时再调大。",
            ],
        ),
        (
            2,
            "side_center_vertical_aspect_min:",
            [
                "  # 每一侧中间黑块至少要足够竖直细长。",
                "  # 越大越严格；当前加这条是为了避免 NG 某一侧被误放成 OK。",
            ],
        ),
        (
            2,
            "side_white_big_count_min:",
            [
                "  # 每一侧至少要看到多少个面积足够大的白焊锡连通域。",
            ],
        ),
        (
            0,
            "template_image:",
            [
                "",
                "# 模板图，以及模板坐标系下的标注区域。",
            ],
        ),
        (
            0,
            "anchors:",
            [
                "",
                "# 锚点框尽量放在稳定、纹理明显、每张图都能看到的 PCB 区域上。",
            ],
        ),
        (
            0,
            "detect_roi:",
            [
                "",
                "# detect_roi 同时覆盖左、右两组电阻。",
            ],
        ),
    ]

    rendered: list[str] = []
    for line in lines:
        stripped = line.lstrip()
        indent = len(line) - len(stripped)
        for comment_indent, key_prefix, comments in comment_map:
            if indent == comment_indent and stripped.startswith(key_prefix):
                rendered.extend(comments)
        rendered.append(line)
    return "\n".join(rendered) + "\n"


def write_config_file(config_path: Path, config: dict[str, Any]) -> None:
    suffix = config_path.suffix.lower()
    if suffix in {".yaml", ".yml"}:
        config_path.write_text(render_commented_config_yaml(config), encoding="utf-8")
        return

    if suffix == ".json":
        with config_path.open("w", encoding="utf-8") as handle:
            json.dump(config, handle, indent=2, ensure_ascii=False)
        return

    raise ValueError("Config path must end with .json, .yaml, or .yml")


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_config(config_path: Path) -> dict[str, Any]:
    raw_config = load_raw_config_file(config_path)
    config = deep_merge(DEFAULT_CONFIG, raw_config)
    if "template_image" not in config:
        raise ValueError("config.template_image is required")
    select_platform_path_value(config["template_image"], "config.template_image")
    if not config.get("anchors"):
        raise ValueError("config.anchors must contain at least two anchor regions")
    if len(config["anchors"]) < 2:
        raise ValueError("At least two anchor regions are required")
    if "detect_roi" not in config:
        raise ValueError("config.detect_roi is required")
    return config


def resolve_region_polygon(region: dict[str, Any]) -> np.ndarray:
    if "rect" in region:
        x, y, width, height = region["rect"]
        points = np.array(
            [[x, y], [x + width, y], [x + width, y + height], [x, y + height]],
            dtype=np.float32,
        )
    elif "polygon" in region:
        points = np.array(region["polygon"], dtype=np.float32)
    else:
        raise ValueError("Each region must define either 'rect' or 'polygon'")

    if points.ndim != 2 or points.shape[1] != 2 or len(points) < 3:
        raise ValueError("Region polygon must be an array of at least three [x, y] points")
    return points


def clip_polygon(points: np.ndarray, image_shape: Sequence[int]) -> np.ndarray:
    height, width = image_shape[:2]
    clipped = points.copy()
    clipped[:, 0] = np.clip(clipped[:, 0], 0, width - 1)
    clipped[:, 1] = np.clip(clipped[:, 1], 0, height - 1)
    return clipped


def polygon_mask(image_shape: Sequence[int], polygon: np.ndarray) -> np.ndarray:
    mask = np.zeros(image_shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask, [np.round(polygon).astype(np.int32)], 255)
    return mask


def read_image_bgr(image_path: Path, honor_exif_orientation: bool) -> np.ndarray:
    with Image.open(image_path) as image:
        if honor_exif_orientation:
            image = ImageOps.exif_transpose(image)
        image_rgb = image.convert("RGB")
        image_array = np.asarray(image_rgb)
    return cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)


def border_median_color(image_bgr: np.ndarray) -> np.ndarray:
    top = image_bgr[0, :, :]
    bottom = image_bgr[-1, :, :]
    left = image_bgr[:, 0, :]
    right = image_bgr[:, -1, :]
    border_pixels = np.concatenate([top, bottom, left, right], axis=0)
    return np.median(border_pixels, axis=0).astype(np.uint8)


def standardize_image_to_shape(
    image_bgr: np.ndarray,
    target_shape: Sequence[int],
    preprocess_cfg: dict[str, Any],
) -> np.ndarray:
    if not bool(preprocess_cfg.get("standardize_to_template", False)):
        return image_bgr

    target_height, target_width = target_shape[:2]
    source_height, source_width = image_bgr.shape[:2]
    if (source_height, source_width) == (target_height, target_width):
        return image_bgr

    mode = str(preprocess_cfg.get("standardize_mode", "fit_pad"))
    interpolation = cv2.INTER_AREA if source_width > target_width or source_height > target_height else cv2.INTER_LINEAR

    if mode == "stretch":
        return cv2.resize(image_bgr, (target_width, target_height), interpolation=interpolation)

    if mode != "fit_pad":
        raise ValueError("preprocess.standardize_mode must be one of fit_pad or stretch")

    scale = min(target_width / source_width, target_height / source_height)
    resized_width = max(1, int(round(source_width * scale)))
    resized_height = max(1, int(round(source_height * scale)))
    resized = cv2.resize(image_bgr, (resized_width, resized_height), interpolation=interpolation)

    canvas = np.full(
        (target_height, target_width, 3),
        border_median_color(image_bgr),
        dtype=np.uint8,
    )
    x_offset = (target_width - resized_width) // 2
    y_offset = (target_height - resized_height) // 2
    canvas[y_offset:y_offset + resized_height, x_offset:x_offset + resized_width] = resized
    return canvas


def preprocess_gray(image_bgr: np.ndarray, preprocess_cfg: dict[str, Any]) -> np.ndarray:
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

    blur_ksize = int(preprocess_cfg.get("gaussian_blur_ksize", 0))
    if blur_ksize >= 3 and blur_ksize % 2 == 1:
        gray = cv2.GaussianBlur(gray, (blur_ksize, blur_ksize), 0)

    clip_limit = float(preprocess_cfg.get("clahe_clip_limit", 0))
    if clip_limit > 0:
        tile_size = int(preprocess_cfg.get("clahe_tile_grid_size", 8))
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_size, tile_size))
        gray = clahe.apply(gray)

    return gray


def build_orb(orb_cfg: dict[str, Any]) -> cv2.ORB:
    return cv2.ORB_create(
        nfeatures=int(orb_cfg.get("nfeatures", 2500)),
        scaleFactor=float(orb_cfg.get("scaleFactor", 1.2)),
        nlevels=int(orb_cfg.get("nlevels", 8)),
        edgeThreshold=int(orb_cfg.get("edgeThreshold", 31)),
        fastThreshold=int(orb_cfg.get("fastThreshold", 12)),
    )


def load_template_state(config: dict[str, Any], config_path: Path) -> TemplateState:
    template_path = resolve_config_path_value(
        config["template_image"],
        config_path,
        "config.template_image",
    )

    honor_exif_orientation = bool(config["preprocess"].get("honor_exif_orientation", True))
    template_image = read_image_bgr(template_path, honor_exif_orientation)

    template_gray = preprocess_gray(template_image, config["preprocess"])
    orb = build_orb(config["orb"])

    anchors: list[AnchorFeatures] = []
    for index, anchor_cfg in enumerate(config["anchors"], start=1):
        polygon = clip_polygon(resolve_region_polygon(anchor_cfg), template_image.shape)
        mask = polygon_mask(template_image.shape, polygon)
        keypoints, descriptors = orb.detectAndCompute(template_gray, mask)
        if descriptors is None or len(keypoints) == 0:
            raise ValueError(
                f"Anchor '{anchor_cfg.get('name', f'anchor_{index}')}' has no ORB features. "
                "Pick a more distinctive region."
            )
        anchors.append(
            AnchorFeatures(
                name=anchor_cfg.get("name", f"anchor_{index}"),
                polygon=polygon,
                keypoints=keypoints,
                descriptors=descriptors,
            )
        )

    detect_roi = clip_polygon(resolve_region_polygon(config["detect_roi"]), template_image.shape)
    return TemplateState(
        image_path=template_path,
        image_bgr=template_image,
        gray=template_gray,
        detect_roi=detect_roi,
        anchors=anchors,
    )


def match_anchor_descriptors(
    matcher: cv2.BFMatcher,
    anchor: AnchorFeatures,
    test_keypoints: list[Any],
    test_descriptors: np.ndarray,
    matching_cfg: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    ratio_test = float(matching_cfg["ratio_test"])
    max_matches = int(matching_cfg.get("max_matches_per_anchor", 80))
    min_anchor_matches = int(matching_cfg["min_anchor_matches"])

    raw_matches = matcher.knnMatch(anchor.descriptors, test_descriptors, k=2)
    good_matches = []
    for pair in raw_matches:
        if len(pair) != 2:
            continue
        best, second = pair
        if best.distance < ratio_test * second.distance:
            good_matches.append(best)

    good_matches.sort(key=lambda item: item.distance)
    if max_matches > 0:
        good_matches = good_matches[:max_matches]

    stats = {
        "anchor_name": anchor.name,
        "template_keypoints": len(anchor.keypoints),
        "good_matches": len(good_matches),
        "match_ok": len(good_matches) >= min_anchor_matches,
    }

    if len(good_matches) < min_anchor_matches:
        return np.empty((0, 2), dtype=np.float32), np.empty((0, 2), dtype=np.float32), stats

    src_points = np.float32([anchor.keypoints[m.queryIdx].pt for m in good_matches])
    dst_points = np.float32([test_keypoints[m.trainIdx].pt for m in good_matches])
    return src_points, dst_points, stats


def estimate_transform(
    src_points: np.ndarray,
    dst_points: np.ndarray,
    matching_cfg: dict[str, Any],
) -> tuple[np.ndarray | None, np.ndarray | None]:
    transform_type = matching_cfg["transform_type"]
    ransac_threshold = float(matching_cfg["ransac_reproj_threshold"])

    if transform_type == "affine_partial":
        return cv2.estimateAffinePartial2D(
            src_points,
            dst_points,
            method=cv2.RANSAC,
            ransacReprojThreshold=ransac_threshold,
            maxIters=2000,
            refineIters=20,
            confidence=0.99,
        )
    if transform_type == "affine":
        return cv2.estimateAffine2D(
            src_points,
            dst_points,
            method=cv2.RANSAC,
            ransacReprojThreshold=ransac_threshold,
            maxIters=2000,
            refineIters=20,
            confidence=0.99,
        )
    if transform_type == "homography":
        matrix, inliers = cv2.findHomography(
            src_points,
            dst_points,
            method=cv2.RANSAC,
            ransacReprojThreshold=ransac_threshold,
            confidence=0.99,
            maxIters=2000,
        )
        return matrix, inliers
    raise ValueError("matching.transform_type must be one of affine_partial, affine, homography")


def apply_transform(points: np.ndarray, matrix: np.ndarray, transform_type: str) -> np.ndarray:
    points = np.asarray(points, dtype=np.float32)
    if transform_type in {"affine_partial", "affine"}:
        reshaped = points.reshape(-1, 1, 2)
        transformed = cv2.transform(reshaped, matrix)
        return transformed.reshape(-1, 2)

    reshaped = points.reshape(-1, 1, 2)
    transformed = cv2.perspectiveTransform(reshaped, matrix)
    return transformed.reshape(-1, 2)


def warp_to_template(
    image_bgr: np.ndarray,
    matrix: np.ndarray,
    transform_type: str,
    template_shape: Sequence[int],
) -> np.ndarray:
    height, width = template_shape[:2]
    if transform_type in {"affine_partial", "affine"}:
        inverse_matrix = cv2.invertAffineTransform(matrix)
        return cv2.warpAffine(
            image_bgr,
            inverse_matrix,
            (width, height),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE,
        )

    inverse_matrix = np.linalg.inv(matrix)
    return cv2.warpPerspective(
        image_bgr,
        inverse_matrix,
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )


def crop_polygon_region(
    image_bgr: np.ndarray,
    polygon: np.ndarray,
    margin_px: int,
) -> tuple[np.ndarray, np.ndarray, tuple[int, int, int, int]]:
    x, y, width, height = cv2.boundingRect(np.round(polygon).astype(np.int32))
    x0 = max(0, x - margin_px)
    y0 = max(0, y - margin_px)
    x1 = min(image_bgr.shape[1], x + width + margin_px)
    y1 = min(image_bgr.shape[0], y + height + margin_px)

    crop = image_bgr[y0:y1, x0:x1].copy()
    local_polygon = polygon - np.array([x0, y0], dtype=np.float32)
    mask = polygon_mask(crop.shape, local_polygon)
    return crop, mask, (x0, y0, x1 - x0, y1 - y0)


def connected_component_metrics(mask: np.ndarray, min_area: int) -> dict[str, Any]:
    binary_mask = mask.astype(np.uint8) * 255
    num_labels, _, stats, _ = cv2.connectedComponentsWithStats(binary_mask, connectivity=8)
    if num_labels <= 1:
        return {
            "largest_area": 0,
            "largest_width": 0,
            "largest_height": 0,
            "count_big": 0,
        }

    areas = [int(stats[index, cv2.CC_STAT_AREA]) for index in range(1, num_labels)]
    largest_index = 1 + int(np.argmax(areas))
    return {
        "largest_area": int(stats[largest_index, cv2.CC_STAT_AREA]),
        "largest_width": int(stats[largest_index, cv2.CC_STAT_WIDTH]),
        "largest_height": int(stats[largest_index, cv2.CC_STAT_HEIGHT]),
        "count_big": sum(area >= min_area for area in areas),
    }


def mask_bounds(valid_mask: np.ndarray) -> tuple[int, int, int, int] | None:
    ys, xs = np.where(valid_mask)
    if xs.size == 0 or ys.size == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def build_relative_band(
    valid_mask: np.ndarray,
    bounds: tuple[int, int, int, int],
    x_min_ratio: float,
    x_max_ratio: float,
    y_min_ratio: float,
    y_max_ratio: float,
) -> np.ndarray:
    x0, y0, x1, y1 = bounds
    width = max(1, x1 - x0)
    height = max(1, y1 - y0)

    band_x0 = x0 + int(round(width * x_min_ratio))
    band_x1 = x0 + int(round(width * x_max_ratio))
    band_y0 = y0 + int(round(height * y_min_ratio))
    band_y1 = y0 + int(round(height * y_max_ratio))

    band_x0 = max(x0, min(band_x0, x1 - 1))
    band_x1 = max(band_x0 + 1, min(band_x1, x1))
    band_y0 = max(y0, min(band_y0, y1 - 1))
    band_y1 = max(band_y0 + 1, min(band_y1, y1))

    band = np.zeros_like(valid_mask, dtype=bool)
    band[band_y0:band_y1, band_x0:band_x1] = True
    return band & valid_mask


def extract_layout_features(
    roi_bgr: np.ndarray,
    roi_mask: np.ndarray,
    decision_cfg: dict[str, Any],
    *,
    relative_to_valid_bbox: bool = False,
) -> dict[str, Any]:
    valid = roi_mask > 0
    bounds = mask_bounds(valid)
    if bounds is None:
        return {
            "white_ratio": 0.0,
            "dark_ratio": 0.0,
            "green_ratio": 0.0,
            "center_dark_ratio": 0.0,
            "center_green_ratio": 0.0,
            "center_white_ratio": 0.0,
            "white_big_count": 0,
            "top_white_count": 0,
            "bottom_white_count": 0,
            "center_dark_area": 0,
            "center_dark_w": 0,
            "center_dark_h": 0,
            "center_vertical_aspect": 0.0,
        }

    hsv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)

    h = hsv[:, :, 0]
    s = hsv[:, :, 1]
    v = hsv[:, :, 2]

    white_mask = valid & (v >= int(decision_cfg["white_v_min"])) & (s <= int(decision_cfg["white_s_max"]))
    dark_mask = valid & (v <= int(decision_cfg["dark_v_max"]))
    green_mask = (
        valid
        & (h >= int(decision_cfg["green_h_min"]))
        & (h <= int(decision_cfg["green_h_max"]))
        & (s >= int(decision_cfg["green_s_min"]))
    )

    if relative_to_valid_bbox:
        center_band = build_relative_band(
            valid,
            bounds,
            float(decision_cfg["center_x_min_ratio"]),
            float(decision_cfg["center_x_max_ratio"]),
            float(decision_cfg["center_y_min_ratio"]),
            float(decision_cfg["center_y_max_ratio"]),
        )
        top_band = build_relative_band(valid, bounds, 0.15, 0.85, 0.05, 0.38)
        bottom_band = build_relative_band(valid, bounds, 0.15, 0.85, 0.62, 0.95)
    else:
        height, width = valid.shape
        center_band = np.zeros_like(valid, dtype=bool)
        x0 = int(width * float(decision_cfg["center_x_min_ratio"]))
        x1 = int(width * float(decision_cfg["center_x_max_ratio"]))
        y0 = int(height * float(decision_cfg["center_y_min_ratio"]))
        y1 = int(height * float(decision_cfg["center_y_max_ratio"]))
        center_band[y0:y1, x0:x1] = True
        center_band &= valid

        top_band = np.zeros_like(valid, dtype=bool)
        bottom_band = np.zeros_like(valid, dtype=bool)
        top_band[int(height * 0.05):int(height * 0.38), int(width * 0.15):int(width * 0.85)] = True
        bottom_band[int(height * 0.62):int(height * 0.95), int(width * 0.15):int(width * 0.85)] = True
        top_band &= valid
        bottom_band &= valid

    component_min_area = int(decision_cfg["white_component_min_area"])
    white_metrics = connected_component_metrics(white_mask, component_min_area)
    top_white_metrics = connected_component_metrics(white_mask & top_band, component_min_area)
    bottom_white_metrics = connected_component_metrics(white_mask & bottom_band, component_min_area)
    center_dark_metrics = connected_component_metrics(dark_mask & center_band, component_min_area)

    feature_payload = {
        "white_ratio": round(float(np.count_nonzero(white_mask)) / max(1, int(np.count_nonzero(valid))), 6),
        "dark_ratio": round(float(np.count_nonzero(dark_mask)) / max(1, int(np.count_nonzero(valid))), 6),
        "green_ratio": round(float(np.count_nonzero(green_mask)) / max(1, int(np.count_nonzero(valid))), 6),
        "center_dark_ratio": round(
            float(np.count_nonzero(dark_mask & center_band)) / max(1, int(np.count_nonzero(center_band))),
            6,
        ),
        "center_green_ratio": round(
            float(np.count_nonzero(green_mask & center_band)) / max(1, int(np.count_nonzero(center_band))),
            6,
        ),
        "center_white_ratio": round(
            float(np.count_nonzero(white_mask & center_band)) / max(1, int(np.count_nonzero(center_band))),
            6,
        ),
        "white_big_count": int(white_metrics["count_big"]),
        "top_white_count": int(top_white_metrics["count_big"]),
        "bottom_white_count": int(bottom_white_metrics["count_big"]),
        "center_dark_area": int(center_dark_metrics["largest_area"]),
        "center_dark_w": int(center_dark_metrics["largest_width"]),
        "center_dark_h": int(center_dark_metrics["largest_height"]),
        "center_vertical_aspect": round(
            float(center_dark_metrics["largest_height"]) / max(1, int(center_dark_metrics["largest_width"])),
            6,
        ) if center_dark_metrics["largest_area"] > 0 else 0.0,
    }
    return feature_payload


def split_roi_side_masks(roi_mask: np.ndarray, gap_px: int) -> dict[str, np.ndarray]:
    valid = roi_mask > 0
    bounds = mask_bounds(valid)
    empty_mask = np.zeros_like(roi_mask, dtype=np.uint8)
    if bounds is None:
        return {"left": empty_mask.copy(), "right": empty_mask.copy()}

    x0, _, x1, _ = bounds
    split_x = x0 + (x1 - x0) // 2
    left_gap = max(0, int(gap_px) // 2)
    right_gap = max(0, int(gap_px) - left_gap)

    x_coords = np.arange(roi_mask.shape[1], dtype=np.int32)[None, :]
    left_mask = valid & (x_coords < split_x - left_gap)
    right_mask = valid & (x_coords >= split_x + right_gap)
    return {
        "left": left_mask.astype(np.uint8) * 255,
        "right": right_mask.astype(np.uint8) * 255,
    }


def analyze_side_layout(
    side_name: str,
    roi_bgr: np.ndarray,
    side_mask: np.ndarray,
    decision_cfg: dict[str, Any],
) -> dict[str, Any]:
    valid = side_mask > 0
    if not np.any(valid):
        return {
            "name": side_name,
            "result": "NG",
            "decision_reason": "empty_side_mask",
            "failed_rules": ["empty_side_mask"],
            "layout_features": extract_layout_features(
                roi_bgr,
                side_mask,
                decision_cfg,
                relative_to_valid_bbox=True,
            ),
        }

    layout_features = extract_layout_features(
        roi_bgr,
        side_mask,
        decision_cfg,
        relative_to_valid_bbox=True,
    )
    rules = {
        "center_white_ratio_ok": layout_features["center_white_ratio"] <= float(decision_cfg["side_center_white_ratio_max"]),
        "center_green_ratio_ok": layout_features["center_green_ratio"] <= float(decision_cfg["side_center_green_ratio_max"]),
        "center_vertical_aspect_ok": layout_features["center_vertical_aspect"] >= float(decision_cfg["side_center_vertical_aspect_min"]),
        "white_big_count_enough": layout_features["white_big_count"] >= int(decision_cfg["side_white_big_count_min"]),
        "top_white_count_enough": layout_features["top_white_count"] >= int(decision_cfg["side_top_white_count_min"]),
        "bottom_white_count_enough": layout_features["bottom_white_count"] >= int(decision_cfg["side_bottom_white_count_min"]),
    }
    failed_reason_map = {
        "center_white_ratio_ok": "center_white_ratio_high",
        "center_green_ratio_ok": "center_green_ratio_high",
        "center_vertical_aspect_ok": "center_vertical_aspect_low",
        "white_big_count_enough": "white_big_count_low",
        "top_white_count_enough": "top_white_count_low",
        "bottom_white_count_enough": "bottom_white_count_low",
    }
    failed_rule_keys = [name for name, passed in rules.items() if not passed]
    failed_rules = [failed_reason_map[name] for name in failed_rule_keys]
    return {
        "name": side_name,
        "result": "OK" if not failed_rules else "NG",
        "decision_reason": "side_ok" if not failed_rules else failed_rules[0],
        "failed_rules": failed_rules,
        "layout_features": layout_features,
        "rules": rules,
    }


def analyze_roi(
    roi_bgr: np.ndarray,
    roi_mask: np.ndarray,
    decision_cfg: dict[str, Any],
) -> dict[str, Any]:
    gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
    roi_pixels = gray[roi_mask > 0]
    if roi_pixels.size == 0:
        raise ValueError("Detect ROI is empty after clipping")

    ring_margin = int(decision_cfg["ring_margin_px"])
    morph_kernel = int(decision_cfg["morph_kernel"])
    fixed_black_threshold = int(decision_cfg["fixed_black_threshold"])
    min_black_threshold = int(decision_cfg["min_black_threshold"])
    local_dark_offset = int(decision_cfg["local_dark_offset"])

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (max(3, ring_margin), max(3, ring_margin)))
    dilated_mask = cv2.dilate(roi_mask, kernel, iterations=1)
    ring_mask = cv2.subtract(dilated_mask, roi_mask)
    ring_pixels = gray[ring_mask > 0]

    ring_mean = float(np.mean(ring_pixels)) if ring_pixels.size else float(np.mean(roi_pixels))
    roi_mean = float(np.mean(roi_pixels))
    dynamic_black_threshold = int(
        np.clip(
            min(fixed_black_threshold, ring_mean - local_dark_offset),
            min_black_threshold,
            fixed_black_threshold,
        )
    )

    binary_mask = np.zeros_like(roi_mask)
    binary_mask[(gray < dynamic_black_threshold) & (roi_mask > 0)] = 255

    if morph_kernel >= 3 and morph_kernel % 2 == 1:
        morph = cv2.getStructuringElement(cv2.MORPH_RECT, (morph_kernel, morph_kernel))
        binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, morph)
        binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, morph)
        binary_mask[roi_mask == 0] = 0

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary_mask, connectivity=8)

    largest_area = 0
    largest_bbox = (0, 0, 0, 0)
    largest_label = 0
    for label_index in range(1, num_labels):
        area = int(stats[label_index, cv2.CC_STAT_AREA])
        if area > largest_area:
            largest_area = area
            largest_label = label_index
            largest_bbox = (
                int(stats[label_index, cv2.CC_STAT_LEFT]),
                int(stats[label_index, cv2.CC_STAT_TOP]),
                int(stats[label_index, cv2.CC_STAT_WIDTH]),
                int(stats[label_index, cv2.CC_STAT_HEIGHT]),
            )

    roi_area = int(np.count_nonzero(roi_mask))
    black_pixels = int(np.count_nonzero(binary_mask))
    black_ratio = black_pixels / max(roi_area, 1)

    bbox_width = largest_bbox[2]
    bbox_height = largest_bbox[3]
    longer_side = max(bbox_width, bbox_height)
    shorter_side = max(1, min(bbox_width, bbox_height))
    elongation = longer_side / shorter_side if largest_area > 0 else 0.0
    fill_ratio = largest_area / max(1, bbox_width * bbox_height)

    component_mask = (labels == largest_label) if largest_label > 0 else np.zeros_like(labels, dtype=bool)
    component_pixels = gray[component_mask]
    component_mean = float(np.mean(component_pixels)) if component_pixels.size else 255.0
    contrast_component_delta = ring_mean - component_mean

    layout_features = extract_layout_features(roi_bgr, roi_mask, decision_cfg)

    rules = {
        "black_ratio": black_ratio >= float(decision_cfg["black_ratio_min"]),
        "largest_component_area": largest_area >= int(decision_cfg["largest_component_area_min"]),
        "bbox_width": bbox_width >= int(decision_cfg["bbox_width_min"]),
        "bbox_height": bbox_height >= int(decision_cfg["bbox_height_min"]),
        "elongation": float(decision_cfg["elongation_min"]) <= elongation <= float(decision_cfg["elongation_max"]),
        "fill_ratio": fill_ratio >= float(decision_cfg["fill_ratio_min"]),
        "contrast_component_delta": contrast_component_delta >= float(decision_cfg["contrast_component_delta_min"]),
    }

    decision_mode = str(decision_cfg.get("mode", "legacy"))
    decision_reason = "legacy_thresholds"
    side_results: dict[str, Any] | None = None
    failed_sides: list[str] = []
    if decision_mode == "paired_resistors_lr":
        side_masks = split_roi_side_masks(roi_mask, int(decision_cfg.get("side_split_gap_px", 2)))
        side_results = {
            side_name: analyze_side_layout(side_name, roi_bgr, side_mask, decision_cfg)
            for side_name, side_mask in side_masks.items()
        }
        failed_sides = [side_name for side_name, side_result in side_results.items() if side_result["result"] != "OK"]
        result = "OK" if not failed_sides else "NG"
        if not failed_sides:
            decision_reason = "both_sides_ok"
        elif len(failed_sides) == 1:
            decision_reason = f"{failed_sides[0]}_reject"
        else:
            decision_reason = "left_right_reject"
        rules = {f"{side_name}_ok": side_result["result"] == "OK" for side_name, side_result in side_results.items()}
        pass_count = sum(1 for side_result in side_results.values() if side_result["result"] == "OK")
    elif decision_mode == "white_pad_dark_center":
        layout_rules = {
            "center_white_ratio_low": layout_features["center_white_ratio"] <= float(decision_cfg["center_white_ratio_max"]),
            "white_big_count_enough": layout_features["white_big_count"] >= int(decision_cfg["white_big_count_min"]),
            "center_dark_ratio_high": layout_features["center_dark_ratio"] >= float(decision_cfg["center_dark_ratio_min"]),
        }
        if layout_rules["center_white_ratio_low"]:
            result = "OK"
            decision_reason = "center_white_ratio_low"
        elif layout_rules["white_big_count_enough"] and layout_rules["center_dark_ratio_high"]:
            result = "OK"
            decision_reason = "white_and_dark_layout_ok"
        else:
            result = "NG"
            decision_reason = "white_pad_dark_center_reject"
        pass_count = sum(1 for value in layout_rules.values() if value)
        rules = layout_rules
    else:
        pass_count = sum(1 for value in rules.values() if value)
        result = "OK" if pass_count >= int(decision_cfg["min_pass_count"]) else "NG"

    payload = {
        "result": result,
        "decision_mode": decision_mode,
        "decision_reason": decision_reason,
        "roi_area": roi_area,
        "roi_mean": round(roi_mean, 3),
        "ring_mean": round(ring_mean, 3),
        "dynamic_black_threshold": dynamic_black_threshold,
        "black_pixels": black_pixels,
        "black_ratio": round(black_ratio, 6),
        "largest_component_area": largest_area,
        "largest_component_bbox": {
            "x": largest_bbox[0],
            "y": largest_bbox[1],
            "width": bbox_width,
            "height": bbox_height,
        },
        "elongation": round(elongation, 6),
        "fill_ratio": round(fill_ratio, 6),
        "component_mean": round(component_mean, 3),
        "contrast_component_delta": round(contrast_component_delta, 3),
        "layout_features": layout_features,
        "rules": rules,
        "pass_count": pass_count,
        "binary_mask": binary_mask,
    }
    if side_results is not None:
        payload["side_results"] = side_results
        payload["failed_sides"] = failed_sides
    return payload


def draw_polygon(
    image_bgr: np.ndarray,
    polygon: np.ndarray,
    color: tuple[int, int, int],
    label: str | None = None,
) -> None:
    contour = np.round(polygon).astype(np.int32).reshape(-1, 1, 2)
    cv2.polylines(image_bgr, [contour], isClosed=True, color=color, thickness=2)
    if label:
        anchor = tuple(contour[0, 0])
        cv2.putText(
            image_bgr,
            label,
            (int(anchor[0]), int(anchor[1]) - 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2,
            cv2.LINE_AA,
        )


def summarize_transform(matrix: np.ndarray, transform_type: str) -> dict[str, Any]:
    transform = np.asarray(matrix, dtype=np.float64)

    if transform_type == "affine_partial":
        scale = float(np.sqrt(transform[0, 0] ** 2 + transform[1, 0] ** 2))
        angle_deg = float(np.degrees(np.arctan2(transform[1, 0], transform[0, 0])))
        return {
            "scale": round(scale, 6),
            "angle_deg": round(angle_deg, 6),
            "tx": round(float(transform[0, 2]), 6),
            "ty": round(float(transform[1, 2]), 6),
        }

    if transform_type == "affine":
        scale_x = float(np.sqrt(transform[0, 0] ** 2 + transform[1, 0] ** 2))
        scale_y = float(np.sqrt(transform[0, 1] ** 2 + transform[1, 1] ** 2))
        angle_deg = float(np.degrees(np.arctan2(transform[1, 0], transform[0, 0])))
        return {
            "scale_x": round(scale_x, 6),
            "scale_y": round(scale_y, 6),
            "angle_deg": round(angle_deg, 6),
            "tx": round(float(transform[0, 2]), 6),
            "ty": round(float(transform[1, 2]), 6),
        }

    if transform_type == "homography":
        return {
            "h20": round(float(transform[2, 0]), 9),
            "h21": round(float(transform[2, 1]), 9),
            "h22": round(float(transform[2, 2]), 9),
        }

    return {}


def save_debug_outputs(
    debug_root: Path,
    image_path: Path,
    original_image: np.ndarray,
    projected_roi: np.ndarray,
    template_state: TemplateState,
    aligned_image: np.ndarray,
    roi_bbox: tuple[int, int, int, int],
    roi_analysis: dict[str, Any],
    transform_matrix: np.ndarray,
    transform_type: str,
) -> None:
    debug_dir = debug_root / image_path.stem
    debug_dir.mkdir(parents=True, exist_ok=True)

    template_overlay = template_state.image_bgr.copy()
    draw_polygon(template_overlay, template_state.detect_roi, (0, 255, 255), "detect_roi")
    for anchor in template_state.anchors:
        draw_polygon(template_overlay, anchor.polygon, (255, 0, 0), anchor.name)
    cv2.imwrite(str(debug_dir / "00_template_overlay.jpg"), template_overlay)

    original_overlay = original_image.copy()
    draw_polygon(original_overlay, projected_roi, (0, 255, 255), "detect_roi")
    for anchor in template_state.anchors:
        projected_anchor = apply_transform(anchor.polygon, transform_matrix, transform_type)
        draw_polygon(original_overlay, projected_anchor, (255, 0, 0), anchor.name)
    cv2.imwrite(str(debug_dir / "01_original_overlay.jpg"), original_overlay)

    aligned_overlay = aligned_image.copy()
    draw_polygon(aligned_overlay, template_state.detect_roi, (0, 255, 255), "detect_roi")
    x, y, width, height = roi_bbox
    cv2.rectangle(aligned_overlay, (x, y), (x + width, y + height), (0, 255, 0), 2)
    cv2.putText(
        aligned_overlay,
        f"result={roi_analysis['result']} black_ratio={roi_analysis['black_ratio']:.4f}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0) if roi_analysis["result"] == "OK" else (0, 0, 255),
        2,
        cv2.LINE_AA,
    )
    if "side_results" in roi_analysis:
        side_summary = " ".join(
            f"{side_name}={side_result['result']}" for side_name, side_result in roi_analysis["side_results"].items()
        )
        cv2.putText(
            aligned_overlay,
            side_summary,
            (20, 72),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 0),
            2,
            cv2.LINE_AA,
        )
    cv2.imwrite(str(debug_dir / "02_aligned_overlay.jpg"), aligned_overlay)
    cv2.imwrite(str(debug_dir / "03_binary_mask.png"), roi_analysis["binary_mask"])

    metrics_for_json = {key: value for key, value in roi_analysis.items() if key != "binary_mask"}
    metrics_for_json["transform_type"] = transform_type
    metrics_for_json["transform_matrix"] = np.asarray(transform_matrix).tolist()
    metrics_for_json["transform_summary"] = summarize_transform(transform_matrix, transform_type)
    with (debug_dir / "04_metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics_for_json, handle, indent=2, ensure_ascii=False)


def save_failure_debug_outputs(
    debug_root: Path,
    image_path: Path,
    original_image: np.ndarray,
    template_state: TemplateState,
    result: dict[str, Any],
    transform_matrix: np.ndarray | None = None,
    transform_type: str | None = None,
) -> None:
    debug_dir = debug_root / image_path.stem
    debug_dir.mkdir(parents=True, exist_ok=True)

    template_overlay = template_state.image_bgr.copy()
    draw_polygon(template_overlay, template_state.detect_roi, (0, 255, 255), "detect_roi")
    for anchor in template_state.anchors:
        draw_polygon(template_overlay, anchor.polygon, (255, 0, 0), anchor.name)
    cv2.imwrite(str(debug_dir / "00_template_overlay.jpg"), template_overlay)

    failure_overlay = original_image.copy()
    status_text = f"result={result.get('result', 'NG')} reason={result.get('reason', '-') }"
    cv2.putText(
        failure_overlay,
        status_text,
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 0, 255),
        2,
        cv2.LINE_AA,
    )

    if transform_matrix is not None and transform_type is not None:
        try:
            projected_roi = apply_transform(template_state.detect_roi, transform_matrix, transform_type)
            draw_polygon(failure_overlay, projected_roi, (0, 165, 255), "detect_roi_attempt")
            for anchor in template_state.anchors:
                projected_anchor = apply_transform(anchor.polygon, transform_matrix, transform_type)
                draw_polygon(failure_overlay, projected_anchor, (255, 0, 0), f"{anchor.name}_attempt")
        except Exception:
            pass

    cv2.imwrite(str(debug_dir / "01_original_failure.jpg"), failure_overlay)

    metrics_for_json = dict(result)
    with (debug_dir / "04_metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics_for_json, handle, indent=2, ensure_ascii=False)


def analyze_image(
    image_path: Path,
    config: dict[str, Any],
    template_state: TemplateState,
    debug_root: Path | None,
) -> dict[str, Any]:
    honor_exif_orientation = bool(config["preprocess"].get("honor_exif_orientation", True))
    raw_image_bgr = read_image_bgr(image_path, honor_exif_orientation)
    image_bgr = standardize_image_to_shape(raw_image_bgr, template_state.image_bgr.shape, config["preprocess"])

    test_gray = preprocess_gray(image_bgr, config["preprocess"])
    orb = build_orb(config["orb"])
    test_keypoints, test_descriptors = orb.detectAndCompute(test_gray, None)
    if test_descriptors is None or len(test_keypoints) == 0:
        result = {
            "image": str(image_path),
            "result": "NG",
            "reason": "no_features_in_test_image",
            "input_original_size": [raw_image_bgr.shape[1], raw_image_bgr.shape[0]],
            "input_working_size": [image_bgr.shape[1], image_bgr.shape[0]],
        }
        if debug_root is not None:
            save_failure_debug_outputs(debug_root, image_path, image_bgr, template_state, result)
        return result

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
    if total_good_matches < int(config["matching"]["min_total_matches"]):
        result = {
            "image": str(image_path),
            "result": "NG",
            "reason": "not_enough_anchor_matches",
            "total_good_matches": total_good_matches,
            "anchor_stats": anchor_stats,
            "input_original_size": [raw_image_bgr.shape[1], raw_image_bgr.shape[0]],
            "input_working_size": [image_bgr.shape[1], image_bgr.shape[0]],
        }
        if debug_root is not None:
            save_failure_debug_outputs(debug_root, image_path, image_bgr, template_state, result)
        return result

    src_points = np.concatenate(all_src_points, axis=0)
    dst_points = np.concatenate(all_dst_points, axis=0)
    transform_matrix, inliers = estimate_transform(src_points, dst_points, config["matching"])
    if transform_matrix is None or inliers is None:
        result = {
            "image": str(image_path),
            "result": "NG",
            "reason": "transform_estimation_failed",
            "total_good_matches": total_good_matches,
            "anchor_stats": anchor_stats,
            "input_original_size": [raw_image_bgr.shape[1], raw_image_bgr.shape[0]],
            "input_working_size": [image_bgr.shape[1], image_bgr.shape[0]],
        }
        if debug_root is not None:
            save_failure_debug_outputs(debug_root, image_path, image_bgr, template_state, result)
        return result

    inlier_count = int(np.count_nonzero(inliers))
    if inlier_count < int(config["matching"]["min_inliers"]):
        transform_type = config["matching"]["transform_type"]
        result = {
            "image": str(image_path),
            "result": "NG",
            "reason": "not_enough_inliers",
            "total_good_matches": total_good_matches,
            "inlier_count": inlier_count,
            "anchor_stats": anchor_stats,
            "transform_type": transform_type,
            "transform_matrix": np.asarray(transform_matrix).tolist(),
            "transform_summary": summarize_transform(transform_matrix, transform_type),
            "input_original_size": [raw_image_bgr.shape[1], raw_image_bgr.shape[0]],
            "input_working_size": [image_bgr.shape[1], image_bgr.shape[0]],
        }
        if debug_root is not None:
            save_failure_debug_outputs(
                debug_root,
                image_path,
                image_bgr,
                template_state,
                result,
                transform_matrix=transform_matrix,
                transform_type=transform_type,
            )
        return result

    transform_type = config["matching"]["transform_type"]
    aligned_image = warp_to_template(image_bgr, transform_matrix, transform_type, template_state.image_bgr.shape)
    roi_crop, roi_mask, roi_bbox = crop_polygon_region(
        aligned_image,
        template_state.detect_roi,
        margin_px=int(config["decision"].get("ring_margin_px", 10)),
    )
    roi_analysis = analyze_roi(roi_crop, roi_mask, config["decision"])
    projected_roi = apply_transform(template_state.detect_roi, transform_matrix, transform_type)

    result = {
        "image": str(image_path),
        "result": roi_analysis["result"],
        "reason": "rule_based_decision",
        "total_good_matches": total_good_matches,
        "inlier_count": inlier_count,
        "anchor_stats": anchor_stats,
        "transform_type": transform_type,
        "transform_matrix": np.asarray(transform_matrix).tolist(),
        "transform_summary": summarize_transform(transform_matrix, transform_type),
        "input_original_size": [raw_image_bgr.shape[1], raw_image_bgr.shape[0]],
        "input_working_size": [image_bgr.shape[1], image_bgr.shape[0]],
    }
    for key, value in roi_analysis.items():
        if key != "binary_mask":
            result[key] = value

    if debug_root is not None:
        save_debug_outputs(
            debug_root=debug_root,
            image_path=image_path,
            original_image=image_bgr,
            projected_roi=projected_roi,
            template_state=template_state,
            aligned_image=aligned_image,
            roi_bbox=roi_bbox,
            roi_analysis=roi_analysis,
            transform_matrix=transform_matrix,
            transform_type=transform_type,
        )

    return result


def format_result(result: dict[str, Any]) -> str:
    core = [
        f"image={result['image']}",
        f"result={result['result']}",
        f"reason={result.get('reason', '-')}",
    ]
    if "total_good_matches" in result:
        core.append(f"matches={result['total_good_matches']}")
    if "inlier_count" in result:
        core.append(f"inliers={result['inlier_count']}")
    if "black_ratio" in result:
        core.append(f"black_ratio={result['black_ratio']}")
    if "largest_component_area" in result:
        core.append(f"largest_area={result['largest_component_area']}")
    if "elongation" in result:
        core.append(f"elongation={result['elongation']}")
    if "dynamic_black_threshold" in result:
        core.append(f"black_thr={result['dynamic_black_threshold']}")
    if "pass_count" in result:
        core.append(f"pass_count={result['pass_count']}")
    if "side_results" in result:
        side_summary = ",".join(
            f"{side_name}:{side_result['result']}" for side_name, side_result in result["side_results"].items()
        )
        core.append(f"sides={side_summary}")
    if "transform_summary" in result and result["transform_summary"]:
        summary = result["transform_summary"]
        if "angle_deg" in summary:
            core.append(f"angle_deg={summary['angle_deg']}")
        if "scale" in summary:
            core.append(f"scale={summary['scale']}")
    return " | ".join(core)


def print_verbose_metrics(result: dict[str, Any]) -> None:
    print(format_result(result))
    if "anchor_stats" in result:
        for anchor_stat in result["anchor_stats"]:
            print(
                "  anchor={anchor_name} template_kp={template_keypoints} good_matches={good_matches} match_ok={match_ok}".format(
                    **anchor_stat
                )
            )
    if "rules" in result:
        print(f"  rules={json.dumps(result['rules'], ensure_ascii=False)}")
    if "side_results" in result:
        for side_name, side_result in result["side_results"].items():
            side_features = side_result.get("layout_features", {})
            side_summary = {
                "center_white_ratio": side_features.get("center_white_ratio"),
                "center_green_ratio": side_features.get("center_green_ratio"),
                "center_vertical_aspect": side_features.get("center_vertical_aspect"),
                "white_big_count": side_features.get("white_big_count"),
                "top_white_count": side_features.get("top_white_count"),
                "bottom_white_count": side_features.get("bottom_white_count"),
            }
            print(
                f"  side={side_name} result={side_result['result']} reason={side_result['decision_reason']} "
                f"features={json.dumps(side_summary, ensure_ascii=False)}"
            )
    if result["result"] in {"OK", "NG"}:
        print(f"FINAL_RESULT={result['result']}")


def resolve_image_paths(single_image: str | None, pattern: str | None) -> list[Path]:
    if single_image:
        return [resolve_path_argument(single_image)]

    matches: list[Path] = []
    seen: set[Path] = set()
    for candidate_pattern in path_text_candidates(pattern or ""):
        for item in sorted(glob.glob(candidate_pattern)):
            resolved = Path(item).resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            matches.append(resolved)
        if matches:
            break

    if not matches:
        raise FileNotFoundError(f"No files matched glob: {pattern}")
    return matches


def write_output_json(path: Path, results: Iterable[dict[str, Any]]) -> None:
    serializable = []
    for item in results:
        converted = {
            key: value
            for key, value in item.items()
            if key not in {"binary_mask"}
        }
        serializable.append(converted)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(serializable, handle, indent=2, ensure_ascii=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Check whether a resistor chip is present in a fixed PCB location."
    )
    parser.add_argument("--config", required=True, help="Path to config JSON or YAML")
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--image", help="Single test image path")
    input_group.add_argument("--glob", dest="glob_pattern", help="Glob for batch images")
    parser.add_argument(
        "--debug-dir",
        help="Optional directory to save overlays, masks, and per-image metrics",
    )
    parser.add_argument(
        "--output-json",
        help="Optional JSON file to save results for one or many images",
    )
    parser.add_argument(
        "--result-only",
        action="store_true",
        help="Print only OK/NG for single-image mode",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config_path = resolve_path_argument(args.config)
    config = load_config(config_path)
    template_state = load_template_state(config, config_path)
    image_paths = resolve_image_paths(args.image, args.glob_pattern)
    debug_root = resolve_path_argument(args.debug_dir) if args.debug_dir else None

    if args.result_only and len(image_paths) != 1:
        raise SystemExit("--result-only can only be used with --image")

    results = []
    for image_path in image_paths:
        result = analyze_image(image_path, config, template_state, debug_root)
        results.append(result)
        if args.result_only:
            print(result["result"])
        else:
            print_verbose_metrics(result)

    if args.output_json:
        write_output_json(resolve_path_argument(args.output_json), results)
    return 0


if __name__ == "__main__":
    sys.exit(main())