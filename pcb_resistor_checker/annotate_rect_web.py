#!/usr/bin/env python3

from __future__ import annotations

import argparse
import base64
import json
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import cv2

from detect_resistor_presence import (
  DEFAULT_CONFIG,
  build_platform_path_value,
  deep_merge,
  load_raw_config_file,
  read_image_bgr,
  resolve_path_argument,
  write_config_file,
)


def load_existing_config(config_path: Path | None) -> dict[str, Any]:
  if config_path is None or not config_path.exists():
    return {}
  return load_raw_config_file(config_path)


def resolve_existing_rectangles(config: dict[str, Any], detect_name: str) -> dict[str, list[int]]:
    rectangles: dict[str, list[int]] = {}
    for anchor in config.get("anchors", []):
        if "rect" in anchor:
            rectangles[anchor.get("name", f"anchor_{len(rectangles) + 1}")] = [int(v) for v in anchor["rect"]]
    detect_roi = config.get("detect_roi", {})
    if isinstance(detect_roi, dict) and "rect" in detect_roi:
        rectangles[detect_name] = [int(v) for v in detect_roi["rect"]]
    return rectangles


def infer_region_names(
    explicit_region_names: list[str] | None,
    config: dict[str, Any],
    detect_name: str,
) -> list[str]:
    if explicit_region_names:
        return explicit_region_names

    anchor_names = [anchor.get("name") for anchor in config.get("anchors", []) if anchor.get("name")]
    if anchor_names:
        return anchor_names + [detect_name]

    return ["anchor_left", "anchor_mid", "anchor_right", detect_name]


def build_output_config(
    image_path: Path,
    region_names: list[str],
    detect_name: str,
    base_config: dict[str, Any],
    rectangles: dict[str, list[int]],
) -> dict[str, Any]:
    merged = deep_merge(DEFAULT_CONFIG, base_config)
    merged["template_image"] = build_platform_path_value(
        image_path,
        base_config.get("template_image"),
    )

    anchors = []
    for region_name in region_names:
        if region_name == detect_name:
            continue
        rect = rectangles.get(region_name)
        if rect is None:
            continue
        anchors.append({"name": region_name, "rect": [int(v) for v in rect]})
    merged["anchors"] = anchors

    detect_rect = rectangles.get(detect_name)
    if detect_rect is not None:
        merged["detect_roi"] = {"rect": [int(v) for v in detect_rect]}
    return merged


def encode_image_data_url(image_bgr) -> str:
    success, encoded = cv2.imencode(".jpg", image_bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])
    if not success:
        raise RuntimeError("Failed to encode image for browser annotation")
    payload = base64.b64encode(encoded.tobytes()).decode("ascii")
    return f"data:image/jpeg;base64,{payload}"


def make_html(app_state: dict[str, Any]) -> str:
    page_payload = {
        "imagePath": app_state["imagePath"],
        "imageDataUrl": app_state["imageDataUrl"],
        "imageWidth": app_state["imageWidth"],
        "imageHeight": app_state["imageHeight"],
        "regionNames": app_state["regionNames"],
        "detectName": app_state["detectName"],
        "existingRectangles": app_state["existingRectangles"],
        "baseConfig": app_state["baseConfig"],
        "port": app_state["port"],
    }
    app_json = json.dumps(page_payload, ensure_ascii=False)
    return f"""<!DOCTYPE html>
<html lang=\"en\">
<head>
  <meta charset=\"UTF-8\" />
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1.0\" />
  <title>PCB Rect Annotator</title>
  <style>
    :root {{
      color-scheme: light;
      --bg: #f1efe8;
      --panel: #fffaf0;
      --ink: #1f2521;
      --muted: #5f655f;
      --accent: #0b6e4f;
      --accent-2: #f4b400;
      --danger: #b93827;
      --line: #d9d2c2;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: "Segoe UI", "Helvetica Neue", Arial, sans-serif;
      background: radial-gradient(circle at top left, #fff8e6, var(--bg) 55%);
      color: var(--ink);
    }}
    .layout {{
      display: grid;
      grid-template-columns: 340px 1fr;
      min-height: 100vh;
    }}
    .sidebar {{
      border-right: 1px solid var(--line);
      padding: 20px;
      background: linear-gradient(180deg, rgba(255,255,255,0.92), rgba(255,250,240,0.96));
      overflow-y: auto;
    }}
    .sidebar h1 {{
      margin: 0 0 10px;
      font-size: 24px;
      letter-spacing: 0.02em;
    }}
    .sidebar p, .sidebar li {{
      color: var(--muted);
      line-height: 1.45;
      font-size: 14px;
    }}
    .stage-wrap {{
      padding: 18px;
      overflow: auto;
    }}
    .toolbar {{
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      margin: 14px 0;
    }}
    button {{
      border: 1px solid var(--line);
      background: white;
      color: var(--ink);
      padding: 9px 12px;
      border-radius: 10px;
      cursor: pointer;
      font-weight: 600;
    }}
    button.primary {{
      background: var(--accent);
      color: white;
      border-color: var(--accent);
    }}
    button.warn {{
      color: var(--danger);
    }}
    .status {{
      padding: 12px;
      border-radius: 12px;
      background: rgba(11, 110, 79, 0.08);
      color: var(--ink);
      margin-bottom: 14px;
      font-size: 14px;
    }}
    .coords {{
      font-family: Consolas, "Courier New", monospace;
      font-size: 13px;
      color: var(--muted);
      margin-top: 10px;
    }}
    .region-list {{
      display: grid;
      gap: 10px;
      margin: 14px 0 18px;
    }}
    .region-card {{
      border: 1px solid var(--line);
      border-radius: 12px;
      padding: 10px 12px;
      background: rgba(255,255,255,0.9);
      cursor: pointer;
    }}
    .region-card.active {{
      border-color: var(--accent);
      box-shadow: 0 0 0 3px rgba(11,110,79,0.12);
    }}
    .region-card .name {{
      font-weight: 700;
      margin-bottom: 4px;
    }}
    .region-card .value {{
      font-family: Consolas, "Courier New", monospace;
      font-size: 12px;
      color: var(--muted);
      word-break: break-all;
    }}
    .json-box {{
      width: 100%;
      min-height: 180px;
      border-radius: 12px;
      border: 1px solid var(--line);
      padding: 10px;
      font-family: Consolas, "Courier New", monospace;
      font-size: 12px;
      resize: vertical;
      background: rgba(255,255,255,0.85);
    }}
    .stage {{
      position: relative;
      display: inline-block;
      border-radius: 18px;
      overflow: hidden;
      background: #222;
      box-shadow: 0 18px 40px rgba(0,0,0,0.12);
      max-width: calc(100vw - 420px);
    }}
    #pcbImage {{
      display: block;
      max-width: 100%;
      height: auto;
    }}
    #overlay {{
      position: absolute;
      left: 0;
      top: 0;
      width: 100%;
      height: 100%;
      cursor: crosshair;
    }}
    .footer-note {{
      margin-top: 14px;
      font-size: 12px;
      color: var(--muted);
    }}
    @media (max-width: 1000px) {{
      .layout {{ grid-template-columns: 1fr; }}
      .sidebar {{ border-right: none; border-bottom: 1px solid var(--line); }}
      .stage {{ max-width: calc(100vw - 36px); }}
    }}
  </style>
</head>
<body>
  <div class=\"layout\">
    <aside class=\"sidebar\">
      <h1>PCB Rect Annotator</h1>
      <div class=\"status\" id=\"statusBox\"></div>
      <div class=\"toolbar\">
        <button id=\"prevBtn\">Prev</button>
        <button id=\"nextBtn\">Next</button>
        <button id=\"undoBtn\">Undo Current</button>
        <button id=\"clearBtn\" class=\"warn\">Clear All</button>
        <button id=\"copyBtn\">Copy JSON</button>
        <button id=\"saveBtn\" class=\"primary\">Save To File</button>
      </div>
      <p>Draw rectangles in order. Click a region card to re-label that region. The browser uses the same EXIF-corrected image coordinates as the detector.</p>
      <div class=\"coords\" id=\"coordsBox\">mouse=( -, - )</div>
      <div class=\"region-list\" id=\"regionList\"></div>
      <textarea id=\"jsonBox\" class=\"json-box\" spellcheck=\"false\"></textarea>
      <div class=\"footer-note\">Open this page from Windows with <strong>http://localhost:{app_state['port']}</strong>. Stop the server with Ctrl+C after saving.</div>
    </aside>
    <main class=\"stage-wrap\">
      <div class=\"stage\" id=\"stage\">
        <img id=\"pcbImage\" alt=\"PCB image\" />
        <canvas id=\"overlay\"></canvas>
      </div>
    </main>
  </div>

  <script>
    const APP = {app_json};
    const regionNames = APP.regionNames;
    const detectName = APP.detectName;
    const imageWidth = APP.imageWidth;
    const imageHeight = APP.imageHeight;
    const state = {{
      rectangles: {{ ...APP.existingRectangles }},
      history: [],
      currentIndex: 0,
      dragging: false,
      dragStart: null,
      dragEnd: null,
    }};

    const img = document.getElementById('pcbImage');
    const canvas = document.getElementById('overlay');
    const ctx = canvas.getContext('2d');
    const statusBox = document.getElementById('statusBox');
    const coordsBox = document.getElementById('coordsBox');
    const regionList = document.getElementById('regionList');
    const jsonBox = document.getElementById('jsonBox');

    img.src = APP.imageDataUrl;

    function currentRegionName() {{
      return regionNames[Math.max(0, Math.min(state.currentIndex, regionNames.length - 1))];
    }}

    function setCanvasSize() {{
      canvas.width = img.clientWidth;
      canvas.height = img.clientHeight;
      draw();
    }}

    function scaleX() {{ return canvas.width / imageWidth; }}
    function scaleY() {{ return canvas.height / imageHeight; }}

    function imageToCanvasRect(rect) {{
      return [rect[0] * scaleX(), rect[1] * scaleY(), rect[2] * scaleX(), rect[3] * scaleY()];
    }}

    function pointerToImage(event) {{
      const bounds = canvas.getBoundingClientRect();
      const x = Math.max(0, Math.min(imageWidth, (event.clientX - bounds.left) / scaleX()));
      const y = Math.max(0, Math.min(imageHeight, (event.clientY - bounds.top) / scaleY()));
      return [Math.round(x), Math.round(y)];
    }}

    function normalizeRect(a, b) {{
      const x0 = Math.min(a[0], b[0]);
      const y0 = Math.min(a[1], b[1]);
      const x1 = Math.max(a[0], b[0]);
      const y1 = Math.max(a[1], b[1]);
      return [x0, y0, Math.max(1, x1 - x0), Math.max(1, y1 - y0)];
    }}

    function regionColor(name) {{
      if (name === detectName) return '#00d2ff';
      return name === currentRegionName() ? '#10b981' : '#f4b400';
    }}

    function drawRect(name, rect, isActive) {{
      const [x, y, w, h] = imageToCanvasRect(rect);
      ctx.save();
      ctx.strokeStyle = regionColor(name);
      ctx.lineWidth = isActive ? 3 : 2;
      ctx.setLineDash(isActive ? [] : [8, 6]);
      ctx.strokeRect(x, y, w, h);
      ctx.fillStyle = 'rgba(0,0,0,0.65)';
      const label = `${{name}}  [${{rect.join(', ')}}]`;
      ctx.font = '14px Consolas';
      const width = ctx.measureText(label).width + 12;
      ctx.fillRect(x, Math.max(0, y - 24), width, 20);
      ctx.fillStyle = '#ffffff';
      ctx.fillText(label, x + 6, Math.max(14, y - 9));
      ctx.restore();
    }}

    function draw() {{
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      for (const name of regionNames) {{
        const rect = state.rectangles[name];
        if (rect) drawRect(name, rect, name === currentRegionName());
      }}
      if (state.dragging && state.dragStart && state.dragEnd) {{
        const rect = normalizeRect(state.dragStart, state.dragEnd);
        drawRect(currentRegionName(), rect, true);
      }}
    }}

    function renderRegionList() {{
      regionList.innerHTML = '';
      regionNames.forEach((name, index) => {{
        const rect = state.rectangles[name];
        const card = document.createElement('div');
        card.className = 'region-card' + (index === state.currentIndex ? ' active' : '');
        card.innerHTML = `
          <div class=\"name\">${{name}}</div>
          <div class=\"value\">${{rect ? '[' + rect.join(', ') + ']' : '(not set)'}} </div>
        `;
        card.addEventListener('click', () => {{
          state.currentIndex = index;
          updatePanels();
        }});
        regionList.appendChild(card);
      }});
    }}

    function exportConfig() {{
      const anchors = [];
      for (const name of regionNames) {{
        if (name === detectName) continue;
        if (state.rectangles[name]) {{
          anchors.push({{ name, rect: state.rectangles[name] }});
        }}
      }}
      const output = JSON.parse(JSON.stringify(APP.baseConfig));
      output.anchors = anchors;
      if (state.rectangles[detectName]) {{
        output.detect_roi = {{ rect: state.rectangles[detectName] }};
      }}
      return output;
    }}

    function updatePanels() {{
      const currentName = currentRegionName();
      statusBox.textContent = `Current region: ${{currentName}}. Drag a box on the image. Existing rectangles can be overwritten.`;
      renderRegionList();
      jsonBox.value = JSON.stringify(exportConfig(), null, 2);
      draw();
    }}

    canvas.addEventListener('mousedown', (event) => {{
      state.dragging = true;
      state.dragStart = pointerToImage(event);
      state.dragEnd = state.dragStart;
      draw();
    }});

    canvas.addEventListener('mousemove', (event) => {{
      const point = pointerToImage(event);
      coordsBox.textContent = `mouse=(${{point[0]}}, ${{point[1]}})`;
      if (!state.dragging) return;
      state.dragEnd = point;
      draw();
    }});

    function finishDrag() {{
      if (!state.dragging || !state.dragStart || !state.dragEnd) return;
      const rect = normalizeRect(state.dragStart, state.dragEnd);
      const name = currentRegionName();
      state.history.push({{ name, prev: state.rectangles[name] || null }});
      state.rectangles[name] = rect;
      state.dragging = false;
      state.dragStart = null;
      state.dragEnd = null;
      if (state.currentIndex < regionNames.length - 1) state.currentIndex += 1;
      updatePanels();
    }}

    canvas.addEventListener('mouseup', finishDrag);
    canvas.addEventListener('mouseleave', () => {{
      if (state.dragging) finishDrag();
    }});

    document.getElementById('prevBtn').addEventListener('click', () => {{
      state.currentIndex = Math.max(0, state.currentIndex - 1);
      updatePanels();
    }});

    document.getElementById('nextBtn').addEventListener('click', () => {{
      state.currentIndex = Math.min(regionNames.length - 1, state.currentIndex + 1);
      updatePanels();
    }});

    document.getElementById('undoBtn').addEventListener('click', () => {{
      const last = state.history.pop();
      if (!last) return;
      if (last.prev) state.rectangles[last.name] = last.prev;
      else delete state.rectangles[last.name];
      state.currentIndex = regionNames.indexOf(last.name);
      updatePanels();
    }});

    document.getElementById('clearBtn').addEventListener('click', () => {{
      state.rectangles = {{}};
      state.history = [];
      state.currentIndex = 0;
      updatePanels();
    }});

    document.getElementById('copyBtn').addEventListener('click', async () => {{
      await navigator.clipboard.writeText(jsonBox.value);
      statusBox.textContent = 'JSON copied to clipboard.';
    }});

    document.getElementById('saveBtn').addEventListener('click', async () => {{
      const response = await fetch('/save', {{
        method: 'POST',
        headers: {{ 'Content-Type': 'application/json' }},
        body: JSON.stringify({{ rectangles: state.rectangles }})
      }});
      const data = await response.json();
      if (!response.ok) {{
        statusBox.textContent = `Save failed: ${{data.error || 'unknown error'}}`;
        return;
      }}
      jsonBox.value = JSON.stringify(data.config, null, 2);
      statusBox.textContent = `Saved to ${{data.output_config}}`;
    }});

    window.addEventListener('resize', setCanvasSize);
    img.addEventListener('load', () => {{
      setCanvasSize();
      updatePanels();
    }});
  </script>
</body>
</html>
"""


def make_handler(state: dict[str, Any]):
    class AnnotatorHandler(BaseHTTPRequestHandler):
        def _send_json(self, payload: dict[str, Any], status: int = HTTPStatus.OK) -> None:
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:
            if self.path not in {"/", "/index.html"}:
                self.send_error(HTTPStatus.NOT_FOUND)
                return

            html = make_html(state).encode("utf-8")
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(html)))
            self.end_headers()
            self.wfile.write(html)

        def do_POST(self) -> None:
            if self.path != "/save":
                self.send_error(HTTPStatus.NOT_FOUND)
                return

            content_length = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(content_length).decode("utf-8"))
            raw_rectangles = payload.get("rectangles", {})

            normalized_rectangles: dict[str, list[int]] = {}
            for name, rect in raw_rectangles.items():
                if not isinstance(rect, list) or len(rect) != 4:
                    continue
                normalized_rectangles[name] = [int(v) for v in rect]

            config = build_output_config(
                image_path=state["image_path"],
                region_names=state["region_names"],
                detect_name=state["detect_name"],
                base_config=state["base_config"],
                rectangles=normalized_rectangles,
            )

            output_config = state["output_config"]
            if output_config is not None:
              write_config_file(output_config, config)

            self._send_json(
                {
                    "ok": True,
                    "output_config": str(output_config) if output_config else "(not written)",
                    "config": config,
                }
            )

        def log_message(self, format: str, *args: Any) -> None:
            return

    return AnnotatorHandler


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a browser-based rectangle annotation tool that works in headless WSL environments."
    )
    parser.add_argument("--image", required=True, help="Image path to annotate")
    parser.add_argument("--config", help="Optional existing JSON or YAML config to preload")
    parser.add_argument("--output-config", help="Optional JSON or YAML config path to write on save")
    parser.add_argument(
        "--region-names",
        nargs="+",
      help="Optional region names to annotate in order. By default, anchor names are inferred from the config.",
    )
    parser.add_argument("--detect-name", default="detect_roi", help="Which region is the detect ROI")
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind")
    parser.add_argument("--port", type=int, default=8765, help="Port to bind")
    parser.add_argument(
        "--no-exif",
        action="store_true",
        help="Disable EXIF orientation correction. Usually keep this off for phone photos.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    image_path = resolve_path_argument(args.image)
    config_path = resolve_path_argument(args.config) if args.config else None
    output_config = resolve_path_argument(args.output_config) if args.output_config else None

    base_config = load_existing_config(config_path)
    region_names = infer_region_names(args.region_names, base_config, args.detect_name)
    image_bgr = read_image_bgr(image_path, honor_exif_orientation=not args.no_exif)
    image_height, image_width = image_bgr.shape[:2]
    app_state = {
        "imagePath": str(image_path),
        "imageDataUrl": encode_image_data_url(image_bgr),
        "imageWidth": image_width,
        "imageHeight": image_height,
        "regionNames": region_names,
        "detectName": args.detect_name,
        "existingRectangles": resolve_existing_rectangles(base_config, args.detect_name),
        "baseConfig": build_output_config(
            image_path=image_path,
            region_names=region_names,
            detect_name=args.detect_name,
            base_config=base_config,
            rectangles=resolve_existing_rectangles(base_config, args.detect_name),
        ),
        "image_path": image_path,
        "region_names": region_names,
        "detect_name": args.detect_name,
        "base_config": base_config,
        "output_config": output_config,
        "port": args.port,
    }

    server = ThreadingHTTPServer((args.host, args.port), make_handler(app_state))
    print(f"Open in browser: http://localhost:{args.port}")
    print(f"Listening on: http://{args.host}:{args.port}")
    print("Press Ctrl+C to stop the annotator server.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())