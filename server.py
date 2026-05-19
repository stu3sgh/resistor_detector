#!/usr/bin/env python3
"""
产线良品识别 - 统一后端服务
合并了: save_server (8001), smd_classifier (8002), predict/stats (原8000)
统一端口: 8000
路由:
  POST /predict          - AI 预测 (图片FormData)
  GET  /stats            - 统计数据
  POST /upload/save      - 保存裁剪图片
  POST /upload/save_result - 保存检测结果
  GET  /upload/latest    - 最新上传记录
  GET  /upload/list      - 所有上传记录
  POST /classify         - SMD 分类
  GET  /classify         - 分类器状态
  POST /classify/reload  - 重载训练数据
"""

import os
import sys
import time
import json
import base64
import io
import numpy as np
from PIL import Image, ImageFilter
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs
from datetime import datetime

# ============ Config ============
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR = os.path.join(BASE_DIR, 'uploads')
SAVE_DIR = os.path.join(BASE_DIR, 'detection_results')
DATA_DIR = os.path.join(SAVE_DIR, 'smd_components')
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(SAVE_DIR, exist_ok=True)
META_FILE = os.path.join(UPLOAD_DIR, 'meta.json')
PORT = 8000


# ============ Meta helpers ============
def load_meta():
    if os.path.exists(META_FILE):
        with open(META_FILE, 'r') as f:
            return json.load(f)
    return []

def save_meta(meta):
    with open(META_FILE, 'w') as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)


# ============ SMD Classifier ============
class SMDClassifier:
    """兼容 wrapper — 委托给 classifiers/knn_classifier.py"""
    def __init__(self, data_dir):
        sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'classifiers'))
        from knn_classifier import KNNClassifier
        self._clf = KNNClassifier(data_dir, k=3)

    def predict(self, img_input):
        r = self._clf.predict(img_input)
        pred = 0 if r['result'] == 'good' else 1
        return pred, r['confidence'], r['details']

    def reload(self):
        self._clf.reload()


classifier = SMDClassifier(DATA_DIR)


# ============ HTTP Handler ============
class Handler(BaseHTTPRequestHandler):

    def do_OPTIONS(self):
        self.send_response(204)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()

    def do_POST(self):
        parsed = urlparse(self.path)
        path = parsed.path

        # --- /predict (AI prediction) ---
        if path == '/predict':
            content_length = int(self.headers.get('Content-Length', 0))
            body = self.rfile.read(content_length)
            # Demo fallback - no separate AI model, return demo results
            # In future, integrate actual detection model here
            self.send_json(200, {
                "success": True,
                "predictions": [
                    {"class": "SMD 元件 - 外观检测", "confidence": 0.95},
                    {"class": "主芯片 - 焊点检测", "confidence": 0.93},
                    {"class": "底部芯片 - 引脚检测", "confidence": 0.91}
                ]
            })

        # --- /upload/save ---
        elif path == '/upload/save':
            content_length = int(self.headers.get('Content-Length', 0))
            body = self.rfile.read(content_length)
            if not body:
                self.send_json(400, {'error': '空文件'})
                return
            ts = datetime.now().strftime('%Y%m%d_%H%M%S')
            uid = str(int(time.time() * 1e6))[-6:]
            filename = f'{ts}_{uid}.png'
            filepath = os.path.join(UPLOAD_DIR, filename)
            with open(filepath, 'wb') as f:
                f.write(body)
            w, h = 0, 0
            try:
                with Image.open(filepath) as img:
                    w, h = img.size
            except:
                pass
            meta = load_meta()
            record = {
                'filename': filename, 'width': w, 'height': h,
                'size': len(body),
                'time': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
            meta.append(record)
            save_meta(meta)
            self.send_json(200, {
                'ok': True, 'filename': filename,
                'width': w, 'height': h, 'size': len(body)
            })

        # --- /upload/save_result ---
        elif path == '/upload/save_result':
            content_length = int(self.headers.get('Content-Length', 0))
            body = self.rfile.read(content_length)
            try:
                data = json.loads(body)
                image_data_url = data.get('image', '')
                results = data.get('results', [])
                if not image_data_url.startswith('data:'):
                    self.send_json(400, {'error': '无效的图片数据'})
                    return
                header, b64 = image_data_url.split(',', 1)
                img_bytes = base64.b64decode(b64)
                img = Image.open(io.BytesIO(img_bytes))
                W, H = img.size
                regions = [
                    ('smd_components', [0.01, 0.01, 0.51, 0.14]),
                    ('main_chip', [0.21, 0.54, 0.74, 0.77]),
                    ('bottom_chip', [0.18, 0.81, 0.60, 0.95])
                ]
                ts = datetime.now().strftime('%Y%m%d_%H%M%S')
                uid = str(int(time.time() * 1e6))[-6:]
                saved = []
                for i, (label, box) in enumerate(regions):
                    verdict = results[i]['verdict'] if i < len(results) else 'good'
                    folder = os.path.join(SAVE_DIR, label, verdict)
                    os.makedirs(folder, exist_ok=True)
                    x1, y1, x2, y2 = [int(v) for v in [box[0]*W, box[1]*H, box[2]*W, box[3]*H]]
                    sub = img.crop((x1, y1, x2, y2))
                    fname = f'{ts}_{uid}.png'
                    sub.save(os.path.join(folder, fname))
                    saved.append({'label': label, 'verdict': verdict, 'file': fname})
                self.send_json(200, {'ok': True, 'saved': saved})
            except Exception as e:
                self.send_json(500, {'error': str(e)})

        # --- /classify ---
        elif path == '/classify':
            length = int(self.headers.get('Content-Length', 0))
            body = self.rfile.read(length)
            try:
                data = json.loads(body)
                image_data = data.get('image', '')
                if not image_data.startswith('data:'):
                    self.send_json(400, {"error": "invalid image data"})
                    return
                header, b64 = image_data.split(',', 1)
                img_bytes = base64.b64decode(b64)
                pred, conf, details = classifier.predict(img_bytes)
                self.send_json(200, {
                    "result": "good" if pred == 0 else "bad",
                    "confidence": round(conf, 4),
                    **details
                })
            except Exception as e:
                self.send_json(500, {"error": str(e)})

        # --- /classify/reload ---
        elif path == '/classify/reload':
            classifier.reload()
            self.send_json(200, {"ok": True, "samples": len(classifier.labels)})

        else:
            self.send_json(404, {'error': 'not found'})

    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path

        # --- /stats ---
        if path == '/stats':
            self.send_json(200, {
                "total": len(load_meta()),
                "defects": 0
            })

        # --- /upload/latest ---
        elif path == '/upload/latest':
            meta = load_meta()
            if meta:
                self.send_json(200, meta[-1])
            else:
                self.send_json(404, {'error': '没有保存的图片'})

        # --- /upload/list ---
        elif path == '/upload/list':
            self.send_json(200, load_meta())

        # --- /upload/latest/download ---
        elif path == '/upload/latest/download':
            meta = load_meta()
            if meta:
                latest = meta[-1]
                filepath = os.path.join(UPLOAD_DIR, latest['filename'])
                if os.path.exists(filepath):
                    self.send_file(filepath, latest['filename'])
                    return
            self.send_json(404, {'error': '没有保存的图片'})

        # --- /classify (status) ---
        elif path == '/classify':
            self.send_json(200, {
                "status": "ready",
                "samples": len(classifier.labels),
                "good": int((classifier.labels==0).sum()),
                "bad": int((classifier.labels==1).sum())
            })

        else:
            self.send_json(404, {'error': 'not found'})

    def send_json(self, code, data):
        body = json.dumps(data, ensure_ascii=False).encode()
        self.send_response(code)
        self.send_header('Content-Type', 'application/json; charset=utf-8')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def send_file(self, filepath, filename):
        import mimetypes
        mime = mimetypes.guess_type(filename)[0] or 'application/octet-stream'
        size = os.path.getsize(filepath)
        self.send_response(200)
        self.send_header('Content-Type', mime)
        self.send_header('Content-Length', str(size))
        self.send_header('Content-Disposition', f'attachment; filename="{filename}"')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        with open(filepath, 'rb') as f:
            self.wfile.write(f.read())

    def log_message(self, format, *args):
        print("[server]", *args)


if __name__ == '__main__':
    server = HTTPServer(('0.0.0.0', PORT), Handler)
    print(f"产线良品识别服务 running on port {PORT}")
    print(f"Upload dir: {UPLOAD_DIR}")
    print(f"Training data: {DATA_DIR}")
    server.serve_forever()
