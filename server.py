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
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.features = []
        self.labels = []
        self.k = 3
        self.load_data()

    def load_data(self):
        if not os.path.exists(self.data_dir):
            return
        # 支持 good/bad 子目录结构，同时兼容旧的扁平结构
        good_dir = os.path.join(self.data_dir, 'good')
        bad_dir = os.path.join(self.data_dir, 'bad')

        if os.path.isdir(good_dir) and os.path.isdir(bad_dir):
            for f in sorted(os.listdir(good_dir)):
                if f.endswith('.png'):
                    self.features.append(self._extract_features(os.path.join(good_dir, f)))
                    self.labels.append(0)
            for f in sorted(os.listdir(bad_dir)):
                if f.endswith('.png'):
                    self.features.append(self._extract_features(os.path.join(bad_dir, f)))
                    self.labels.append(1)
        else:
            for f in sorted(os.listdir(self.data_dir)):
                if not f.endswith('.png'):
                    continue
                path = os.path.join(self.data_dir, f)
                self.features.append(self._extract_features(path))
                self.labels.append(0 if 'good' in f else 1)

        self.features = np.array(self.features) if self.features else np.array([])
        self.labels = np.array(self.labels) if self.labels else np.array([])
        print(f"[classifier] Loaded {len(self.labels)} samples: "
              f"{(self.labels==0).sum()} good, {(self.labels==1).sum()} bad")

    def _extract_features(self, img_input):
        if isinstance(img_input, str):
            img = Image.open(img_input).convert('RGB')
        elif not isinstance(img_input, Image.Image):
            img = Image.open(io.BytesIO(img_input)).convert('RGB')
        else:
            img = img_input.convert('RGB')

        img = img.resize((340, 215), Image.LANCZOS)
        arr = np.array(img, dtype=float)
        w, h = img.size
        gray = np.mean(arr, axis=2)
        features = []

        r, g, b = arr[:,:,0], arr[:,:,1], arr[:,:,2]
        is_pcb = (g > r) & (g > b) & (g > 60)
        is_silver = (r > 160) & (g > 160) & (b > 160)
        is_dark = gray < 40
        features.extend([is_pcb.sum()/arr.size, is_silver.sum()/arr.size, is_dark.sum()/arr.size])

        if is_silver.sum() > 10:
            sp = arr[is_silver]
            features.append(np.std(sp) / 255)
        else:
            features.append(0.5)

        blur = np.array(img.filter(ImageFilter.GaussianBlur(radius=3)), dtype=float)
        features.append(np.abs(arr - blur).mean() / 255)

        row_means = gray.mean(axis=1)
        features.append(np.std(row_means) / 255)
        features.append(np.std(np.diff(row_means)) / 255)

        for c_idx in range(3):
            ch = arr[:,:,c_idx]
            hist, _ = np.histogram(ch, bins=16, range=(0, 256))
            features.extend((hist / hist.sum()).tolist())

        for iy in range(4):
            for ix in range(4):
                y1, y2 = iy*h//4, (iy+1)*h//4
                x1, x2 = ix*w//4, (ix+1)*w//4
                features.append(gray[y1:y2, x1:x2].mean() / 255)

        return np.array(features)

    def _cosine_sim(self, a, b):
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10)

    def predict(self, img_input):
        if len(self.features) < 2:
            return 0, 0.5, {"error": "not enough training data"}
        test_feat = self._extract_features(img_input)
        sims = []
        for j in range(len(self.features)):
            cos = self._cosine_sim(test_feat, self.features[j])
            sims.append((cos, self.labels[j]))
        sims.sort(reverse=True)
        topk = sims[:self.k]
        good_sims = [s for s, l in topk if l == 0]
        bad_sims = [s for s, l in topk if l == 1]
        avg_good = np.mean(good_sims) if good_sims else 0
        avg_bad = np.mean(bad_sims) if bad_sims else 0
        pred = 1 if avg_bad > avg_good else 0
        confidence = abs(avg_bad - avg_good) / max(avg_good, avg_bad, 0.001)
        return pred, min(confidence, 1.0), {
            "avg_good_sim": round(avg_good, 4),
            "avg_bad_sim": round(avg_bad, 4),
            "k": self.k,
            "total_samples": len(self.labels),
            "good_samples": int((self.labels==0).sum()),
            "bad_samples": int((self.labels==1).sum()),
            "top_k": [{"sim": round(s, 4), "label": int(l)} for s, l in topk]
        }

    def reload(self):
        self.features = []
        self.labels = []
        self.load_data()


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
