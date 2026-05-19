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


# ============ 分类方案管理 ============
# 方案一: KNN (classifiers/knn_classifier.py)
# 方案二: HOG+SVM (pcb_resistor_checker/)

SCHEME_FILE = os.path.join(BASE_DIR, 'current_scheme.txt')

def get_current_scheme():
    if os.path.exists(SCHEME_FILE):
        with open(SCHEME_FILE, 'r') as f:
            return f.read().strip()
    return '1'

def set_scheme(scheme):
    with open(SCHEME_FILE, 'w') as f:
        f.write(scheme.strip())


# ============ SMD Classifier (方案一: KNN) ============
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

    @property
    def labels(self):
        return self._clf.labels


# ============ HOG+SVM Classifier (方案二) ============
class HOGSVMClassifier:
    """委托给 pcb_resistor_checker 的 HOG+Linear SVM 分类器"""
    def __init__(self):
        checker_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'pcb_resistor_checker')
        sys.path.insert(0, checker_dir)
        import importlib
        self._mod = importlib.import_module('detect_resistor_presence')
        self._roi_common = importlib.import_module('roi_classifier_common')
        self._roi_hog = importlib.import_module('roi_classifier_hog_svm')

        config_path = os.path.join(checker_dir, 'config.yaml')
        model_dir = os.path.join(checker_dir, 'models', 'hog_svm_v1')

        from pathlib import Path
        self._config = self._mod.load_config(Path(config_path))
        self._template_state = self._mod.load_template_state(self._config, Path(config_path))
        self._svm, self._feature_config, self._metadata = self._roi_hog.load_model_bundle(Path(model_dir))
        print(f"[hog_svm] Model loaded: {self._metadata.get('model_type')}")

    def predict(self, img_bytes):
        """输入: PNG bytes (子图或完整图)"""
        import cv2
        import tempfile
        # 写临时文件用于 localize
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
            tmp.write(img_bytes)
            tmp_path = tmp.name
        try:
            from pathlib import Path
            localized, payload = self._roi_common.localize_detect_roi(
                Path(tmp_path), self._config, self._template_state
            )
            if localized is None:
                return 1, 0.0, {
                    "reason": "localization_failed",
                    "localization_reason": payload.get("reason", "unknown"),
                    "scheme": "hog_svm"
                }
            masked_roi = self._roi_common.mask_roi_crop(localized.roi_crop_bgr, localized.roi_mask)
            features = self._roi_hog.compute_hog_features(masked_roi, self._feature_config).reshape(1, -1)
            predicted_ids, raw_scores = self._roi_hog.predict_label_ids(self._svm, features)
            predicted_label = self._roi_hog.decode_label(int(predicted_ids[0]))
            pred = 0 if predicted_label == 'good' else 1
            confidence = min(abs(float(raw_scores[0])) / 2.0, 1.0)
            details = {
                "reason": "hog_linear_svm",
                "svm_margin": round(abs(float(raw_scores[0])), 6),
                "total_good_matches": localized.total_good_matches,
                "inlier_count": localized.inlier_count,
                "scheme": "hog_svm"
            }
            return pred, confidence, details
        finally:
            os.unlink(tmp_path)

    def reload(self):
        pass  # HOG+SVM 模型是静态的

    @property
    def labels(self):
        return np.array([])


classifier_v1 = SMDClassifier(DATA_DIR)
classifier_v2 = None  # 延迟加载

def get_classifier():
    scheme = get_current_scheme()
    if scheme == '2':
        global classifier_v2
        if classifier_v2 is None:
            classifier_v2 = HOGSVMClassifier()
        return classifier_v2, '2'
    return classifier_v1, '1'


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
                clf, scheme = get_classifier()
                pred, conf, details = clf.predict(img_bytes)
                self.send_json(200, {
                    "result": "good" if pred == 0 else "bad",
                    "confidence": round(conf, 4),
                    "scheme": scheme,
                    **details
                })
            except Exception as e:
                self.send_json(500, {"error": str(e)})

        # --- /classify/reload ---
        elif path == '/classify/reload':
            clf, scheme = get_classifier()
            clf.reload()
            n_samples = len(clf.labels) if len(clf.labels) > 0 else '?'
            self.send_json(200, {"ok": True, "scheme": scheme, "samples": n_samples})

        # --- /classify/scheme --- 切换分类方案
        elif path == '/classify/scheme':
            length = int(self.headers.get('Content-Length', 0))
            body = self.rfile.read(length)
            try:
                data = json.loads(body)
                new_scheme = str(data.get('scheme', '1'))
                if new_scheme not in ('1', '2'):
                    self.send_json(400, {"error": "scheme must be 1 or 2"})
                    return
                set_scheme(new_scheme)
                # 如果切到方案二，预加载
                if new_scheme == '2':
                    get_classifier()
                self.send_json(200, {"ok": True, "scheme": new_scheme})
            except Exception as e:
                self.send_json(500, {"error": str(e)})

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
            clf, scheme = get_classifier()
            labels = clf.labels
            self.send_json(200, {
                "status": "ready",
                "scheme": scheme,
                "samples": len(labels),
                "good": int((labels == 0).sum()) if len(labels) > 0 else '?',
                "bad": int((labels == 1).sum()) if len(labels) > 0 else '?'
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
