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
class HOGSVMClassifierV2:
    """V2 多区域 HOG+SVM 分类器（支持 smd_components / main_chip / bottom_chip）"""
    def __init__(self):
        import cv2
        import importlib
        checker_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'pcb_resistor_checker')
        sys.path.insert(0, checker_dir)

        self._cv2 = cv2
        self._roi_common = importlib.import_module('roi_classifier_common')
        self._roi_hog = importlib.import_module('roi_classifier_hog_svm')
        self._detect_mod = importlib.import_module('detect_resistor_presence')
        self._v2_mod = importlib.import_module('infer_multi_region_hog_svm_v2')

        models_root = os.path.join(checker_dir, 'models')
        config_path = os.path.join(checker_dir, 'config.yaml')

        from pathlib import Path
        models_path = Path(models_root)
        self._config = self._detect_mod.load_config(Path(config_path))
        self._template_state = self._detect_mod.load_template_state(self._config, Path(config_path))

        # 加载三个区域的模型
        smd_dir = self._v2_mod.find_latest_versioned_model_dir(models_path, "hog_svm")
        main_dir = self._v2_mod.find_latest_versioned_model_dir(models_path, "main_chip_hog_svm")
        bottom_dir = self._v2_mod.find_latest_versioned_model_dir(models_path, "bottom_chip_hog_svm")

        self._smd_svm, self._smd_feat, self._smd_meta = self._roi_hog.load_model_bundle(smd_dir)
        self._main_svm, self._main_feat, self._main_meta = self._roi_hog.load_model_bundle(main_dir)
        self._bottom_svm, self._bottom_feat, self._bottom_meta = self._roi_hog.load_model_bundle(bottom_dir)

        print(f"[hog_svm_v2] Models loaded: smd={smd_dir.name}, main={main_dir.name}, bottom={bottom_dir.name}")

    def classify_region(self, region_name, img_bytes):
        """对单个子图做 HOG+SVM 分类（用于 main_chip / bottom_chip）"""
        import tempfile
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
            tmp.write(img_bytes)
            tmp_path = tmp.name
        try:
            from pathlib import Path
            return self._v2_mod.classify_direct_region(
                region_name, Path(tmp_path),
                self._main_svm if region_name == 'main_chip' else self._bottom_svm,
                self._main_feat if region_name == 'main_chip' else self._bottom_feat,
                self._main_meta if region_name == 'main_chip' else self._bottom_meta,
            )
        finally:
            os.unlink(tmp_path)

    def classify_smd(self, img_bytes):
        """对 SMD 子图做 ORB 配准 + HOG+SVM 分类"""
        import tempfile
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
            tmp.write(img_bytes)
            tmp_path = tmp.name
        try:
            from pathlib import Path
            return self._v2_mod.classify_smd_region(
                Path(tmp_path), self._config, self._template_state,
                self._smd_svm, self._smd_feat, self._smd_meta,
            )
        finally:
            os.unlink(tmp_path)

    def predict(self, img_bytes):
        """兼容接口：返回 (pred, confidence, details)，用 SMD 区域的结果"""
        result = self.classify_smd(img_bytes)
        pred = 0 if result['result'] == 'good' else 1
        margin = result.get('svm_margin_abs', 0)
        confidence = min(margin / 1.5, 1.0) if margin else 0.5
        return pred, confidence, result

    def reload(self):
        pass

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
            classifier_v2 = HOGSVMClassifierV2()
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
            t0 = time.time()
            body = self.rfile.read(content_length)
            t1 = time.time()
            try:
                data = json.loads(body)
                t2 = time.time()
                image_data_url = data.get('image', '')
                results = data.get('results', [])
                if not image_data_url.startswith('data:'):
                    self.send_json(400, {'error': '无效的图片数据'})
                    return
                header, b64 = image_data_url.split(',', 1)
                img_bytes = base64.b64decode(b64)
                t3 = time.time()
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
                t4 = time.time()
                print(f"[save_result] read={(t1-t0)*1000:.0f}ms json_parse={(t2-t1)*1000:.0f}ms decode={(t3-t2)*1000:.0f}ms crop_save={(t4-t3)*1000:.0f}ms total={(t4-t0)*1000:.0f}ms")
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
                t0 = time.time()
                img_bytes = base64.b64decode(b64)
                t1 = time.time()
                clf, scheme = get_classifier()
                t2 = time.time()
                region_name = data.get('region', 'smd_components')

                # 方案二: 根据区域选择分类方式
                if scheme == '2' and hasattr(clf, 'classify_region'):
                    import cv2 as cv2_mod
                    from pathlib import Path
                    import tempfile
                    if region_name == 'smd_components':
                        r = clf.classify_smd(img_bytes)
                    else:
                        r = clf.classify_region(region_name, img_bytes)
                    pred = 0 if r['result'] == 'good' else 1
                    margin = r.get('svm_margin_abs', 0)
                    conf = min(margin / 1.5, 1.0) if margin else 0.5
                    details = {k: v for k, v in r.items() if k != 'result'}
                    details['scheme'] = 'hog_svm_v2'
                else:
                    pred, conf, details = clf.predict(img_bytes)
                    details['scheme'] = scheme

                t3 = time.time()
                details["_timing"] = {
                    "decode_ms": round((t1-t0)*1000),
                    "load_clf_ms": round((t2-t1)*1000),
                    "predict_ms": round((t3-t2)*1000),
                    "total_ms": round((t3-t0)*1000)
                }
                print(f"[classify] scheme={scheme} region={region_name} decode={(t1-t0)*1000:.0f}ms clf={(t2-t1)*1000:.0f}ms predict={(t3-t2)*1000:.0f}ms total={(t3-t0)*1000:.0f}ms")
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
