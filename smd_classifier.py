#!/usr/bin/env python3
"""
SMD Component Defect Classifier
Uses KNN (k=3, cosine similarity) with color histogram + spatial features.
Trained on labeled data from detection_results/smd_components/.
"""

import os
import json
import numpy as np
from PIL import Image, ImageFilter
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse
import base64
import io

# ============ Config ============
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'detection_results', 'smd_components')
PORT = 8002

# ============ Feature Extraction ============
def extract_features(img):
    """Extract 71-dim feature vector from an image."""
    if isinstance(img, str):
        img = Image.open(img).convert('RGB')
    elif not isinstance(img, Image.Image):
        img = Image.open(io.BytesIO(img)).convert('RGB')

    # Resize to standard size for consistency
    img = img.resize((340, 215), Image.LANCZOS)
    arr = np.array(img, dtype=float)
    w, h = img.size
    gray = np.mean(arr, axis=2)
    features = []

    # 1. Region ratios
    r, g, b = arr[:,:,0], arr[:,:,1], arr[:,:,2]
    is_pcb = (g > r) & (g > b) & (g > 60)
    is_silver = (r > 160) & (g > 160) & (b > 160)
    is_dark = gray < 40
    features.extend([is_pcb.sum()/arr.size, is_silver.sum()/arr.size, is_dark.sum()/arr.size])

    # 2. Silver uniformity
    if is_silver.sum() > 10:
        sp = arr[is_silver]
        features.append(np.std(sp) / 255)
    else:
        features.append(0.5)

    # 3. Texture energy
    blur = np.array(img.filter(ImageFilter.GaussianBlur(radius=3)), dtype=float)
    features.append(np.abs(arr - blur).mean() / 255)

    # 4. Vertical pattern regularity
    row_means = gray.mean(axis=1)
    features.append(np.std(row_means) / 255)
    features.append(np.std(np.diff(row_means)) / 255)

    # 5. Color histogram (16 bins x 3 channels = 48)
    for c_idx in range(3):
        ch = arr[:,:,c_idx]
        hist, _ = np.histogram(ch, bins=16, range=(0, 256))
        features.extend((hist / hist.sum()).tolist())

    # 6. Spatial grid (4x4 = 16)
    for iy in range(4):
        for ix in range(4):
            y1, y2 = iy*h//4, (iy+1)*h//4
            x1, x2 = ix*w//4, (ix+1)*w//4
            features.append(gray[y1:y2, x1:x2].mean() / 255)

    return np.array(features)


def cosine_sim(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10)


# ============ Classifier ============
class SMDClassifier:
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.features = []
        self.labels = []  # 0=good, 1=bad
        self.k = 3
        self.load_data()

    def load_data(self):
        if not os.path.exists(self.data_dir):
            return
        for f in sorted(os.listdir(self.data_dir)):
            if not f.endswith('.png'):
                continue
            path = os.path.join(self.data_dir, f)
            feat = extract_features(path)
            label = 0 if 'good' in f else 1
            self.features.append(feat)
            self.labels.append(label)
        self.features = np.array(self.features) if self.features else np.array([])
        self.labels = np.array(self.labels) if self.labels else np.array([])
        print(f"Loaded {len(self.labels)} samples: "
              f"{(self.labels==0).sum()} good, {(self.labels==1).sum()} bad")

    def predict(self, img_input):
        """Predict if image is good (0) or bad (1).
        img_input: PIL Image, file path, or bytes.
        Returns: (prediction, confidence, details)
        """
        if len(self.features) < 2:
            return 0, 0.5, {"error": "not enough training data"}

        test_feat = extract_features(img_input)

        # KNN with cosine similarity
        sims = []
        for j in range(len(self.features)):
            cos = cosine_sim(test_feat, self.features[j])
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
        """Reload training data (call after new saves)."""
        self.features = []
        self.labels = []
        self.load_data()


# ============ HTTP Server ============
classifier = SMDClassifier(DATA_DIR)


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path == '/classify':
            self.send_json(200, {
                "status": "ready",
                "samples": len(classifier.labels),
                "good": int((classifier.labels==0).sum()),
                "bad": int((classifier.labels==1).sum())
            })
        elif parsed.path == '/reload':
            classifier.reload()
            self.send_json(200, {"ok": True, "samples": len(classifier.labels)})
        else:
            self.send_json(404, {"error": "not found"})

    def do_POST(self):
        parsed = urlparse(self.path)
        if parsed.path == '/classify':
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
                    "verdict": "good" if pred == 0 else "bad",
                    "confidence": round(conf, 4),
                    **details
                })
            except Exception as e:
                self.send_json(500, {"error": str(e)})
        elif parsed.path == '/reload':
            classifier.reload()
            self.send_json(200, {"ok": True, "samples": len(classifier.labels)})
        else:
            self.send_json(404, {"error": "not found"})

    def send_json(self, code, data):
        body = json.dumps(data, ensure_ascii=False).encode()
        self.send_response(code)
        self.send_header('Content-Type', 'application/json; charset=utf-8')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self):
        self.send_response(204)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()

    def log_message(self, format, *args):
        print("[classifier]", *args)


if __name__ == '__main__':
    server = HTTPServer(('0.0.0.0', PORT), Handler)
    print(f"SMD Classifier running on port {PORT}")
    print(f"Training data: {DATA_DIR}")
    print(f"Samples: {len(classifier.labels)} "
          f"({(classifier.labels==0).sum()} good, {(classifier.labels==1).sum()} bad)")
    server.serve_forever()
