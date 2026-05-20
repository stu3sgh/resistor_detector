"""
KNN 图片分类器 — 基于 HOG/颜色直方图/纹理特征的 K-近邻分类

输入: PNG 图片 (文件路径 / bytes / PIL Image)
输出: dict with "result" and "confidence"

result:
  - "good" → 良品 (label 0)
  - "bad"  → 不良品 (label 1)
"""

import os
import io
import numpy as np
from PIL import Image, ImageFilter


class KNNClassifier:
    """KNN 图片分类器，基于图像特征向量的余弦相似度"""

    def __init__(self, data_dir, k=3):
        """
        Args:
            data_dir: 训练数据目录，支持 good/bad 子目录结构
            k: K-近邻数量
        """
        self.data_dir = data_dir
        self.features = []
        self.labels = []
        self.k = k
        self.load_data()

    def load_data(self):
        """从 data_dir 加载训练样本"""
        if not os.path.exists(self.data_dir):
            return
        # 优先使用 good/bad 子目录结构，兼容旧的扁平结构
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
        print(f"[knn_classifier] Loaded {len(self.labels)} samples: "
              f"{(self.labels == 0).sum()} good, {(self.labels == 1).sum()} bad")

    def _extract_features(self, img_input):
        """提取图像特征向量 (85维)"""
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

        # 颜色空间特征
        r, g, b = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]
        is_pcb = (g > r) & (g > b) & (g > 60)
        is_silver = (r > 160) & (g > 160) & (b > 160)
        is_dark = gray < 40
        features.extend([is_pcb.sum() / arr.size, is_silver.sum() / arr.size, is_dark.sum() / arr.size])

        # 银色区域纹理
        if is_silver.sum() > 10:
            sp = arr[is_silver]
            features.append(np.std(sp) / 255)
        else:
            features.append(0.5)

        # 高频细节 (锐度)
        blur = np.array(img.filter(ImageFilter.GaussianBlur(radius=3)), dtype=float)
        features.append(np.abs(arr - blur).mean() / 255)

        # 行亮度均匀性
        row_means = gray.mean(axis=1)
        features.append(np.std(row_means) / 255)
        features.append(np.std(np.diff(row_means)) / 255)

        # RGB 直方图 (3×16 = 48 维)
        for c_idx in range(3):
            ch = arr[:, :, c_idx]
            hist, _ = np.histogram(ch, bins=16, range=(0, 256))
            features.extend((hist / hist.sum()).tolist())

        # 4×4 网格灰度均值 (16 维)
        for iy in range(4):
            for ix in range(4):
                y1, y2 = iy * h // 4, (iy + 1) * h // 4
                x1, x2 = ix * w // 4, (ix + 1) * w // 4
                features.append(gray[y1:y2, x1:x2].mean() / 255)

        return np.array(features)

    def _cosine_sim(self, a, b):
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10)

    def predict(self, img_input):
        """
        分类预测

        Args:
            img_input: PNG 图片 (文件路径 / bytes / PIL Image)

        Returns:
            dict: {
                "result": "good" 或 "bad",
                "confidence": 0~1 置信度,
                "details": { ... }
            }
        """
        if len(self.features) < 2:
            return {
                "result": "good",
                "confidence": 0.5,
                "details": {"error": "not enough training data"}
            }
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
        return {
            "result": "good" if pred == 0 else "bad",
            "confidence": min(confidence, 1.0),
            "details": {
                "avg_good_sim": round(avg_good, 4),
                "avg_bad_sim": round(avg_bad, 4),
                "k": self.k,
                "total_samples": len(self.labels),
                "good_samples": int((self.labels == 0).sum()),
                "bad_samples": int((self.labels == 1).sum()),
                "top_k": [{"sim": round(s, 4), "label": "good" if l == 0 else "bad"} for s, l in topk]
            }
        }

    def reload(self):
        """重新加载训练数据"""
        self.features = []
        self.labels = []
        self.load_data()


if __name__ == '__main__':
    import sys
    import json

    if len(sys.argv) < 3:
        print("Usage: python knn_classifier.py <data_dir> <image_path>")
        sys.exit(1)

    data_dir = sys.argv[1]
    image_path = sys.argv[2]

    clf = KNNClassifier(data_dir)
    result = clf.predict(image_path)
    print(json.dumps(result, indent=2, ensure_ascii=False))
