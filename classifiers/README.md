# Classifiers

存放产线良品识别的分类器实现。

## 目录结构

```
classifiers/
├── knn_classifier.py   # KNN 分类器 (基于图像特征 + 余弦相似度)
└── (你的新分类器放这里)
```

## 接口规范

所有分类器必须实现统一的 `predict` 方法：

### 输入
- **格式**: PNG 图片（文件路径 / bytes / PIL Image）

### 输出 JSON
```json
{
    "result": "good",
    "confidence": 0.8521,
    "details": { ... }
}
```

- `result`: `"good"` = 良品, `"bad"` = 不良品
- `confidence`: 0~1 置信度

### 调用方式
```python
from classifiers.knn_classifier import KNNClassifier

clf = KNNClassifier("detection_results/smd_components")
result = clf.predict("test_image.png")
print(result["result"])     # "good" 或 "bad"
print(result["confidence"]) # 0.8521
```
