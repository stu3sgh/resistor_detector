# 产线良品识别 - AI 智能检测

基于 AI 视觉的产线元件质量检测系统，支持 SMD 元件、主芯片、底部芯片的良品/不良品分类。

## 功能

- 📷 拍照识别元件缺陷
- 🤖 KNN 分类器自动判断良品/不良品
- 📊 检测结果保存与统计

## 数据集

`detection_results/` 目录包含标注的子图数据集：

| 区域 | good | bad | 用途 |
|------|------|-----|------|
| smd_components | SMD元件 | 外观检测 |
| main_chip | 主芯片 | 焊点检测 |
| bottom_chip | 底部芯片 | 引脚检测 |

## 分类器接口规范（Classifier API Spec）

后端统一运行在端口 **8000**，分类器路由为 `/classify`。

### 输入（POST /classify）

**Content-Type:** `application/json`

```json
{
  "image": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAA..."
}
```

| 字段 | 类型 | 说明 |
|------|------|------|
| `image` | string | **Data URL 格式的 PNG 图片**（`data:image/png;base64,<base64编码>`） |

图片来源：从裁剪后的 PCB 板图中按区域截取的子图（如 SMD 元件区域 340×215px 左右）。

### 输出

```json
{
  "verdict": "good",
  "confidence": 0.8521,
  "avg_good_sim": 0.9123,
  "avg_bad_sim": 0.0602,
  "k": 3,
  "total_samples": 21,
  "good_samples": 9,
  "bad_samples": 12,
  "top_k": [
    {"sim": 0.9123, "label": 0},
    {"sim": 0.8856, "label": 0},
    {"sim": 0.0602, "label": 1}
  ]
}
```

| 字段 | 类型 | 说明 |
|------|------|------|
| `verdict` | string | **`"good"` = 良品**，**`"bad"` = 不良品** |
| `confidence` | float | 置信度（0~1），越高越确定 |
| `avg_good_sim` | float | 与良品样本的平均相似度 |
| `avg_bad_sim` | float | 与不良品样本的平均相似度 |
| `k` | int | KNN 的 k 值 |
| `total_samples` | int | 训练样本总数 |
| `good_samples` | int | 良品训练样本数 |
| `bad_samples` | int | 不良品训练样本数 |
| `top_k` | array | K 个最近邻的详细信息（`label: 0` = 良品，`label: 1` = 不良品） |

### verdict 映射关系

| verdict | 含义 | 内部 label |
|---------|------|-----------|
| `"good"` | **良品** | `0` |
| `"bad"` | **不良品** | `1` |

### 状态查询（GET /classify）

```json
{
  "status": "ready",
  "samples": 21,
  "good": 9,
  "bad": 12
}
```

### 重载训练数据（POST /classify/reload）

```json
{
  "ok": true,
  "samples": 21
}
```

> 调用后重新从 `detection_results/smd_components/` 加载训练数据（新增 good/bad 标签图片后使用）。

---

## 部署

```bash
# 启动后端（统一端口 8000）
python3 server.py

# 前端通过 nginx 提供静态服务
# /resistor → 静态页面
# /resistor/api/ → 后端 API
```
