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

## 部署

```bash
# 启动后端（统一端口 8000）
python3 server.py

# 前端通过 nginx 提供静态服务
# /resistor → 静态页面
# /resistor/api/ → 后端 API
```
