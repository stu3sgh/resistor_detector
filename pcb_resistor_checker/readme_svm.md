# PCB Resistor Checker SVM Inference Guide

## 1. 这份文档的范围

这份文档只说明 SVM 模型版的推理接入方式。

下游集成时，只需要关心：

1. 输入给哪个脚本。
2. 输入图片是什么形式。
3. 模型文件放在哪里。
4. 输出 JSON 长什么样。

训练、规则判定、调参过程都不是本文档重点。

## 2. 输入是什么

输入给推理脚本的图片，不是整张原始大图，也不是已经精确裁好的 `detect_roi` 小块。

输入图片是：

1. 从上游大图中先裁出来的一张局部图。
2. 这张局部图里包含电阻片区域。
3. 脚本会在这张局部图内部继续做模板匹配、定位 `detect_roi`、特征提取和二分类。

可以把它理解为：

1. 上游先给一张“包含目标器件区域的完整局部图”。
2. 本脚本再在这张局部图里做最终定位和分类。

## 3. 输出是什么

推理输出是 JSON。

结果字段约定：

1. `result=good`：检测到电阻片。
2. `result=bad`：未检测到电阻片。
3. 如果 `detect_roi` 定位失败，也输出 `bad`，并附带 `localization_reason`。

## 4. 直接可用的文件

当前目录里，推理直接依赖以下文件：

1. `infer_detect_roi_hog_svm.py`
2. `detect_resistor_presence.py`
3. `roi_classifier_common.py`
4. `roi_classifier_hog_svm.py`
5. `config.yaml`
6. `models/hog_svm_v1/hog_svm.xml`
7. `models/hog_svm_v1/hog_svm.json`

其中：

1. `hog_svm.xml` 是模型权重。
2. `hog_svm.json` 是模型配置和元数据。
3. 这两个文件已经放在目录里，下游可以直接用，不需要重新训练。

## 5. 推理脚本

推理入口是：

`infer_detect_roi_hog_svm.py`

它的内部流程是：

1. 读取输入局部图。
2. 根据 `config.yaml` 加载模板图、锚点框、`detect_roi`。
3. 在当前局部图里做 ORB 配准。
4. 将图对齐到模板坐标系。
5. 裁出 `detect_roi`。
6. 对 ROI 做 HOG 特征提取。
7. 加载 `models/hog_svm_v1` 里的模型。
8. 输出 `good/bad`。

## 6. 命令行用法

### 单张图，只看终端结果

```powershell
python.exe "D:\bosch_project\workspace\new_script_jd\gt_scripts_jdz\pcb_resistor_checker\infer_detect_roi_hog_svm.py" --config "D:\bosch_project\workspace\new_script_jd\gt_scripts_jdz\pcb_resistor_checker\config.yaml" --model-dir "D:\bosch_project\workspace\new_script_jd\gt_scripts_jdz\pcb_resistor_checker\models\hog_svm_v1" --image "D:\bosch_project\workspace\resistor_detector\detection_results\smd_components\bad\20260519_092116_bad.png" --result-only
```

### 单张图，输出到 JSON

```powershell
python.exe "D:\bosch_project\workspace\new_script_jd\gt_scripts_jdz\pcb_resistor_checker\infer_detect_roi_hog_svm.py" --config "D:\bosch_project\workspace\new_script_jd\gt_scripts_jdz\pcb_resistor_checker\config.yaml" --model-dir "D:\bosch_project\workspace\new_script_jd\gt_scripts_jdz\pcb_resistor_checker\models\hog_svm_v1" --image "D:\bosch_project\workspace\resistor_detector\detection_results\smd_components\bad\20260519_092116_bad.png" --output-json "D:\bosch_project\workspace\new_script_jd\gt_scripts_jdz\pcb_resistor_checker\one_result.json"
```

### 批量图片，输出到 JSON

```powershell
python.exe "D:\bosch_project\workspace\new_script_jd\gt_scripts_jdz\pcb_resistor_checker\infer_detect_roi_hog_svm.py" --config "D:\bosch_project\workspace\new_script_jd\gt_scripts_jdz\pcb_resistor_checker\config.yaml" --model-dir "D:\bosch_project\workspace\new_script_jd\gt_scripts_jdz\pcb_resistor_checker\models\hog_svm_v1" --glob "D:\bosch_project\workspace\resistor_detector\detection_results\smd_components\good\*.png" --output-json "D:\bosch_project\workspace\new_script_jd\gt_scripts_jdz\pcb_resistor_checker\results_good_hog_svm.json"
```

## 7. 参数说明

推理脚本主要参数如下：

1. `--config`
模板图、锚点、`detect_roi` 的配置文件。

2. `--model-dir`
模型目录，当前直接指向：
`models/hog_svm_v1`

3. `--image`
单张输入图路径。

4. `--glob`
批量输入图路径模式。

5. `--output-json`
结果 JSON 输出路径。

6. `--result-only`
单张图时终端只输出 `good` 或 `bad`。

## 8. 输出 JSON 结构

即使输入是一张图，输出文件里也是数组，长度为 1。

典型成功结果：

```json
[
  {
    "image": "D:/.../20260519_092116_bad.png",
    "result": "bad",
    "reason": "hog_linear_svm",
    "localization_ok": true,
    "model_type": "opencv_hog_linear_svm",
    "total_good_matches": 240,
    "inlier_count": 240,
    "transform_type": "affine_partial",
    "transform_summary": {
      "scale": 1.0,
      "angle_deg": 0.0,
      "tx": 0.0,
      "ty": 0.0
    },
    "roi_bbox": {
      "x": 128,
      "y": 126,
      "width": 83,
      "height": 79
    },
    "svm_label_id": -1,
    "svm_raw_output": 1.0,
    "svm_margin_abs": 1.0
  }
]
```

定位失败结果：

```json
[
  {
    "image": "D:/.../xxx.png",
    "result": "bad",
    "reason": "localization_failed",
    "localization_reason": "not_enough_anchor_matches",
    "localization_ok": false
  }
]
```

## 9. 下游最简单的集成方式

最简单的方式就是直接命令行调用 `infer_detect_roi_hog_svm.py`。

下游只需要保证：

1. 传入的是“包含电阻片区域的局部图”。
2. `--config` 指向正确的 `config.yaml`。
3. `--model-dir` 指向已经放好的 `models/hog_svm_v1`。
4. 读取输出 JSON 里的 `result` 字段即可。

对下游来说，核心判断字段就是：

1. `result=good`
2. `result=bad`

## 10. Python 直接调用方式

如果下游不想走命令行，也可以直接在 Python 里调用当前代码。

```python
from pathlib import Path

from detect_resistor_presence import load_config, load_template_state
from roi_classifier_common import localize_detect_roi, mask_roi_crop
from roi_classifier_hog_svm import load_model_bundle, compute_hog_features, predict_label_ids, decode_label

config_path = Path(r"D:\bosch_project\workspace\new_script_jd\gt_scripts_jdz\pcb_resistor_checker\config.yaml")
model_dir = Path(r"D:\bosch_project\workspace\new_script_jd\gt_scripts_jdz\pcb_resistor_checker\models\hog_svm_v1")

config = load_config(config_path)
template_state = load_template_state(config, config_path)
svm, feature_config, _ = load_model_bundle(model_dir)

def classify_patch_image(image_path: str) -> dict:
    localized, payload = localize_detect_roi(Path(image_path), config, template_state)
    if localized is None:
        return {
            "result": "bad",
            "reason": "localization_failed",
            "localization_reason": payload.get("reason", "unknown"),
            "localization_ok": False,
        }

    masked_roi = mask_roi_crop(localized.roi_crop_bgr, localized.roi_mask)
    features = compute_hog_features(masked_roi, feature_config).reshape(1, -1)
    predicted_ids, raw_scores = predict_label_ids(svm, features)
    return {
        "result": decode_label(int(predicted_ids[0])),
        "reason": "hog_linear_svm",
        "localization_ok": True,
        "svm_raw_output": float(raw_scores[0]),
    }
```

## 11. 当前结论

当前目录里的模型已经训练完成，并且权重已经放在：

1. `models/hog_svm_v1/hog_svm.xml`
2. `models/hog_svm_v1/hog_svm.json`

因此下游当前要做的事情不是重新训练，而是直接调用推理脚本或直接引用推理代码。