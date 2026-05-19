# PCB Resistor Presence Checker

当前目录已经收口到单模板版本，核心思路是：

1. 用 `9-1.jpg` 作为模板图。
2. 所有图片先标准化到模板画布尺寸。
3. 做几何配准后，在固定 `detect_roi` 内判断是否存在电阻。
4. `detect_roi` 会再拆成左右两组，每组单独判断“白锡暴露 + 绿板暴露 + 上下焊锡结构”，只要有一组不合格就判 `NG`。

## 保留文件

1. `OneDrive_1_5-18-2026/`: 原始图片。
2. `config.yaml`: 当前主配置，里面直接带调参注释。
3. `config.json`: 兼容旧命令的等价配置。
4. `detect_resistor_presence.py`: 主检测脚本。
5. `annotate_rect_web.py`: WSL 可用的浏览器标注工具。
6. `evaluate_results.py`: 基于 `-1/-2` 文件名标签的评估脚本。
7. `requirements.txt`: 依赖说明。
8. `README.md`: 当前说明文档。

## 安装依赖

```bash
/bin/python3 -m pip install -r /isilon01/scripts/pcb_resistor_checker/requirements.txt
```

## 路径兼容

`config.yaml` 里的 `template_image` 现在支持两种写法：

1. 单个字符串路径。
2. 按平台拆开的路径映射，例如：

```yaml
template_image:
  windows: 'D:\bosch_project\workspace\resistor_detector\detection_results\smd_components\bad\20260519_092116_bad.png'
  linux: '/mnt/d/bosch_project/workspace/resistor_detector/detection_results/smd_components/bad/20260519_092116_bad.png'
```

脚本会优先读取当前平台对应的键；如果只写了 Windows 盘符路径，WSL/Linux 也会自动尝试转换成 `/mnt/<drive>/...`。命令行参数 `--image`、`--config`、`--glob`、`--debug-dir`、`--output-json`、`--output-config` 也会走同一套解析逻辑。

## 标注锚点和 ROI

浏览器版工具会自动从 `config.yaml` 里读取当前锚点名字，所以现在不用再手写 `--region-names`。

```bash
/bin/python3 /isilon01/scripts/pcb_resistor_checker/annotate_rect_web.py \
  --image /mnt/d/bosch_project/workspace/resistor_detector/detection_results/smd_components/bad/20260519_092116_bad.png \
  --config /isilon01/scripts/pcb_resistor_checker/config.yaml \
  --output-config /isilon01/scripts/pcb_resistor_checker/config.yaml
```

然后在 Windows 浏览器打开：

```text
http://localhost:8765
```

操作方式：

1. 依次拖框标注 `anchor_left`、`anchor_mid`、`anchor_right`、`detect_roi`。
2. 左侧点区域卡片可以切换要重画的区域。
3. `Save To File` 会直接写回 `config.yaml`，并保留这套内置参数注释模板。
4. 完成后回终端按 `Ctrl+C` 停服务。

## 运行检测

单张图：

```bash
/bin/python3 /isilon01/scripts/pcb_resistor_checker/detect_resistor_presence.py \
  --config /isilon01/scripts/pcb_resistor_checker/config.yaml \
  --image /isilon01/scripts/pcb_resistor_checker/OneDrive_1_5-18-2026/1-1.jpg \
  --debug-dir /isilon01/scripts/pcb_resistor_checker/debug
```

整批图：

```bash
/bin/python3 /isilon01/scripts/pcb_resistor_checker/detect_resistor_presence.py \
  --config /isilon01/scripts/pcb_resistor_checker/config.yaml \
  --glob '/isilon01/scripts/pcb_resistor_checker/OneDrive_1_5-18-2026/*.jpg' \
  --output-json /isilon01/scripts/pcb_resistor_checker/results.json \
  --debug-dir /isilon01/scripts/pcb_resistor_checker/debug
```

## 二分类模型版

旧规则版 `detect_resistor_presence.py` 保留不动。新模型版当前基线是：

1. 复用前面的配准和 `detect_roi` 定位。
2. 把对齐后的 `detect_roi` 裁出来。
3. 用 HOG + Linear SVM 做 `good/bad` 二分类。

当前新增脚本：

1. `extract_detect_roi_dataset.py`: 从完整图中裁训练 ROI。
2. `train_detect_roi_hog_svm.py`: 训练 HOG + Linear SVM 模型。
3. `infer_detect_roi_hog_svm.py`: 输入完整图，定位 ROI 后输出 `good/bad`。

### 1. 生成训练集 ROI

```powershell
c:/Users/OZJ6SZH/.conda/envs/3d-bat/python.exe "D:\bosch_project\workspace\new_script_jd\gt_scripts_jdz\pcb_resistor_checker\extract_detect_roi_dataset.py" --config "D:\bosch_project\workspace\new_script_jd\gt_scripts_jdz\pcb_resistor_checker\config.yaml" --input-root "D:\bosch_project\workspace\resistor_detector\detection_results\smd_components" --output-root "D:\bosch_project\workspace\resistor_detector\detection_results\smd_components_train"
```

输出目录会生成：

1. `smd_components_train/good`
2. `smd_components_train/bad`
3. `smd_components_train/metadata.csv`

### 2. 训练二分类模型

```powershell
c:/Users/OZJ6SZH/.conda/envs/3d-bat/python.exe "D:\bosch_project\workspace\new_script_jd\gt_scripts_jdz\pcb_resistor_checker\train_detect_roi_hog_svm.py" --dataset-root "D:\bosch_project\workspace\resistor_detector\detection_results\smd_components_train" --output-dir "D:\bosch_project\workspace\new_script_jd\gt_scripts_jdz\pcb_resistor_checker\models\hog_svm_v1" --folds 5
```

模型文件会保存到：

1. `models/hog_svm_v1/hog_svm.xml`
2. `models/hog_svm_v1/hog_svm.json`

### 3. 用完整图做推理

单张图：

```powershell
c:/Users/OZJ6SZH/.conda/envs/3d-bat/python.exe "D:\bosch_project\workspace\new_script_jd\gt_scripts_jdz\pcb_resistor_checker\infer_detect_roi_hog_svm.py" --config "D:\bosch_project\workspace\new_script_jd\gt_scripts_jdz\pcb_resistor_checker\config.yaml" --model-dir "D:\bosch_project\workspace\new_script_jd\gt_scripts_jdz\pcb_resistor_checker\models\hog_svm_v1" --image "D:\bosch_project\workspace\resistor_detector\detection_results\smd_components\bad\20260519_092116_bad.png" --result-only
```

整批图：

```powershell
c:/Users/OZJ6SZH/.conda/envs/3d-bat/python.exe "D:\bosch_project\workspace\new_script_jd\gt_scripts_jdz\pcb_resistor_checker\infer_detect_roi_hog_svm.py" --config "D:\bosch_project\workspace\new_script_jd\gt_scripts_jdz\pcb_resistor_checker\config.yaml" --model-dir "D:\bosch_project\workspace\new_script_jd\gt_scripts_jdz\pcb_resistor_checker\models\hog_svm_v1" --glob "D:\bosch_project\workspace\resistor_detector\detection_results\smd_components\bad\*.png" --output-json "D:\bosch_project\workspace\new_script_jd\gt_scripts_jdz\pcb_resistor_checker\results_bad_hog_svm.json"
```

模型版输出字段约定：

1. `result=good` 表示检测到电阻片。
2. `result=bad` 表示未检测到电阻片。
3. 如果定位失败，也会保守输出 `bad`，并额外带 `localization_reason`。

## 评估结果

当前约定：

1. `*-1` 表示真值 `OK`，即有电阻。
2. `*-2` 表示真值 `NG`，即无电阻。

评估命令：

```bash
/bin/python3 /isilon01/scripts/pcb_resistor_checker/evaluate_results.py \
  --results /isilon01/scripts/pcb_resistor_checker/results.json \
  --output-json /isilon01/scripts/pcb_resistor_checker/eval.json
```

输出会给出：

1. `TP/FP/FN/TN`
2. `accuracy`
3. `precision_ok`
4. `recall_ok`
5. `specificity_ng`
6. 误判名单和失败原因

## Debug 输出

每张图通常会生成：

1. `00_template_overlay.jpg`: 模板图上的锚点和 ROI。
2. `01_original_overlay.jpg`: 原图上的投影锚点和投影 ROI。
3. `02_aligned_overlay.jpg`: 对齐到模板坐标系后的图。
4. `03_binary_mask.png`: ROI 内暗色候选区域的二值图。
5. `04_metrics.json`: 当前图的详细特征和判定依据。

如果定位失败，则会看到：

1. `01_original_failure.jpg`
2. `04_metrics.json`

### 03_binary_mask.png 怎么看

1. 白色表示在 ROI 内被当前暗色阈值判成“黑色候选”的像素。
2. 黑色表示没有被判成黑色候选的像素，或者 ROI 外部。
3. 它的作用是检查“程序抓黑色是不是抓得太松或太严”，不是最终 `OK/NG` 分割结果。

观察建议：

1. 有电阻时，左右两颗电阻本体通常会在各自那一半里留下比较稳定的白色竖向暗块。
2. 无电阻时，黑块通常会变碎、变少，或者直接被更大的白焊锡和青绿色 PCB 替代。
3. 阴影、反光边缘、脏污也可能变成白块，所以要同时看 `04_metrics.json` 里的 `side_results.left` 和 `side_results.right`。

当前默认值已经把 `local_dark_offset` 从 `35` 放宽到 `30`，目的是让 `03_binary_mask.png` 更容易看出电阻黑体的大体形状。

按 `03_binary_mask.png` 调黑色阈值时，可以这样看：

1. 如果真有电阻，但 `03_binary_mask.png` 里电阻主体只剩零碎几块，说明抓黑太严了。先尝试把 `fixed_black_threshold` 调大 `5`，或者把 `local_dark_offset` 调小 `5`。
2. 如果没电阻时，大片阴影、绿板、背景也被刷成白色，说明抓黑太松了。先尝试把 `fixed_black_threshold` 调小 `5`，或者把 `local_dark_offset` 调大 `5`。
3. 如果白块形状大体对，但满是小孔、小噪点，优先调 `morph_kernel`。噪点太多就调大一点，细长黑块被吃掉就调小一点。
4. `dynamic_black_threshold` 的实际值会写在 `04_metrics.json` 里。它不是固定常数，而是根据周围亮度动态算出来的，所以调参时要同时看 `03_binary_mask.png` 和 `04_metrics.json`。
5. 如果 `03_binary_mask.png` 看起来已经合理，但结果仍然错了，就不要再先动黑色阈值，而是看 `side_results` 里到底是白焊锡暴露过多，还是绿板暴露过多。
6. 如果你看到某些 `NG` 图里某一侧还是被放成 `OK`，重点看 `center_vertical_aspect`。它太小通常说明那块黑色更像散块或短粗块，不像竖着的电阻本体。

## 当前判定规则

`config.yaml` 现在使用：

1. `preprocess.standardize_to_template=true`
2. `preprocess.standardize_mode=fit_pad`
3. `decision.mode=paired_resistors_lr`

含义：

1. 所有图先按比例缩放，再 padding 到模板 `9-1.jpg` 的画布尺寸。
2. 不直接强行拉伸，尽量避免 PCB 比例失真。
3. 判定阶段把 `detect_roi` 拆成左半和右半，分别看每一组电阻是否成立。

当前最终判定顺序：

1. 先把 `detect_roi` 拆成左组和右组，两组分别计算自己的特征。
2. 左组必须同时满足 `side_center_white_ratio_max=0.34`、`side_center_green_ratio_max=0.28`、`side_center_vertical_aspect_min=1.5`、`side_white_big_count_min=4`、`side_top_white_count_min=1`、`side_bottom_white_count_min=1`。
3. 右组也必须满足同样条件。
4. 只要左组或右组任一组失败，整张图就是 `NG`。
5. 只有左右两组都合格，整张图才是 `OK`。

字段解释：

1. `side_results.left` / `side_results.right`: 左右两组各自的判定结果。
2. `center_white_ratio`: 这一组电阻中间区域里，亮白焊锡像素占比。越大，越像没有被电阻遮住。
3. `center_green_ratio`: 这一组电阻中间区域里，青绿色 PCB 像素占比。越大，越像没贴电阻。
4. `center_vertical_aspect`: 这一组中间最大黑块的竖直细长度，越大越像竖着放的电阻本体。
5. `white_big_count`: 这一组里面积足够大的白焊锡连通域数量。
6. `top_white_count` / `bottom_white_count`: 这一组上端和下端还能看到多少个白焊锡块。
7. `decision_reason`: 本次命中的具体规则分支，例如 `left_reject` 或 `right_reject`。

这套规则对应你的业务理解：

1. 每一组都应该看到“黑色电阻本体压在上下白焊锡之间”的结构。
2. 如果某一组没有电阻，白焊锡会露得更大，中间更容易看到绿板。
3. 所以当前不是问“整块 ROI 看起来像不像有东西”，而是问“左边这颗像不像装好了，右边这颗像不像也装好了”。
4. `04_metrics.json` 里会直接告诉你左右哪一组失败、失败在哪条规则上。

