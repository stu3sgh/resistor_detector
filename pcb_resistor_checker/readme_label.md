# PCB Resistor Checker Relabel Guide

## 1. 什么时候需要看这份文档

这份文档用于重新标注当前仓里的模板图：

1. `template.png`

当前 `config.yaml` 里保留的锚点框和 `detect_roi` 坐标不是这张 `template.png` 上的有效坐标，所以在推理前需要重新标注。

## 2. 标注目标

需要在 `template.png` 上重新标 4 个区域：

1. `anchor_left`
2. `anchor_mid`
3. `anchor_right`
4. `detect_roi`

其中：

1. 3 个 `anchor_*` 用来做 ORB 配准。
2. `detect_roi` 是后续做 SVM 二分类的目标区域。

## 3. 启动标注工具

在 PowerShell 中执行下面这条命令：

```powershell
c:/Users/OZJ6SZH/.conda/envs/3d-bat/python.exe "D:\bosch_project\workspace\new_script_jd\gt_scripts_jdz\pcb_resistor_checker\annotate_rect_web.py" --image "D:\bosch_project\workspace\new_script_jd\gt_scripts_jdz\pcb_resistor_checker\template.png" --config "D:\bosch_project\workspace\new_script_jd\gt_scripts_jdz\pcb_resistor_checker\config.yaml" --output-config "D:\bosch_project\workspace\new_script_jd\gt_scripts_jdz\pcb_resistor_checker\config.yaml"
```

启动成功后，终端会打印：

1. `Open in browser: http://localhost:8765`
2. `Listening on: http://127.0.0.1:8765`

然后在浏览器里打开：

```text
http://localhost:8765
```

## 4. 标注操作步骤

由于 `config.yaml` 里当前还是旧模板上的坐标，进入页面后先执行下面的动作：

1. 点击左侧 `Clear All`，清空旧框。
2. 按顺序重新画 `anchor_left`。
3. 再画 `anchor_mid`。
4. 再画 `anchor_right`。
5. 最后画 `detect_roi`。
6. 点击 `Save To File`，把新坐标写回 `config.yaml`。
7. 回到终端按 `Ctrl+C` 停掉标注服务。

## 5. 每个区域怎么选

### 5.1 `anchor_left` / `anchor_mid` / `anchor_right`

锚点框选择原则：

1. 选稳定不变的区域。
2. 选纹理明显的区域。
3. 尽量不要选纯色、反光太强、特征太少的地方。
4. 尽量让 3 个锚点分散在图里，不要都挤在一起。

否则推理时容易报：

1. `Anchor '<name>' has no ORB features`
2. `not_enough_anchor_matches`
3. `not_enough_inliers`

### 5.2 `detect_roi`

`detect_roi` 要框住后续要做二分类的目标区域。

要求：

1. 把电阻片区域完整框进去。
2. 不要框得过大，避免带入太多无关背景。
3. 也不要框得过小，避免把电阻主体截断。

## 6. 标注完成后的验证

标完并保存后，先用单张图做一次验证：

```powershell
python.exe "D:\bosch_project\workspace\new_script_jd\gt_scripts_jdz\pcb_resistor_checker\infer_detect_roi_hog_svm.py" --config "D:\bosch_project\workspace\new_script_jd\gt_scripts_jdz\pcb_resistor_checker\config.yaml" --model-dir "D:\bosch_project\workspace\new_script_jd\gt_scripts_jdz\pcb_resistor_checker\models\hog_svm_v1" --image "D:\bosch_project\workspace\resistor_detector\detection_results\smd_components\bad\20260519_092116_bad.png" --output-json "D:\bosch_project\workspace\new_script_jd\gt_scripts_jdz\pcb_resistor_checker\one_result.json"
```

如果运行成功，说明：

1. 模板图能正常读取。
2. 锚点框可以完成配准。
3. `detect_roi` 能正常定位。
4. 模型可以正常输出 JSON。

如果仍然报错，优先检查：

1. `anchor_mid` 或其他锚点是不是选在了纹理太少的位置。
2. `detect_roi` 是否框偏了。
3. 3 个锚点是否太靠边或者太集中。

## 7. 标注完成后是否要重新训练

如果你这次重新标注后，`detect_roi` 的位置、大小、模板坐标系和之前训练模型时相比有明显变化，建议重新生成训练数据并重新训练 SVM。

推荐顺序：

1. 先重新标注 `template.png`。
2. 再重新抽训练 ROI。
3. 再重新训练模型。
4. 最后再做批量推理。

## 8. 重新抽训练 ROI

```powershell
c:/Users/OZJ6SZH/.conda/envs/3d-bat/python.exe "D:\bosch_project\workspace\new_script_jd\gt_scripts_jdz\pcb_resistor_checker\extract_detect_roi_dataset.py" --config "D:\bosch_project\workspace\new_script_jd\gt_scripts_jdz\pcb_resistor_checker\config.yaml" --input-root "D:\bosch_project\workspace\resistor_detector\detection_results\smd_components" --output-root "D:\bosch_project\workspace\resistor_detector\detection_results\smd_components_train"
```

## 9. 重新训练模型

```powershell
c:/Users/OZJ6SZH/.conda/envs/3d-bat/python.exe "D:\bosch_project\workspace\new_script_jd\gt_scripts_jdz\pcb_resistor_checker\train_detect_roi_hog_svm.py" --dataset-root "D:\bosch_project\workspace\resistor_detector\detection_results\smd_components_train" --output-dir "D:\bosch_project\workspace\new_script_jd\gt_scripts_jdz\pcb_resistor_checker\models\hog_svm_v1" --folds 5
```

## 10. 标注完成后的整体验证顺序

推荐你按下面顺序走：

1. 起标注工具。
2. 清空旧框。
3. 重画 3 个锚点和 `detect_roi`。
4. 保存到 `config.yaml`。改template_image中windows: template.png
5. 用单张图跑一次 `infer_detect_roi_hog_svm.py`。
6. 如果单张图能通过，再重新抽训练集。
7. 重新训练 SVM。
8. 最后再跑批量推理。