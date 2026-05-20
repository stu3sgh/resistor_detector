# Multi-Region V2 训练与推理说明

## 1. 文档范围

这份文档说明三类模型的正常训练流程和 V2 推理调用方式：

1. `smd_components`
2. `main_chip`
3. `bottom_chip`

## 2. 运行环境

下面所有命令都可以直接使用这套 Python 环境：

```powershell
c:/Users/OZJ6SZH/.conda/envs/3d-bat/python.exe
```

如果你更习惯先激活 Conda 环境，也可以：

```powershell
conda activate c:\Users\OZJ6SZH\.conda\envs\3d-bat
```

## 3. 模型目录命名规则

为了让 V2 推理脚本自动选择最新模型，模型目录建议按下面的规则命名：

1. `smd_components` -> `models/hog_svm_v1`、`models/hog_svm_v2`、`models/hog_svm_v3`
2. `main_chip` -> `models/main_chip_hog_svm_v1`、`models/main_chip_hog_svm_v2`
3. `bottom_chip` -> `models/bottom_chip_hog_svm_v1`、`models/bottom_chip_hog_svm_v2`

V2 推理脚本默认会自动扫描 `models/`，并选择每个区域版本号最高的目录。

## 4. 通用训练脚本

通用训练入口是：

```powershell
D:/bosch_project/workspace/new_PCB_nan/resistor_detector/pcb_resistor_checker/train_detect_roi_hog_svm.py
```

只要训练数据目录结构满足下面这种形式，就可以直接训练：

1. `good/`
2. `bad/`

## 5. `smd_components` 的正常训练流程

`smd_components` 的训练分两步：

1. 先从原始局部图里提取 `detect_roi` 训练样本。
2. 再用提取出的 ROI 数据训练 SVM。

### 第一步：提取 `detect_roi` 数据集

```powershell
c:/Users/OZJ6SZH/.conda/envs/3d-bat/python.exe D:/bosch_project/workspace/new_PCB_nan/resistor_detector/pcb_resistor_checker/extract_detect_roi_dataset.py --config D:/bosch_project/workspace/new_PCB_nan/resistor_detector/pcb_resistor_checker/config.yaml --input-root D:/bosch_project/workspace/new_PCB_nan/resistor_detector/detection_results/smd_components --output-root D:/bosch_project/workspace/new_PCB_nan/resistor_detector/detection_results/smd_components_train
```

如果你不想覆盖现有提取结果，可以把输出目录换成版本化名字，例如：

```powershell
--output-root D:/bosch_project/workspace/new_PCB_nan/resistor_detector/detection_results/smd_components_train_v2
```

### 第二步：训练 `smd_components` 模型

```powershell
c:/Users/OZJ6SZH/.conda/envs/3d-bat/python.exe D:/bosch_project/workspace/new_PCB_nan/resistor_detector/pcb_resistor_checker/train_detect_roi_hog_svm.py --dataset-root D:/bosch_project/workspace/new_PCB_nan/resistor_detector/detection_results/smd_components_train --output-dir D:/bosch_project/workspace/new_PCB_nan/resistor_detector/pcb_resistor_checker/models/hog_svm_v1 --region-name smd_components
```

如果你想保留旧模型，建议直接训到新版本目录，例如：

```powershell
--output-dir D:/bosch_project/workspace/new_PCB_nan/resistor_detector/pcb_resistor_checker/models/hog_svm_v2
```

如果当前 `good/bad` 比例差异比较大，可以在训练命令后面加：

```powershell
--balance oversample
```

## 6. 训练 `main_chip`

`main_chip` 不需要 ROI 二次提取，可以直接训练。

推荐命令：

```powershell
c:/Users/OZJ6SZH/.conda/envs/3d-bat/python.exe D:/bosch_project/workspace/new_PCB_nan/resistor_detector/pcb_resistor_checker/train_detect_roi_hog_svm.py --dataset-root D:/bosch_project/workspace/new_PCB_nan/resistor_detector/detection_results/main_chip --output-dir D:/bosch_project/workspace/new_PCB_nan/resistor_detector/pcb_resistor_checker/models/main_chip_hog_svm_v1 --region-name main_chip --balance oversample
```

如果你要新版本，直接改输出目录，例如：

```powershell
--output-dir D:/bosch_project/workspace/new_PCB_nan/resistor_detector/pcb_resistor_checker/models/main_chip_hog_svm_v2
```

## 7. 训练 `bottom_chip`

`bottom_chip` 也是直接训练。

推荐命令：

```powershell
c:/Users/OZJ6SZH/.conda/envs/3d-bat/python.exe D:/bosch_project/workspace/new_PCB_nan/resistor_detector/pcb_resistor_checker/train_detect_roi_hog_svm.py --dataset-root D:/bosch_project/workspace/new_PCB_nan/resistor_detector/detection_results/bottom_chip --output-dir D:/bosch_project/workspace/new_PCB_nan/resistor_detector/pcb_resistor_checker/models/bottom_chip_hog_svm_v1 --region-name bottom_chip --balance oversample
```

如果你要保留旧版本，改成新目录，例如：

```powershell
--output-dir D:/bosch_project/workspace/new_PCB_nan/resistor_detector/pcb_resistor_checker/models/bottom_chip_hog_svm_v2
```

## 8. 训练产物是什么

每个模型目录都会产出两个核心文件：

1. `hog_svm.xml`
2. `hog_svm.json`

其中：

1. `hog_svm.xml` 是 OpenCV 的 SVM 权重文件。
2. `hog_svm.json` 记录特征配置和训练元数据。

终端还会打印验证指标，通常重点看：

1. `accuracy`
2. `recall_bad`
3. `precision_bad`

## 9. V2 推理脚本

三路联合推理入口是：

```powershell
D:/bosch_project/workspace/new_PCB_nan/resistor_detector/pcb_resistor_checker/infer_multi_region_hog_svm_v2.py
```

默认行为是：

1. 自动扫描 `models/`
2. 自动选择每个区域版本号最高的模型目录
3. 不需要在命令里手动写三种模型路径

## 10. V2 推理命令

最常用的三路推理命令如下：

```powershell
c:/Users/OZJ6SZH/.conda/envs/3d-bat/python.exe D:/bosch_project/workspace/new_PCB_nan/resistor_detector/pcb_resistor_checker/infer_multi_region_hog_svm_v2.py --smd-components-image D:/bosch_project/workspace/new_PCB_nan/resistor_detector/detection_results/smd_components/good/20260519_131630_695743.png --main-chip-image D:/bosch_project/workspace/new_PCB_nan/resistor_detector/detection_results/main_chip/good/20260519_131630_695743.png --bottom-chip-image D:/bosch_project/workspace/new_PCB_nan/resistor_detector/detection_results/bottom_chip/good/20260519_131630_695743.png
```

如果还想把结果落成 JSON 文件，加上：

```powershell
--output-json D:/bosch_project/workspace/new_PCB_nan/resistor_detector/pcb_resistor_checker/result_v2.json
```

完整示例如下：

```powershell
c:/Users/OZJ6SZH/.conda/envs/3d-bat/python.exe D:/bosch_project/workspace/new_PCB_nan/resistor_detector/pcb_resistor_checker/infer_multi_region_hog_svm_v2.py --smd-components-image D:/bosch_project/workspace/new_PCB_nan/resistor_detector/detection_results/smd_components/good/20260519_131630_695743.png --main-chip-image D:/bosch_project/workspace/new_PCB_nan/resistor_detector/detection_results/main_chip/good/20260519_131630_695743.png --bottom-chip-image D:/bosch_project/workspace/new_PCB_nan/resistor_detector/detection_results/bottom_chip/good/20260519_131630_695743.png --output-json D:/bosch_project/workspace/new_PCB_nan/resistor_detector/pcb_resistor_checker/result_v2.json
```
输出json示例: 仓内的three_results.json

## 11. 如果需要手动覆盖模型目录

一般情况下不需要指定模型目录。

只有在你想临时测试某个指定版本时，才需要使用：

1. `--smd-components-model-dir`
2. `--main-chip-model-dir`
3. `--bottom-chip-model-dir`

如果你只是正常使用，直接省略即可。

## 12. 如果需要同步到镜像副本

如果你还希望镜像副本 `new_script_jd/gt_scripts_jdz/pcb_resistor_checker` 也使用相同的新模型，可以直接复制对应目录：

```powershell
Copy-Item -Path D:/bosch_project/workspace/new_PCB_nan/resistor_detector/pcb_resistor_checker/models/hog_svm_v2 -Destination D:/bosch_project/workspace/new_script_jd/gt_scripts_jdz/pcb_resistor_checker/models -Recurse -Force
Copy-Item -Path D:/bosch_project/workspace/new_PCB_nan/resistor_detector/pcb_resistor_checker/models/main_chip_hog_svm_v2 -Destination D:/bosch_project/workspace/new_script_jd/gt_scripts_jdz/pcb_resistor_checker/models -Recurse -Force
Copy-Item -Path D:/bosch_project/workspace/new_PCB_nan/resistor_detector/pcb_resistor_checker/models/bottom_chip_hog_svm_v2 -Destination D:/bosch_project/workspace/new_script_jd/gt_scripts_jdz/pcb_resistor_checker/models -Recurse -Force
```

如果你当前训练出来的还是 `v1`，就把命令里的目录名改成 `v1`。

## 13. 建议的标准流程

建议你后续按下面顺序做：

1. 如果需要，先提取 `smd_components` 的 ROI 数据集。
2. 训练 `smd_components`。
3. 训练 `main_chip`。
4. 训练 `bottom_chip`。
5. 直接运行一次 V2 推理命令检查输出。
6. 如果训了更高版本目录，V2 会自动使用版本号最高的模型。