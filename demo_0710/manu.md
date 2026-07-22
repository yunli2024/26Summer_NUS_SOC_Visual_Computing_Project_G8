# Beginner Demo 操作手册

## 目标

本目录完成 Beginner Level：本地 webcam 实时显示、Haar Cascade 人脸检测、OpenCV LBF 68 点 facial landmarks 绘制、按键退出、鲁棒性观察，以及可选替代 face detector 对比。

## 文件说明

| 文件 | 作用 |
|---|---|
| `beginner_demo.py` | 主程序，打开摄像头并实时绘制 face box + 68 landmarks |
| `face_pipeline.py` | 可复用模块：Haar/YuNet/MediaPipe detector、LBF landmark、FPS、绘制函数 |
| `haarcascade_frontalface_default.xml` | OpenCV Haar face detector 模型 |
| `face_detection_yunet_2023mar.onnx` | Expert 实时模式默认使用的 OpenCV YuNet 人脸检测模型 |
| `lbfmodel.yaml` | OpenCV LBF 68-point facial landmark 模型 |
| `requirements.txt` | Beginner + Expert 基础依赖 |
| `robustness_notes.md` | 鲁棒性测试记录与汇报要点 |
| `train_expert.py` | Expert 训练与评估脚本：从 FER zip 提取 landmarks 特征并训练分类器 |
| `expert_demo.py` | Expert 实时 webcam 表情分类与 visual effects demo |
| `expression_features.py` | landmark 归一化、几何特征和 Expert 特征提取 |
| `realtime_stability.py` | Expert 实时模式稳定器：多人 track、误检过滤、表情概率平滑 |
| `fer_dataset.py` | 从 `facial_expression_dataset.zip` 验证并读取 train/test 七类数据 |
| `expression_effects.py` | 表情文字、置信度和实时视觉特效 |
| `expert_report.md` | 当前 Expert 方法、评估结果、confusion matrix 和改进分析 |

## 是否需要创建虚拟环境

建议创建独立环境。因为本项目依赖 `opencv-contrib-python`，它和普通 `opencv-python` 容易冲突；单独环境可以避免影响你电脑上其他课程或项目。

本项目统一使用 Conda 管理环境，不使用 `python -m venv`。

## 完整逐步操作流程

在 PowerShell 中进入本目录：

```powershell
cd D:\26spring_NUS_SOC_SWS\project\demo_0710
```

### Step 1：确认 Conda 可用

```powershell
conda --version
```

如果提示找不到 `conda`，请先安装 Anaconda 或 Miniconda，然后重新打开 PowerShell 或 Anaconda Prompt。

### Step 2：创建并进入 Conda 环境

如果是第一次运行本 demo，创建环境：

```powershell
conda create -n sws3026-face python=3.10 -y
conda activate sws3026-face
```

如果环境已经创建过，以后只需要：

```powershell
conda activate sws3026-face
cd D:\26spring_NUS_SOC_SWS\project\demo_0710
```

### Step 3：安装依赖

在 `sws3026-face` 环境内执行：

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

注意：必须安装 `opencv-contrib-python`，因为 `cv2.face.createFacemarkLBF()` 不在普通 `opencv-python` 中。这里使用 `pip` 安装包，但环境仍然由 Conda 管理。

### Step 4：检查 OpenCV 和 LBF 模块

```powershell
python -c "import cv2; print('cv2:', cv2.__version__); print('has cv2.face:', hasattr(cv2, 'face'))"
```

输出中必须看到：

```text
has cv2.face: True
```

如果是 `False`，执行：

```powershell
python -m pip uninstall opencv-python opencv-contrib-python
python -m pip install -r requirements.txt
```

### Step 5：检查模型文件是否存在

```powershell
Test-Path .\haarcascade_frontalface_default.xml
Test-Path .\face_detection_yunet_2023mar.onnx
Test-Path .\lbfmodel.yaml
```

三行都应该输出 `True`。YuNet 模型来自 [OpenCV Zoo](https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet)，无需额外训练。

### Step 6：运行 Beginner Demo

基础运行：

```powershell
python beginner_demo.py
```

更适合自拍视频的运行方式：

```powershell
python beginner_demo.py --mirror
```

如果光照不稳定，可尝试：

```powershell
python beginner_demo.py --mirror --preprocess clahe --min-neighbors 4
```

如果出现同一张脸上有两个重叠黄色框，默认版本已经加入重叠框抑制，同时仍保留多人识别。若仍偶尔出现，可继续提高 Haar 严格程度：

```powershell
python beginner_demo.py --mirror --min-neighbors 7 --overlap-threshold 0.45
```

如果摄像头不是默认编号：

```powershell
python beginner_demo.py --camera-index 1
```

运行后应看到一个 OpenCV 窗口，画面中会显示：

- 人脸 bounding box；
- 68 个 facial landmarks；
- FPS、检测耗时、检测到的人脸数量；
- 当前 detector 和 preprocessing 模式。

### Step 7：保存测试截图并退出

| 按键 | 功能 |
|---|---|
| `s` | 保存当前带检测结果的截图到 `snapshots/` |
| `q` | 退出 |
| `Esc` | 退出 |

建议至少保存一张正常正脸截图，方便放进 presentation 或报告。

### Step 8：做 Beginner 鲁棒性测试

依次测试并记录观察结果：

1. 正常室内光照、正脸看摄像头。
2. 弱光或背光。
3. 左右转头或上下低头/抬头。
4. 用手遮住嘴巴或半张脸。

每个场景观察两点：

- face box 是否稳定；
- 68 个 landmarks 是否漂移或贴错位置。

详细分析模板见 `robustness_notes.md`。

## 操作方式

| 按键 | 功能 |
|---|---|
| `q` | 退出 |
| `Esc` | 退出 |
| `s` | 保存当前带检测结果的截图到 `snapshots/` |

窗口左上角会显示 FPS、检测到的人脸数量、单帧检测耗时、当前 detector 和 preprocessing 模式。

## 可选：替代 Face Detector 对比

课程文档鼓励至少实验一个替代 detector。可选在同一个 Conda 环境中安装 MediaPipe：

```powershell
conda activate sws3026-face
cd D:\26spring_NUS_SOC_SWS\project\demo_0710
python -m pip install mediapipe
python beginner_demo.py --detector mediapipe --mirror
```

对比方式：

1. 先运行默认 Haar：

```powershell
python beginner_demo.py --detector haar --mirror
```

2. 再运行 MediaPipe：

```powershell
python beginner_demo.py --detector mediapipe --mirror
```

3. 分别观察正脸、偏头、弱光、遮挡嘴部四种情况，记录 face box 是否稳定、landmarks 是否漂移。

如果安装 MediaPipe 失败，不影响 Beginner 基础 demo；默认 Haar + LBF 已经满足核心实现要求。可以在报告中把 MediaPipe 作为计划中的替代 detector，并说明原因。

## Beginner 要求对照

| 要求 | 当前实现 |
|---|---|
| 使用 `cv2.VideoCapture(0)` 打开 webcam | `beginner_demo.py` 默认 `--camera-index 0` |
| 使用 Haar Cascade 检测每帧人脸 | 默认 `--detector haar`，加载 `haarcascade_frontalface_default.xml` |
| 对检测到的人脸画 bounding box | `draw_detections()` 中使用 `cv2.rectangle()` |
| 使用 OpenCV LBF | `LbfLandmarkEstimator` 调用 `cv2.face.createFacemarkLBF()` 和 `loadModel()` |
| 绘制 68 个 landmarks | 对每个 landmark 使用 `cv2.circle()` 绘制 |
| 实时视频窗口显示结果 | `cv2.imshow()` |
| 支持退出并释放资源 | `q`/`Esc` 退出，`finally` 中 `release()` 和 `destroyAllWindows()` |
| 鲁棒性分析 | 见 `robustness_notes.md` |
| 替代 detector 对比 | 支持可选 `--detector mediapipe` |

## 鲁棒性分析要点

- 光照变化：弱光、背光会让 Haar 检测框闪烁或消失，因为传统 cascade 依赖局部灰度/对比模式。当前默认使用 `CLAHE` 预处理改善局部对比度。
- 头部角度：大角度 yaw/pitch 会明显影响 Haar 和 LBF，因为二者对近正脸更友好。可以在汇报中用 MediaPipe 或其他现代 detector 作为改进方向。
- 部分遮挡：手遮嘴时 face box 可能仍存在，但嘴部 landmarks 容易漂移，因为 LBF 会尝试拟合完整脸型。
- 性能：模型在循环外加载；每帧仅执行一次检测和一次 LBF fit；默认最多处理 4 张脸，兼顾多人展示和实时速度。
- 重叠框处理：Haar 偶尔会对同一张脸输出一个紧框和一个大框。代码会根据重叠比例过滤重复框，但不会限制为单人；多人展示仍可正常显示多个互不重叠的人脸。

## 常见问题

### `ModuleNotFoundError: No module named 'cv2'`

说明还没有安装 OpenCV：

```powershell
conda activate sws3026-face
cd D:\26spring_NUS_SOC_SWS\project\demo_0710
python -m pip install -r requirements.txt
```

### `cv2 has no attribute face`

说明安装的是普通 `opencv-python`，需要换成 contrib 版本：

```powershell
conda activate sws3026-face
python -m pip uninstall opencv-python opencv-contrib-python
python -m pip install -r requirements.txt
```

### 摄像头打不开

尝试：

```powershell
python beginner_demo.py --camera-index 1
```

同时检查 Windows 隐私设置中是否允许桌面应用访问摄像头。

## Expert Level 完整流程

Expert Level 需要先离线训练一个基于 facial landmarks 的表情分类器，然后再运行实时 webcam demo。

### Step 1：确认数据 zip 存在

在 `demo_0710` 目录下执行：

```powershell
Test-Path ..\expert\facial_expression_dataset.zip
```

应该输出：

```text
True
```

脚本会直接读取 zip，并自动跳过 `__MACOSX` 和 `._` 这类无效文件；不要依赖当前已解压目录，因为该目录可能不完整。

### Step 2：先跑小样本 smoke test

这一步用于确认 OpenCV、LBF、数据读取和训练流程都能跑通：

```powershell
conda activate sws3026-face
cd D:\26spring_NUS_SOC_SWS\project\demo_0710
python train_expert.py --max-train-per-class 30 --max-test-per-class 10
```

成功后会生成：

| 输出 | 作用 |
|---|---|
| `models/expression_classifier.joblib` | 实时 demo 加载的表情分类器 |
| `reports/dataset_inventory.json` | train/test 七类数据数量核对 |
| `reports/expert_metrics.json` | accuracy、macro F1、平均 prediction ms |
| `reports/confusion_matrix.csv` | confusion matrix |
| `reports/failure_cases.csv` | landmark 提取失败样本 |
| `features/*.npz` | 缓存后的 landmark 特征；同样采样规模重训会直接复用 |

### Step 3：训练更完整的模型

如果 smoke test 正常，直接运行全量训练/评估。当前脚本支持多进程 landmark 提取，推荐使用 8 个 worker：

```powershell
python train_expert.py --workers 8
```

如果电脑较卡，可以先用每类 800 张训练、300 张测试做较快版本：

```powershell
python train_expert.py --max-train-per-class 800 --max-test-per-class 300 --workers 8
```

说明：

- 分类器输入是 68 个 facial landmarks 的归一化坐标和少量几何特征，不使用完整人脸图像作为分类输入。
- 每张 FER 图片都会先运行 Haar + LBF facial keypoint pipeline；对 FER 这种裁剪脸图，如果 Haar 没检测到，会使用中心人脸框 fallback，并在统计中记录 `center_fallback`。
- 分类器使用轻量 RBF SVM，实时阶段只对 landmark 特征做一次预测，通常远低于 30 ms。
- 脚本会把 landmark 特征缓存到 `features/`。全量缓存文件为 `train_m0_seed42_fallback_v1.npz` 和 `test_m0_seed42_fallback_v1.npz`。如果修改了特征提取代码或想强制重提取，添加 `--rebuild-cache`。
- `--workers` 只影响 landmark 特征提取；已经有缓存时会直接加载缓存，不会重复跑 LBF。

### Step 4：查看评估结果

训练完成后，重点看：

```powershell
Get-Content .\reports\expert_metrics.json
Get-Content .\reports\confusion_matrix.csv
Get-Content .\reports\failure_cases.csv -TotalCount 20
```

汇报中至少说明：

- accuracy 或 macro F1；
- confusion matrix 中最容易混淆的类别；
- landmark 失败样本的原因，例如低分辨率、遮挡、侧脸、表情夸张导致 face/landmark 不稳定。

当前展示模型的汇总说明见：

```powershell
Get-Content .\expert_report.md
```

### Step 5：运行实时 Expert Demo

训练生成 `models/expression_classifier.joblib` 后运行：

```powershell
python expert_demo.py --mirror
```

Expert 实时模式默认使用 YuNet，而训练特征提取流程仍保留原有 Haar + LBF。YuNet 使用学习到的人脸置信度和 NMS，在进入 LBF 前过滤背景大框；默认仍支持多人，最多处理 4 张脸：

```powershell
python expert_demo.py --mirror --max-faces 4
```

如果背景仍偶尔被识别为人脸，可只提高 YuNet 置信度，不限制人数：

```powershell
python expert_demo.py --mirror --max-faces 4 --yunet-score-threshold 0.85
```

如果远处或侧面真人漏检，可适当放宽到：

```powershell
python expert_demo.py --mirror --max-faces 4 --yunet-score-threshold 0.65 --min-face-size 45
```

实时模式已经加入以下稳定策略：

- 多目标 face track：每张脸单独稳定，不把默认场景限制成单人。
- YuNet 置信度 + NMS：默认 `--yunet-score-threshold 0.75`，直接验证候选区域是否具有真实人脸外观。
- 缩放检测：默认 `--yunet-input-size 640`，检测后把坐标映射回原画面，兼顾精度和 CPU 实时性能。
- 连续帧确认：默认 `--stable-frames 2`，新检测框需要连续出现才显示，减少一闪而过的“空气脸”。
- 短暂 hold：真实人脸短暂漏检时保留上一帧稳定框，减少闪动。
- landmark 合理性过滤：YuNet 路径只做宽松边界检查，避免严格几何规则误删侧脸；Haar 对照路径保留更完整的眼鼻嘴检查。
- landmark EMA：每个 track 先以 `--landmark-smoothing 0.85` 平滑 68 个关键点，再重新计算分类特征，减少 LBF 逐帧坐标抖动。
- 概率 EMA：默认 `--prob-smoothing 0.85`，每张脸独立平滑七类概率，多人之间互不污染。
- 标签迟滞：第一个标签需要连续 6 帧确认；切换标签需要新类别连续领先 8 帧，并至少保持旧标签 30 帧。
- 低置信度门控：默认要求置信度至少 `0.48`、领先第二名至少 `0.12`，否则保持当前标签或显示 `uncertain`。

如果展示环境中仍然闪动，可使用更稳但响应稍慢的参数：

```powershell
python expert_demo.py --mirror --landmark-smoothing 0.90 --prob-smoothing 0.90 --switch-frames 10 --min-label-hold-frames 45
```

如果感觉真实表情切换响应太慢，可适度放宽为：

```powershell
python expert_demo.py --mirror --landmark-smoothing 0.75 --prob-smoothing 0.75 --switch-frames 4 --min-label-hold-frames 15
```

如果要和原有 Haar 方式对照，可运行：

```powershell
python expert_demo.py --mirror --detector haar --max-faces 4 --min-detection-weight 1.0
```

运行后画面会显示：

- face bounding box；
- 68 个 facial landmarks；
- 表情类别和置信度；
- 单次预测耗时；
- 表情驱动 visual effect。

当前 visual effects：

| 表情 | 效果 |
|---|---|
| `happy` | sparkle 闪光 |
| `surprise` | burst 放射线 |
| `angry` | 红色 tint |
| `sad` | 蓝色 tint |
| `neutral` | 中性边框 |

按键：

| 按键 | 功能 |
|---|---|
| `q` | 退出 |
| `Esc` | 退出 |
| `s` | 保存 Expert demo 截图到 `snapshots/` |

如果只想看分类文字和 landmarks，不显示特效：

```powershell
python expert_demo.py --mirror --no-effects
```

如果只想快速检查摄像头和性能，不打开显示窗口：

```powershell
python expert_demo.py --mirror --benchmark-frames 30
```

命令会默认先跑 3 帧 warmup，不计入平均值。输出中的 `raw_faces` 是当前 detector + LBF 的原始候选数，`faces` 是经过多人稳定器和误检过滤后的最终显示数；`avg_pipeline_ms` 是每帧 face detection + LBF landmarks + expression prediction 的平均耗时；`avg_prediction_ms_per_face` 是分类器本身的平均耗时；`label_switches` 是测量期间已确认标签发生变化的总次数，静止表情测试应尽量接近 `0`。

## 后续迭代接口

后续做表情分类时，可以复用 `face_pipeline.py` 中的 `LbfLandmarkEstimator`，把 `landmarks` 转成特征：

- 按 face box 归一化 x/y 坐标；
- 减去 landmarks 中心点并按脸宽/脸高缩放；
- 构造嘴角、眉眼、眼睛开合等几何距离；
- 将实时预测结果接到 `beginner_demo.py` 的绘制层，增加 expression label 和 video effects。
