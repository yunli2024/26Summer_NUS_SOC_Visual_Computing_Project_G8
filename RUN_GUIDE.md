# Visual Computing Project 运行指南

本指南适用于 Windows PowerShell，覆盖以下程序：

- Beginner Level：实时人脸框与 68 点 facial landmarks
- Expert Level：keypoint-only 表情分类训练、评估与实时特效
- Bonus Level：Just Dance 双面板姿态检测、时序对齐与评分
- Bonus Action Game：摄像头姿态控制的 3D Runner
- Bonus Mario：摄像头姿态控制的平台游戏
- 辅助检查、单元测试和离线姿态分析

所有命令都应在项目根目录运行。不要在 Google Colab 或 Jupyter Notebook
中运行 webcam/GUI 程序。

## 1. 进入项目根目录

本机当前路径：

```powershell
cd D:\AAA_SHERRY\NUS_school_computing_summer_workshop\VisualComputingProjects\project
```

如果从 GitHub 重新克隆，请先进入包含 `VisualComputingProject`、
`environment_setup` 和 `run_*.ps1` 的目录。

## 2. 第一次运行：安装环境

### 方案 A：Conda（推荐）

```powershell
conda env create -f environment_setup\environment.yml
conda activate vc_sws3026
python -m pip install --upgrade pip
python -m pip install -r environment_setup\requirements.txt
```

以后每次打开新的 PowerShell，只需执行：

```powershell
conda activate vc_sws3026
```

### 方案 B：普通 Python 3.11

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r environment_setup\requirements.txt
```

项目需要 `opencv-contrib-python`，因为 LBF facial landmarks 使用
`cv2.face`。不要同时保留普通版 `opencv-python` 和
`opencv-contrib-python`。

检查环境：

```powershell
python --version
python -c "import cv2; print(cv2.__version__); print('cv2.face:', hasattr(cv2, 'face'))"
```

第二行应显示 `cv2.face: True`。

### PowerShell 禁止运行 `.ps1` 时

只对当前 PowerShell 窗口临时放行：

```powershell
Set-ExecutionPolicy -Scope Process Bypass
```

然后再运行本指南中的 `.\run_*.ps1` 命令。

## 3. 运行前必须确认的资源

以下两个大文件不会上传到 GitHub，但 Beginner/Expert 会在本地使用：

| 文件 | 准确路径 | 用途 |
| --- | --- | --- |
| `lbfmodel.yaml` | `VisualComputingProject\resources\face_models\lbfmodel.yaml` | Beginner、Expert 的 68 点 LBF |
| `facial_expression_dataset.zip` | `VisualComputingProject\resources\expression_data\facial_expression_dataset.zip` | Expert 重新训练和完整数据检查 |

从 GitHub 新克隆项目后，需要从课程资料或现有本地项目中恢复到上述路径。
详细说明见 `VisualComputingProject\resources\README.md`。

检查文件是否存在：

```powershell
Test-Path VisualComputingProject\resources\face_models\lbfmodel.yaml
Test-Path VisualComputingProject\resources\expression_data\facial_expression_dataset.zip
Test-Path VisualComputingProject\resources\pose_models\yolov8n-pose.pt
Test-Path VisualComputingProject\expert_level\models\keypoint\current\expression_classifier.joblib
```

四条命令应尽量全部返回 `True`。其中 FER ZIP 只影响 Expert 训练，不影响
Bonus 和 Mario。

## 4. Beginner Level：实时 facial landmarks

### 4.1 不打开摄像头的环境检查

```powershell
python VisualComputingProject\beginner_level\tests\check_part2_setup.py
```

它会检查 OpenCV、Haar cascade、LBF 模型和相关模块。

### 4.2 运行主程序

```powershell
.\run_beginner_level.ps1
```

等价的 Python 命令：

```powershell
python VisualComputingProject\beginner_level\main.py
```

程序会打开摄像头，显示 face bounding box、68 个 landmarks、检测状态和
FPS。

### 4.3 常用参数

使用 CLAHE 改善不均匀光照：

```powershell
.\run_beginner_level.ps1 --preprocess clahe
```

使用 CLAHE 与 gamma 组合：

```powershell
.\run_beginner_level.ps1 --preprocess clahe-gamma
```

第二个摄像头：

```powershell
.\run_beginner_level.ps1 --camera 1
```

只跟踪一个最稳定的人脸：

```powershell
.\run_beginner_level.ps1 --single-face
```

### 4.4 运行时按键

- `1`：raw preprocessing
- `2`：CLAHE
- `3`：gamma
- `4`：CLAHE + gamma
- `v`：切换显示画面增强
- `m`：切换镜像方向
- `r`：重置检测器状态
- `q`：退出并释放摄像头

## 5. Expert Level：表情分类与实时特效

Expert 分类器只使用 facial keypoints 及其几何特征，不使用完整人脸图像。

### 5.1 检查数据、模型和历史指标

```powershell
.\run_expert_level.ps1 inspect
```

等价命令：

```powershell
python VisualComputingProject\expert_level\main.py inspect
```

如果 FER ZIP 没有恢复，`inspect` 会报告 dataset `MISSING`；实时 demo 仍可在
LBF 模型和最终分类器存在时运行。

### 5.2 运行实时表情特效

推荐命令：

```powershell
.\run_expert_level.ps1 demo --mirror
```

它默认使用 YuNet 检测人脸、LBF 提取 68 点、Geometry-SVM 分类表情，并根据
表情显示动态特效。

使用 Haar detector 做对照：

```powershell
.\run_expert_level.ps1 demo --mirror --detector haar
```

第二个摄像头：

```powershell
.\run_expert_level.ps1 demo --mirror --camera-index 1
```

关闭视觉特效，只看 landmarks 和分类结果：

```powershell
.\run_expert_level.ps1 demo --mirror --no-effects
```

CPU 较慢时降低摄像头分辨率：

```powershell
.\run_expert_level.ps1 demo --mirror --width 640 --height 360
```

### 5.3 Demo 运行时按键

- `s`：保存当前截图到 `VisualComputingProject\expert_level\snapshots\`
- `q` 或 `Esc`：退出

### 5.4 测量实时延迟

下面命令处理 300 帧但不打开显示窗口，并输出 FPS、pipeline latency 和
单次分类 latency：

```powershell
.\run_expert_level.ps1 demo --benchmark-frames 300 --warmup-frames 10
```

目标是单次 expression prediction 接近或低于 30 ms。

### 5.5 快速验证训练流程

快速训练会使用少量样本。下面命令把实验模型写入单独目录，不覆盖当前正式
demo 模型：

```powershell
.\run_expert_level.ps1 train `
  --max-train-per-class 100 `
  --max-test-per-class 50 `
  --cv-folds 3 `
  --workers 1 `
  --model-out VisualComputingProject\expert_level\models\keypoint\experiments\quick_classifier.joblib `
  --report-dir VisualComputingProject\expert_level\results\quick_check `
  --cache-dir VisualComputingProject\expert_level\data\cache_quick
```

### 5.6 完整正式训练

完整训练会读取 FER ZIP 中已有的 `train`/`test`，在 train split 内进行
Stratified K-fold 与 PCA 候选模型比较，最后只在 test split 评估一次。

为避免覆盖当前正式模型，先输出到 candidate 路径：

```powershell
.\run_expert_level.ps1 train `
  --cv-folds 5 `
  --workers 1 `
  --model-out VisualComputingProject\expert_level\models\keypoint\experiments\candidate_classifier.joblib `
  --report-dir VisualComputingProject\expert_level\results\candidate_cv `
  --cache-dir VisualComputingProject\expert_level\data\cache
```

Windows 首次训练建议先使用 `--workers 1`。确认稳定后，可改为
`--workers 2` 或 `--workers 4`。完整 landmark extraction 在 CPU 上可能需要
较长时间。

训练报告包括：

- `cross_validation.csv/json`
- `expert_metrics.json`
- `confusion_matrix.csv`
- `misclassification_cases.csv`
- `extraction_failures.csv`

## 6. Bonus Level：Just Dance 双面板评分

### 6.1 不打开 GUI 的资源和评分配置检查

```powershell
.\run_bonus_level.ps1 --check
```

它会检查 YOLOv8 Pose 模型、默认舞蹈视频，并显示空间归一化、时间对齐窗口和
评分权重。

### 6.2 打开主程序

```powershell
.\run_bonus_level.ps1
```

等价命令：

```powershell
python VisualComputingProject\bonus_level\main.py
```

### 6.3 GUI 操作顺序

1. 左侧默认使用 `dance_example_1.mp4`；也可以点击 `Open` 选择其他视频。
2. 点击左侧 `Start` 播放 reference video。
3. 随后点击右侧 `Start Webcam`。
4. 让全身或主要身体关节进入画面，并跟随左侧舞者。
5. 屏幕会显示 `Perfect`、`Super`、`Good`、`Miss`、combo、总分和 delay。
6. 可用 `Pause`、`Resume`、`Restart` 控制参考视频。
7. 点击 `Show Summary` 查看平均分、最佳分和 delay 统计。
8. 关闭窗口时程序会停止线程并释放视频与摄像头。

推荐先单独启动左侧检查 pose skeleton，再启动 webcam 进行双面板评分。

## 7. Bonus Action Game：3D Runner

### 7.1 启动游戏

```powershell
python VisualComputingProject\bonus_level_action_game\main.py
```

程序使用 Ursina 打开第三人称 3D 三赛道跑酷，并使用 YOLOv8 Pose 读取
webcam 姿态。

### 7.2 姿态操作

- 身体向左移动：切到左侧赛道
- 身体向右移动：切到右侧赛道
- 双手快速举过肩膀：跳跃
- 双臂在胸前交叉：下滑

### 7.3 键盘备用操作

- `A` / 左方向键：向左
- `D` / 右方向键：向右
- `W` / 上方向键：跳跃
- `S` / 下方向键：下滑
- `C`：重新校准
- `R`：Game Over 后重新开始
- `Esc`：退出

摄像头编号和动作阈值位于：

```text
VisualComputingProject\bonus_level_action_game\src\config.py
```

默认值为 `RUNNER_CAMERA_INDEX = 0`。如果摄像头不可用，游戏仍可使用键盘。

## 8. Bonus Mario：姿态控制平台游戏

### 8.1 先执行无 GUI 检查

```powershell
.\run_mario.ps1 --check
```

该命令会读取共享参考视频的一帧，检查 YOLO 模型、主舞者选择、手势控制链路
和游戏素材。

### 8.2 启动游戏

```powershell
.\run_mario.ps1
```

等价命令：

```powershell
python VisualComputingProject\bonus_level_mario\mario_camera_demo.py
```

GUI 打开后点击 `Start Camera`。

使用第二个摄像头：

```powershell
.\run_mario.ps1 --camera 1
```

CPU 推理较慢时：

```powershell
.\run_mario.ps1 --image-size 256
```

### 8.3 操作

- 单手向左或向右：移动
- 双手举起：跳跃
- 双手靠拢：下蹲
- `A` / `D` 或左右方向键：键盘移动
- `Space` / `W` / 上方向键：键盘跳跃
- `S` / 下方向键：键盘下蹲
- `R`：重新开始

## 9. 可选：离线视频姿态分析

使用默认舞蹈视频：

```powershell
python VisualComputingProject\bonus_level_mario\pose_analyzer.py --max-frames 300
```

分析指定视频并显示过程：

```powershell
python VisualComputingProject\bonus_level_mario\pose_analyzer.py "D:\path\to\video.mp4" `
  --max-frames 300 `
  --show
```

输出默认保存在：

```text
VisualComputingProject\bonus_level_mario\task1_results\
```

显示窗口中按 `q` 或 `Esc` 可以提前结束。

## 10. 一次性检查合并项目

不打开正式 webcam GUI 的综合检查：

```powershell
.\check_merge.ps1
```

它会依次检查 Beginner setup、Expert 特征与稳定性测试、Bonus 时间对齐测试、
Mario 测试、Expert 资源、Bonus 资源和 Mario validation。

注意：该命令要求 LBF、FER ZIP、最终分类器、YOLO 模型和参考视频全部存在。

## 11. 分模块运行测试

Beginner：

```powershell
python VisualComputingProject\beginner_level\tests\check_part2_setup.py
```

Expert：

```powershell
python VisualComputingProject\expert_level\tests\test_keypoint_features.py
python VisualComputingProject\expert_level\tests\test_realtime_stability.py
```

Bonus：

```powershell
python -m unittest discover -s VisualComputingProject\bonus_level\tests -p "test_*.py" -v
```

Mario：

```powershell
python -m unittest discover -s VisualComputingProject\bonus_level_mario -p "test_*.py" -v
```

## 12. 常见问题

### `cv2 has no attribute face`

当前环境安装了错误的 OpenCV 版本。执行：

```powershell
python -m pip uninstall -y opencv-python opencv-contrib-python
python -m pip install opencv-contrib-python
```

### `lbfmodel.yaml` 或 FER ZIP 找不到

它们因体积较大没有上传 GitHub。按照第 3 节恢复到准确路径，不要修改文件名。

### 摄像头打不开

先关闭 Teams、Zoom、浏览器摄像头页面和其他 OpenCV 程序，然后尝试：

```powershell
.\run_beginner_level.ps1 --camera 1
.\run_expert_level.ps1 demo --mirror --camera-index 1
.\run_mario.ps1 --camera 1
```

### PowerShell 报脚本被禁止

```powershell
Set-ExecutionPolicy -Scope Process Bypass
```

### YOLO/GUI 很慢

- 关闭其他占用 CPU/GPU 的程序。
- Mario 使用 `--image-size 256`。
- Expert 将摄像头分辨率降到 `640x360`。
- 保证画面中只有一位主要玩家，并保持充足光照。

### 退出后摄像头仍被占用

优先使用程序提供的 `q`、`Esc`、Stop 按钮或关闭 GUI 窗口。不要直接终止整个
IDE；正常退出会调用 `release()` 并关闭窗口。

## 13. Demo Day 推荐启动顺序

```powershell
conda activate vc_sws3026
.\run_beginner_level.ps1 --preprocess clahe
.\run_expert_level.ps1 demo --mirror
.\run_bonus_level.ps1
.\run_mario.ps1
python VisualComputingProject\bonus_level_action_game\main.py
```

一次只运行一个需要 webcam 的程序，退出并确认摄像头释放后，再启动下一个。
