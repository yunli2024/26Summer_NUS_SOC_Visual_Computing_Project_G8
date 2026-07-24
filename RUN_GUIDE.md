# Visual Computing Project 运行指南

本指南适用于当前合并版本：

- Beginner：本地 Haar + LBF 68 点实时检测；
- Expert：Zhangyx `part two`；
- Bonus：Zhangyx `part three`；
- Bonus 扩展：已整合的 Mario 姿态控制游戏、3D Runner。

所有 webcam/GUI 程序必须在本地 PowerShell 中运行，不要使用 Google Colab
或 Jupyter Notebook。

## 1. 进入项目根目录

```powershell
cd D:\AAA_SHERRY\NUS_school_computing_summer_workshop\VisualComputingProjects\project
```

后续命令默认都从这个目录执行。

## 2. 创建和激活环境

### Conda（推荐）

第一次运行：

```powershell
conda env create -f environment_setup\environment.yml
conda activate vc_sws3026
python -m pip install --upgrade pip
python -m pip install -r environment_setup\requirements.txt
```

以后打开新的 PowerShell：

```powershell
conda activate vc_sws3026
```

### 普通 Python 3.11

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r environment_setup\requirements.txt
```

### 安装 Zhangyx Part Two/Three 的精确依赖

Part Two：

```powershell
python -m pip install -r VisualComputingProject\expert_level\requirements.txt
```

Part Three 建议保留 `opencv-contrib-python`，再安装 Ultralytics：

```powershell
python -m pip install -r VisualComputingProject\bonus_level\requirements.txt
python -m pip install ultralytics==8.4.98 --no-deps
```

项目需要 `cv2.face`，所以必须使用 `opencv-contrib-python`。检查：

```powershell
python -c "import cv2; print(cv2.__version__); print('cv2.face:', hasattr(cv2, 'face'))"
```

最后应显示：

```text
cv2.face: True
```

## 3. PowerShell 脚本执行权限

如果 `.\run_*.ps1` 被系统阻止，只对当前窗口临时放行：

```powershell
Set-ExecutionPolicy -Scope Process Bypass
```

## 4. 必需资源

| Resource | Local path | GitHub |
| --- | --- | --- |
| Haar cascade | `VisualComputingProject\resources\face_models\haarcascade_frontalface_default.xml` | included |
| YuNet | `VisualComputingProject\resources\face_models\face_detection_yunet_2023mar.onnx` | included |
| LBF model | `VisualComputingProject\resources\face_models\lbfmodel.yaml` | excluded because large |
| FER dataset ZIP | `VisualComputingProject\resources\expression_data\facial_expression_dataset.zip` | excluded because large |
| Extracted FER | `VisualComputingProject\resources\expression_data\facial_expression_dataset\train` and `test` | local only |
| YOLOv8 Pose | `VisualComputingProject\resources\pose_models\yolov8n-pose.pt` | included |
| Dance video | `VisualComputingProject\resources\videos\dance_example_1.mp4` | included |

检查：

```powershell
Test-Path VisualComputingProject\resources\face_models\lbfmodel.yaml
Test-Path VisualComputingProject\resources\expression_data\facial_expression_dataset\train
Test-Path VisualComputingProject\resources\expression_data\facial_expression_dataset\test
Test-Path VisualComputingProject\resources\pose_models\yolov8n-pose.pt
Test-Path VisualComputingProject\resources\videos\dance_example_1.mp4
```

从 GitHub 新克隆后，需要从课程资料恢复 LBF 与 FER 数据。

## 5. Beginner Level

### 5.1 无摄像头检查

```powershell
python VisualComputingProject\beginner_level\tests\check_part2_setup.py
```

### 5.2 运行实时 landmarks

```powershell
.\run_beginner_level.ps1
```

等价命令：

```powershell
python VisualComputingProject\beginner_level\main.py
```

### 5.3 常用参数

CLAHE：

```powershell
.\run_beginner_level.ps1 --preprocess clahe
```

CLAHE + gamma：

```powershell
.\run_beginner_level.ps1 --preprocess clahe-gamma
```

第二个摄像头：

```powershell
.\run_beginner_level.ps1 --camera 1
```

单人稳定跟踪：

```powershell
.\run_beginner_level.ps1 --single-face
```

### 5.4 运行时按键

- `1`：raw
- `2`：CLAHE
- `3`：gamma
- `4`：CLAHE + gamma
- `v`：切换画面增强
- `m`：切换镜像
- `r`：重置 tracker
- `q`：退出

## 6. Expert Level - Zhangyx Part Two

Expert 主目录：

```text
VisualComputingProject\expert_level\
```

主要程序：

- `task1_pipeline.py`：特征提取、训练和评估；
- `tune_svm.py`：SVM validation search；
- `ablate_geometry_groups.py`：几何特征消融；
- `task2_realtime.py`：webcam 表情分类和特效；
- `test_expression_features.py`：特征单元测试。

### 6.1 检查命令和特征测试

```powershell
python VisualComputingProject\expert_level\task1_pipeline.py --help
python VisualComputingProject\expert_level\task2_realtime.py --help
python VisualComputingProject\expert_level\test_expression_features.py
```

### 6.2 快速训练 smoke test

使用每类 50 train、20 test，输出到临时实验目录，不覆盖正式模型：

```powershell
python VisualComputingProject\expert_level\task1_pipeline.py `
  --max-train-per-class 50 `
  --max-test-per-class 20 `
  --output VisualComputingProject\expert_level\artifacts_smoke
```

### 6.3 完整 HGB pipeline

```powershell
python VisualComputingProject\expert_level\task1_pipeline.py `
  --output VisualComputingProject\expert_level\artifacts_retrain
```

程序会：

1. 读取共享 FER `train`/`test`；
2. 将 48x48 图像放大到 192x192；
3. 用 centered FER crop 拟合 LBF；
4. 眼中心对齐为 136 coordinates；
5. 训练 class-balanced HGB；
6. 输出 metrics、confusion matrix 和 failure cases。

### 6.4 只使用缓存重新训练

首次 extraction 后：

```powershell
python VisualComputingProject\expert_level\task1_pipeline.py `
  --stage train `
  --features VisualComputingProject\expert_level\artifacts_retrain\fer_landmark_features.npz `
  --output VisualComputingProject\expert_level\artifacts_retrain
```

`fer_landmark_features.npz` 是本地生成缓存，不会上传 GitHub。

### 6.5 PCA-SVM 对照实验

```powershell
python VisualComputingProject\expert_level\task1_pipeline.py `
  --stage train `
  --classifier svm `
  --pca-variance 0.95 `
  --features VisualComputingProject\expert_level\artifacts_retrain\fer_landmark_features.npz `
  --output VisualComputingProject\expert_level\artifacts_pca_retrain
```

### 6.6 调参 coordinate SVM

```powershell
python VisualComputingProject\expert_level\tune_svm.py `
  --features VisualComputingProject\expert_level\artifacts_retrain\fer_landmark_features.npz `
  --output VisualComputingProject\expert_level\artifacts_svm_tuned_retrain
```

### 6.7 调参 geometry SVM

```powershell
python VisualComputingProject\expert_level\tune_svm.py `
  --geometry `
  --features VisualComputingProject\expert_level\artifacts_retrain\fer_landmark_features.npz `
  --output VisualComputingProject\expert_level\artifacts_svm_geometry_retrain
```

正式集成模型已经存在：

```text
VisualComputingProject\expert_level\artifacts_svm_geometry\expression_classifier.joblib
```

不要在没有比较 Macro-F1、failure cases 和 latency 前覆盖它。

### 6.8 生成无摄像头特效预览

```powershell
python VisualComputingProject\expert_level\task2_realtime.py `
  --preview tmp\expert_effects_preview.png
```

### 6.9 运行 webcam 表情特效

```powershell
.\run_expert_level.ps1
```

等价命令：

```powershell
python VisualComputingProject\expert_level\task2_realtime.py
```

第二个摄像头：

```powershell
.\run_expert_level.ps1 --camera 1
```

降低分辨率：

```powershell
.\run_expert_level.ps1 --width 640 --height 480
```

关闭特效：

```powershell
.\run_expert_level.ps1 --effects-off
```

### 6.10 Expert 运行时按键

- `E`：切换 expression effects
- `L`：切换 landmarks
- `S`：切换 temporal smoothing
- `C`：切换 CLAHE
- `Q` 或 `Esc`：退出

## 7. Bonus Task 1 - Zhangyx Part Three Pose Analysis

主程序：

```text
VisualComputingProject\bonus_level\pose_analyzer.py
```

### 7.1 查看参数

```powershell
python VisualComputingProject\bonus_level\pose_analyzer.py --help
```

### 7.2 分析 60 帧示例，不写 MP4

```powershell
python VisualComputingProject\bonus_level\pose_analyzer.py `
  --start-frame 300 `
  --max-frames 60 `
  --stride 3 `
  --image-size 416 `
  --no-video `
  --output tmp\bonus_task1_check
```

### 7.3 显示分析过程

```powershell
python VisualComputingProject\bonus_level\pose_analyzer.py `
  --start-frame 300 `
  --max-frames 60 `
  --stride 3 `
  --show `
  --output tmp\bonus_task1_show
```

显示窗口按 `Q` 或 `Esc` 退出。

### 7.4 分析指定 TikTok 视频

```powershell
python VisualComputingProject\bonus_level\pose_analyzer.py `
  "D:\path\to\TikTok\YouTube.mp4" `
  --max-frames 60 `
  --stride 3 `
  --output tmp\tiktok_pose_check
```

不要求运行完整 TikTok dataset。Presentation 应挑选单人和多人案例比较。

## 8. Bonus Task 2 - 准备 reference cache

Just Dance 运行时需要：

- annotated reference MP4；
- `pose_cache.npz`。

现在 `danceapp.py` 会在首次启动时自动检查并生成这两个文件。推荐在正式
Demo 前单独执行一次预处理：

```powershell
python VisualComputingProject\bonus_level\danceapp.py --prepare-only
```

默认会处理完整参考视频，每两帧取一帧，以降低 CPU 预处理时间。第一次运行
可能需要几分钟；完成后再次启动不会重复生成。

预期生成：

```text
VisualComputingProject\bonus_level\task2_results\dance_example_1\annotated.mp4
VisualComputingProject\bonus_level\task2_results\dance_example_1\pose_cache.npz
```

这两个运行产物不会上传 GitHub，新 clone 需要重新生成。

需要强制重新生成：

```powershell
python VisualComputingProject\bonus_level\danceapp.py `
  --prepare-only `
  --rebuild-reference
```

希望使用完整 30 FPS reference：

```powershell
python VisualComputingProject\bonus_level\danceapp.py `
  --prepare-only `
  --rebuild-reference `
  --prepare-stride 1
```

也可以直接调用底层 `pose_analyzer.py`。此时输出目录必须指定为
`bonus_level\task2_results`：

```powershell
python VisualComputingProject\bonus_level\pose_analyzer.py `
  VisualComputingProject\resources\videos\dance_example_1.mp4 `
  --stride 2 `
  --image-size 416 `
  --output VisualComputingProject\bonus_level\task2_results
```

CPU 较慢时先用 30 帧验证：

```powershell
python VisualComputingProject\bonus_level\pose_analyzer.py `
  VisualComputingProject\resources\videos\dance_example_1.mp4 `
  --start-frame 300 `
  --max-frames 30 `
  --stride 3 `
  --image-size 320 `
  --output tmp\bonus_task2_smoke
```

## 9. Bonus Task 2 - Just Dance GUI

### 9.1 评分单元测试

```powershell
python VisualComputingProject\bonus_level\test_dance_scoring.py
```

### 9.2 无 webcam 输入检查

该命令会自动准备缺失的默认 cache，然后执行无摄像头检查：

```powershell
python VisualComputingProject\bonus_level\just_dance_app.py --check
```

### 9.3 启动游戏

```powershell
.\run_bonus_level.ps1
```

等价命令：

```powershell
python VisualComputingProject\bonus_level\danceapp.py
```

CPU 较慢：

```powershell
.\run_bonus_level.ps1 --image-size 320
```

第二个摄像头：

```powershell
.\run_bonus_level.ps1 --camera 1
```

不接受镜像动作：

```powershell
.\run_bonus_level.ps1 --no-mirror
```

调整 delay 与 motion window：

```powershell
.\run_bonus_level.ps1 --max-lag 0.5 --motion-window 0.3
```

### 9.4 GUI 操作

- `Start / Restart`：启动 webcam 并重置分数
- `Pause / Resume`：冻结/恢复游戏时钟和 inference
- `Stop`：停止本轮但保留 GUI
- `Accept mirrored moves`：允许左右镜像动作

屏幕会显示：

- similarity；
- `PERFECT / GREAT / GOOD / MISS`；
- `SYNC / HOLD / MOVE!`；
- points、average、combo；
- selected lag、mirror、motion activity。

## 10. Scoring A/B Tester

先生成正式 Task 2 cache，然后运行：

```powershell
python VisualComputingProject\bonus_level\scoring_video_tester.py
```

它用同一个 cached video 模拟 reference/player，可调：

- player delay；
- allowed reaction lag；
- mirror player；
- accept mirrored moves。

Headless deterministic check：

```powershell
python VisualComputingProject\bonus_level\scoring_video_tester.py --check
```

## 11. 已整合的 Mario Demo

Zhangyx Part Three 的 Mario 已整合到独立的 `bonus_level_mario` 目录。
推荐从这个统一目录运行：

```powershell
.\run_mario.ps1 --check
.\run_mario.ps1
```

GUI 打开后点击 `Start Camera`。

第二个摄像头：

```powershell
.\run_mario.ps1 --camera 1
```

CPU 较慢：

```powershell
.\run_mario.ps1 --image-size 256
```

测试：

```powershell
python -m unittest discover -s VisualComputingProject\bonus_level_mario -p "test_*.py" -v
```

原始的 `bonus_level\mario_demo` 目录作为 Part Three 来源副本保留；后续运行和
修改以 `bonus_level_mario` 为准。

## 12. 3D Runner 扩展

```powershell
python VisualComputingProject\bonus_level_action_game\main.py
```

键盘备用：

- `A/D` 或左右方向键：换赛道
- `W` 或上方向键：跳跃
- `S` 或下方向键：下滑
- `C`：重新校准
- `R`：重开
- `Esc`：退出

## 13. 综合检查

```powershell
.\check_merge.ps1
```

它会检查：

- Beginner setup；
- Part Two expression features；
- Part Three scoring tests；
- Part Two CLI 和 effect preview；
- Part Three CLI；
- 已整合 Mario validation；
- 如果正式 Bonus cache 已生成，再执行 Just Dance input check。

不会自动打开 webcam。

## 14. 常见错误

### `cv2 has no attribute face`

```powershell
python -m pip uninstall -y opencv-python opencv-contrib-python
python -m pip install opencv-contrib-python==4.12.0.88
```

### `lbfmodel.yaml` 找不到

恢复到：

```text
VisualComputingProject\resources\face_models\lbfmodel.yaml
```

### FER class directory 找不到

需要：

```text
VisualComputingProject\resources\expression_data\facial_expression_dataset\train\<class>
VisualComputingProject\resources\expression_data\facial_expression_dataset\test\<class>
```

七类：

```text
angry disgust fear happy neutral sad surprise
```

### Bonus 报 `pose_cache.npz` 找不到

当前版本会自动生成默认 cache。如果自动生成失败，先确认公共视频与模型存在，
再运行：

```powershell
python VisualComputingProject\bonus_level\danceapp.py --prepare-only
```

### 摄像头打不开

关闭 Teams、Zoom、浏览器摄像头页面和其他 OpenCV 程序，然后尝试 camera 1：

```powershell
.\run_beginner_level.ps1 --camera 1
.\run_expert_level.ps1 --camera 1
.\run_bonus_level.ps1 --camera 1
```

### YOLO 很慢

- 使用 `--image-size 320` 或 `256`；
- 关闭其他 CPU/GPU 程序；
- reference side 使用预生成 cache；
- 不要同时运行多个 webcam demo。

### PowerShell 拒绝 `.ps1`

```powershell
Set-ExecutionPolicy -Scope Process Bypass
```

## 15. Demo Day 启动顺序

```powershell
conda activate vc_sws3026
.\run_beginner_level.ps1 --preprocess clahe
.\run_expert_level.ps1
.\run_bonus_level.ps1 --image-size 320
```

可选扩展：

```powershell
.\run_mario.ps1
python VisualComputingProject\bonus_level_action_game\main.py
```

一次只运行一个 webcam 程序。使用 `Q`、`Esc`、Stop 或关闭 GUI 正常退出，
确认 camera 已释放后再启动下一个。
