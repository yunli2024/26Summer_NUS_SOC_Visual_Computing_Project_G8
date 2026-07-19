# VisualComputingProject 运行命令指南

本文档说明三个任务目录的常用运行命令：

```text
VisualComputingProject\beginner_level
VisualComputingProject\bonus_level
VisualComputingProject\expert_level
```

以下命令默认在项目工作目录运行：

```powershell
cd D:\AAA_SHERRY\NUS_school_computing_summer_workshop\VisualComputingProjects\project
```

推荐统一使用 Conda 环境中的 Python：

```powershell
D:\miniconda3\envs\vc_sws3026\python.exe
```

如果摄像头窗口打开后需要退出，通常按 `q` 关闭；Bonus Level 的 Tkinter GUI 直接关闭窗口即可。

## 1. Beginner Level：人脸检测与 68 点人脸关键点

目录：

```text
VisualComputingProject\beginner_level
```

### 1.1 环境安全检查

该命令只检查依赖、模型文件和基础模块，不会打开摄像头。

```powershell
D:\miniconda3\envs\vc_sws3026\python.exe VisualComputingProject\beginner_level\tests\check_part2_setup.py
```

### 1.2 运行原始摄像头检测程序

打开摄像头，检测人脸并显示 68 点人脸关键点：

```powershell
D:\miniconda3\envs\vc_sws3026\python.exe VisualComputingProject\beginner_level\main.py
```

启用 CLAHE 图像增强：

```powershell
D:\miniconda3\envs\vc_sws3026\python.exe VisualComputingProject\beginner_level\main.py --clahe
```

指定摄像头编号。如果默认摄像头 `0` 打不开，可以尝试 `1`：

```powershell
D:\miniconda3\envs\vc_sws3026\python.exe VisualComputingProject\beginner_level\main.py --camera 1
```

### 1.3 运行改进版检测程序

改进版增加了人脸框缓存、过滤和平滑逻辑，建议优先用于展示。

先运行改进版安全检查：

```powershell
D:\miniconda3\envs\vc_sws3026\python.exe VisualComputingProject\beginner_level\improved\tests\check_improved_setup.py
```

运行改进版摄像头 demo：

```powershell
D:\miniconda3\envs\vc_sws3026\python.exe VisualComputingProject\beginner_level\improved\main.py
```

运行改进版并启用 CLAHE：

```powershell
D:\miniconda3\envs\vc_sws3026\python.exe VisualComputingProject\beginner_level\improved\main.py --clahe
```

使用推荐的人脸检测参数：

```powershell
D:\miniconda3\envs\vc_sws3026\python.exe VisualComputingProject\beginner_level\improved\main.py --scale-factor 1.05 --min-neighbors 5 --min-size 80 --max-size 420
```

常见显示状态：

```text
DETECTED：当前帧检测到人脸。
CACHED：当前帧暂时没检测到人脸，正在短暂显示上一帧稳定结果。
LOST：当前没有可用人脸结果。
```

## 2. Bonus Level：人体姿态检测与舞蹈评分 GUI

目录：

```text
VisualComputingProject\bonus_level
```

该任务会打开一个 Tkinter 图形界面。左侧播放参考舞蹈视频，右侧打开摄像头检测用户动作，并实时计算舞蹈相似度分数。

### 2.1 启动 Bonus Level GUI

```powershell
D:\miniconda3\envs\vc_sws3026\python.exe VisualComputingProject\bonus_level\main.py
```

GUI 打开后：

```text
Open：选择参考视频。
Start：开始播放参考视频。
Pause / Resume：暂停或继续参考视频。
Stop / Restart：停止或重新开始参考视频。
Start Webcam：打开摄像头开始用户姿态检测。
Stop Webcam：关闭摄像头。
Reset Score：重置分数。
Show Summary：显示本次舞蹈评分总结。
```

### 2.2 默认资源位置

YOLOv8 pose 模型优先查找：

```text
VisualComputingProject\resources\pose_models\yolov8n-pose.pt
```

备用模型路径：

```text
VisualComputingProject\bonus_level\legacy_pose_app\yolov8n-pose.pt
```

默认参考视频路径：

```text
VisualComputingProject\resources\videos\dance_example_1.mp4
```

如果默认视频不存在，可以在 GUI 中点击 `Open` 手动选择 `.mp4`、`.avi`、`.mov` 或 `.mkv` 文件。

### 2.3 推荐操作顺序

```text
1. 启动 GUI。
2. 点击 Open 选择参考视频。
3. 点击 Start 播放参考视频。
4. 点击 Start Webcam 打开摄像头。
5. 跟随参考视频跳舞，观察 Score、Feedback、Combo 和 Summary。
```

## 3. Expert Level Optimization：表情识别鲁棒优化实验

目录：

```text
VisualComputingProject\expert_level
```

该目录是独立优化实验区。它会读取 `expert_level` 中已有的数据、ROI-CNN 模型和缓存，但新的 robust 模型、评估报告和实时 audit 结果都写入 `expert_level`，不会覆盖原始工程。

### 3.1 查看可用命令

```powershell
D:\miniconda3\envs\vc_sws3026\python.exe VisualComputingProject\expert_level\main.py --help
```

### 3.2 跑原始 ROI-CNN baseline 评估

评估原始 `best_roi_cnn_face_eyes_mouth.pth` 在 clean、eye、mouth、random 四种测试条件下的表现：

```powershell
D:\miniconda3\envs\vc_sws3026\python.exe VisualComputingProject\expert_level\main.py baseline
```

主要输出：

```text
VisualComputingProject\expert_level\results\baseline_roi_cnn_report.json
VisualComputingProject\expert_level\results\baseline_roi_cnn_*_confusion_matrix.png
VisualComputingProject\expert_level\results\baseline_roi_cnn_*_failures.csv
```

### 3.3 训练 robust ROI-CNN

从原始 ROI-CNN checkpoint 初始化，加入遮挡增强进行微调训练：

```powershell
D:\miniconda3\envs\vc_sws3026\python.exe VisualComputingProject\expert_level\main.py train-robust-roi
```

指定初始化 checkpoint：

```powershell
D:\miniconda3\envs\vc_sws3026\python.exe VisualComputingProject\expert_level\main.py train-robust-roi --resume-from VisualComputingProject\expert_level\models\best_roi_cnn_face_eyes_mouth.pth
```

主要输出：

```text
VisualComputingProject\expert_level\models\robust_roi_cnn_face_eyes_mouth.pth
VisualComputingProject\expert_level\results\robust_roi_cnn_training_history_face_eyes_mouth.csv
```

### 3.4 评估 robust ROI-CNN 与融合模型

比较三种模型：

```text
baseline_roi_cnn：原始 ROI-CNN。
robust_roi_cnn：遮挡增强后的 robust ROI-CNN。
roi_ensemble：原始 ROI-CNN + robust ROI-CNN 概率融合。
```

运行命令：

```powershell
D:\miniconda3\envs\vc_sws3026\python.exe VisualComputingProject\expert_level\main.py evaluate-robust-roi
```

主要输出：

```text
VisualComputingProject\expert_level\results\robust_roi_cnn_comparison.csv
VisualComputingProject\expert_level\results\robust_roi_cnn_classification_report.csv
VisualComputingProject\expert_level\results\robust_roi_cnn_focus_errors.csv
VisualComputingProject\expert_level\results\roi_ensemble_*_confusion_matrix.png
```

优先查看：

```text
VisualComputingProject\expert_level\results\robust_roi_cnn_comparison.csv
```

### 3.5 实时多脸检测/跟踪 audit

该命令用于记录实时检测质量，不打开主 demo 效果界面，适合分析多脸、遮挡、漏检、track id 跳变等问题。

使用摄像头：

```powershell
D:\miniconda3\envs\vc_sws3026\python.exe VisualComputingProject\expert_level\main.py realtime-audit --camera 0 --max-frames 300
```

如果默认摄像头不可用：

```powershell
D:\miniconda3\envs\vc_sws3026\python.exe VisualComputingProject\expert_level\main.py realtime-audit --camera 1 --max-frames 300
```

使用视频文件进行可重复测试：

```powershell
D:\miniconda3\envs\vc_sws3026\python.exe VisualComputingProject\expert_level\main.py realtime-audit --video path\to\test_video.mp4 --max-frames 600
```

切换预处理模式：

```powershell
D:\miniconda3\envs\vc_sws3026\python.exe VisualComputingProject\expert_level\main.py realtime-audit --camera 0 --preprocess raw
D:\miniconda3\envs\vc_sws3026\python.exe VisualComputingProject\expert_level\main.py realtime-audit --camera 0 --preprocess clahe
D:\miniconda3\envs\vc_sws3026\python.exe VisualComputingProject\expert_level\main.py realtime-audit --camera 0 --preprocess gamma
D:\miniconda3\envs\vc_sws3026\python.exe VisualComputingProject\expert_level\main.py realtime-audit --camera 0 --preprocess clahe-gamma
```

主要输出：

```text
VisualComputingProject\expert_level\realtime_audit\realtime_audit_report.json
VisualComputingProject\expert_level\realtime_audit\realtime_audit_frames.csv
```

常看字段：

```text
frames_processed：处理帧数。
average_fps：平均处理 FPS。
created_tracks：创建过的人脸 track 数量。
duplicate_detection_frames：存在重复检测框的帧数。
no_face_frames：没有检测到人脸的帧数。
landmark_failure_frames：检测到脸但 landmark 拟合失败的帧数。
large_track_jumps：track 位置大跳变次数。
```

### 3.6 与 Expert Level demo 的关系

原始 ROI-CNN：

```powershell
D:\miniconda3\envs\vc_sws3026\python.exe VisualComputingProject\expert_level\main.py demo --model-source roi-cnn --multi-face
```

robust ROI-CNN：

```powershell
D:\miniconda3\envs\vc_sws3026\python.exe VisualComputingProject\expert_level\main.py demo --model-source robust-roi-cnn --multi-face
```

原始 ROI-CNN 与 robust ROI-CNN 融合：

```powershell
D:\miniconda3\envs\vc_sws3026\python.exe VisualComputingProject\expert_level\main.py demo --model-source roi-ensemble --multi-face
```

显示调试信息：

```powershell
D:\miniconda3\envs\vc_sws3026\python.exe VisualComputingProject\expert_level\main.py demo --model-source roi-ensemble --multi-face --debug-hud
```

## 4. 快速推荐运行顺序

如果只是想快速展示三个任务，可以按下面顺序运行。

Beginner Level：

```powershell
D:\miniconda3\envs\vc_sws3026\python.exe VisualComputingProject\beginner_level\improved\main.py --clahe
```

Bonus Level：

```powershell
D:\miniconda3\envs\vc_sws3026\python.exe VisualComputingProject\bonus_level\main.py
```

Expert Optimization 结果复查：

```powershell
D:\miniconda3\envs\vc_sws3026\python.exe VisualComputingProject\expert_level\main.py evaluate-robust-roi
```

Expert 融合模型实时 demo：

```powershell
D:\miniconda3\envs\vc_sws3026\python.exe VisualComputingProject\expert_level\main.py demo --model-source roi-ensemble --multi-face
```

