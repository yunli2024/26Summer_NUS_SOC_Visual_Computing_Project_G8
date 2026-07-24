# SWS3026 Visual Computing Group Project

NUS School of Computing Summer Workshop SWS3026 小组项目，主题为
**Real-time Video Analysis and Rendering**。

本项目围绕 facial landmarks、基于关键点的表情分类以及人体姿态匹配，提供四个可在
普通笔记本电脑本地运行的实时演示：

1. **Beginner Level**：webcam 人脸检测与 68 点 facial landmarks。
2. **Expert Level**：仅使用 facial keypoints 的七类表情分类与实时特效。
3. **Bonus Level**：reference video 与 webcam 双面板的 Just Dance 类姿态评分。
4. **Bonus Mario**：使用上半身姿态控制的平台游戏扩展。

> 项目必须在本地运行。Webcam、OpenCV GUI 和实时交互不支持 Google Colab 或
> Jupyter Notebook。

## 项目亮点

- Haar Cascade + OpenCV LBF 68 点人脸关键点检测。
- ROI 跟踪、定期全图校准、CLAHE、关键点平滑和实时 FPS 显示。
- 表情分类输入严格来自关键点：136 个归一化坐标与 38 个几何特征。
- 使用官方 FER-style `train` / `test` 划分，训练阶段支持 Stratified K-fold 和 PCA
  对照实验。
- 最终 class-balanced RBF-SVM 的单次预测约为 **18.08 ms**，满足 `< 30 ms`
  的实时目标。
- YOLOv8 Nano Pose 主舞者选择、空间归一化、动作相似度、镜像容忍和 reaction-lag
  时间匹配。
- Just Dance GUI 提供 score、combo、`Perfect` / `Super` / `Good`、`HOLD`
  和 `MOVE!` 等实时反馈。
- Mario 扩展支持摄像头手势、键盘 fallback、完整关卡、音效和原创像素素材。

## 目录结构

```text
project/
├─ beginner_level/          # Haar + LBF 实时人脸关键点
├─ expert_level/            # 表情特征、训练、评估、实时特效
├─ bonus_level/             # 舞蹈姿态分析、双面板 GUI、评分系统
├─ bonus_level_mario/       # 姿态控制平台游戏
├─ resources/               # 公共模型、数据入口和参考视频
├─ environment_setup/       # Conda 与 pip 环境配置
├─ docs/                    # 课程说明、报告和 Presentation 资料
├─ run_beginner_level.ps1
├─ run_expert_level.ps1
├─ run_bonus_level.ps1
└─ run_mario.ps1
```

各模块的详细设计与参数说明：

- [Beginner Level](beginner_level/docs/README.md)
- [Expert Level](expert_level/README.md)
- [Bonus Just Dance](bonus_level/README.md)
- [Bonus Mario](bonus_level_mario/README.md)
- [项目文档索引](docs/README.md)

## 环境安装

推荐使用项目提供的 Conda 环境：

```powershell
conda env create -f environment_setup\environment.yml
conda activate vc_sws3026
python -m pip install -r environment_setup\requirements.txt
```

核心环境为 Python 3.11，支持 CPU-only 运行，不要求 CUDA。

Facemark LBF 依赖 `cv2.face`，因此必须使用 `opencv-contrib-python`。如果环境中还安装了
普通 `opencv-python` 并出现包覆盖问题，请移除两个 OpenCV wheel 后仅重新安装：

```powershell
python -m pip uninstall -y opencv-python opencv-contrib-python
python -m pip install opencv-contrib-python
```

## 本地大文件准备

GitHub 仓库保留源码、最终分类器、YOLOv8 Nano Pose、Haar/YuNet 和演示视频。
以下课程资源因体积较大，不上传到 GitHub：

```text
resources/face_models/lbfmodel.yaml
resources/expression_data/facial_expression_dataset.zip
resources/expression_data/facial_expression_dataset/train/<class>/
resources/expression_data/facial_expression_dataset/test/<class>/
```

请从课程提供的原始材料恢复这些文件。FER-style 数据应包含七类：

```text
angry  disgust  fear  happy  neutral  sad  surprise
```

只有运行 Beginner、Expert 或重新训练分类器时需要 LBF/FER 资源。Bonus Level 和 Mario
使用仓库内已包含的 `resources/pose_models/yolov8n-pose.pt`。

## 快速运行

所有命令均从项目根目录执行，并先激活 `vc_sws3026` 环境。

### Beginner：实时人脸关键点

```powershell
.\run_beginner_level.ps1
```

启用 CLAHE：

```powershell
.\run_beginner_level.ps1 --preprocess clahe
```

按 `Q` 退出并释放摄像头。

### Expert：实时表情分类与特效

```powershell
.\run_expert_level.ps1
```

运行时可使用：

- `E`：开关表情特效。
- `L`：开关 facial landmarks。
- `S`：开关时间平滑。
- `C`：开关 CLAHE。
- `Q` / `Esc`：退出。

无摄像头生成特效预览：

```powershell
.\run_expert_level.ps1 --preview expert_level\artifacts\effects_preview.png
```

### Bonus：Just Dance 双面板应用

先验证模型、参考视频和 pose cache：

```powershell
.\run_bonus_level.ps1 --check
```

启动 GUI：

```powershell
.\run_bonus_level.ps1
```

如果 CPU 推理速度不足，可降低输入尺寸：

```powershell
.\run_bonus_level.ps1 --image-size 320
```

### Bonus Mario：姿态控制平台游戏

```powershell
.\run_mario.ps1 --check
.\run_mario.ps1
```

如果摄像头不是 0 号：

```powershell
.\run_mario.ps1 --camera 1
```

## Expert 模型结果

最终模型为 geometry-augmented、class-balanced RBF-SVM。模型选择在训练集内部完成，
官方 test split 仅用于最终评估。

| 指标 | 结果 |
|---|---:|
| Test samples | 7,178 |
| Accuracy | 47.31% |
| Macro-F1 | 46.33% |
| Weighted-F1 | 47.35% |
| 单次预测延迟 | 18.08 ms / image |
| `< 30 ms` 实时要求 | 通过 |

完整指标、confusion matrix 和 failure cases 位于：

- [metrics.json](expert_level/artifacts_svm_geometry/metrics.json)
- [confusion_matrix.png](expert_level/artifacts_svm_geometry/confusion_matrix.png)
- [failure_cases.png](expert_level/artifacts_svm_geometry/failure_cases.png)
- [Expert Task 1 Report](expert_level/TASK1_REPORT.md)

## Bonus 姿态分析结果

仓库附带的参考视频 pose cache 包含 1,340 帧。CPU 验证结果：

| 指标 | 结果 |
|---|---:|
| Primary dancer detection rate | 91.12% |
| Average visible keypoints | 16.76 / 17 |
| Average pose confidence | 0.932 |
| Average inference time | 19.42 ms |
| Offline processing speed | 35.71 FPS |

评分系统的空间归一化、关节角度、运动分量、mirror matching、reaction-lag window 和
HOLD hysteresis 详见
[Bonus Scoring Report](bonus_level/LEVEL3_DANCE_SCORING_DETAILED_REPORT.md)。

## 验证

无摄像头检查：

```powershell
python beginner_level\tests\check_part2_setup.py
python expert_level\test_expression_features.py
python bonus_level\test_dance_scoring.py
python -m unittest discover -s bonus_level_mario -p "test_*.py" -v
```

模型与 demo 输入检查：

```powershell
.\run_bonus_level.ps1 --check
.\run_mario.ps1 --check
```

摄像头和 GUI 必须在目标 laptop 上最终测试。建议 Demo Day 前分别确认 camera index、
光照、人物距离、FPS 和退出键。

## 已知限制

- Haar + LBF 对大角度侧脸、严重遮挡、快速运动和极端光照仍较敏感。
- FER 图像分辨率低且类别不平衡，`fear`、`sad` 与 `angry` 较容易混淆。
- CPU-only YOLO webcam 推理可能低于参考视频播放帧率；应用使用缓存和时间窗口保持同步。
- YOLOv8 Pose 只提供身体关键点，不识别手指。
- 实时效果会受摄像头质量、背景复杂度和设备性能影响。

## 文档

- [课程项目说明](docs/course_materials/Visual_Computing_Project.pdf)
- [Presentation 资料](docs/presentation/)
- [Beginner 鲁棒性记录](beginner_level/docs/robustness_test.md)
- [Expert 训练报告](expert_level/LEVEL2_TRAINING_DETAILED_REPORT.md)
- [Bonus Task 1 Report](bonus_level/TASK1_REPORT.md)
- [Bonus Task 2 Report](bonus_level/TASK2_REPORT.md)
