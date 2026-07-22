# Bonus Task 1 操作手册

## 目标

本目录完成 Bonus Level Task 1：对 reference dance video 做 YOLOv8 body keypoint detection，并在 GUI 中可视化 skeleton。当前任务只做 reference video，不接 webcam 评分，那是 Bonus Task 2 的范围。

## 文件

| 文件 | 作用 |
|---|---|
| `bonus_task1_app.py` | Tkinter GUI，播放 reference video 并实时画 pose skeleton |
| `bonus_task2_app.py` | Task 2 双 panel GUI，左侧 reference，右侧 webcam/user，并实时评分 |
| `pose_pipeline.py` | YOLOv8 pose 推理、主舞者选择、skeleton 绘制 |
| `pose_scoring.py` | 姿态归一化、空间对齐、时间窗口匹配和相似度评分 |
| `analyze_reference.py` | 无 GUI benchmark/分析脚本，输出 metrics 和 sample frames |
| `analyze_segments.py` | 多片段评估脚本，用于覆盖视频开头、中段和后段 |
| `analyze_task2_simulation.py` | Task 2 无摄像头模拟评估脚本 |
| `analyze_scoring_robustness.py` | 正/负动作对照评估，检查错误动作是否仍被虚高评分 |
| `test_pose_scoring.py` | 姿态归一化和时间匹配的轻量单元测试 |
| `dance_example_1.mp4` | 默认 reference video |
| `yolov8n-pose.pt` | YOLOv8 nano pose model |
| `bonus_task1_report.md` | Task 1 方法、观察和挑战分析 |
| `bonus_task2_report.md` | Task 2 方法、评分系统、验证结果和局限性 |

## 环境

建议复用前面创建的 Conda 环境。`opencv-contrib-python` 可以兼容前面 Beginner/Expert 中用到的 `cv2.face`，同时本任务会使用 Ultralytics YOLOv8。

```powershell
conda activate sws3026-bonus
cd D:\26spring_NUS_SOC_SWS\project\demo_0715
python -m pip install -r requirements.txt
```

如果普通终端找不到 `conda`，先打开 Anaconda Prompt；或者确认 Miniconda/Anaconda 的 `condabin` 已加入系统环境变量。

检查摄像头是否能被 OpenCV 打开：

```powershell
python -c "import cv2; cap=cv2.VideoCapture(0); print('camera_open', cap.isOpened()); cap.release()"
```

当前本机验证结果为 `camera_open True`。

如果这里输出 `False`，先关闭占用摄像头的软件，例如会议软件或浏览器页面，并检查系统摄像头权限。

## 逐步运行流程

### 1. 进入项目目录

```powershell
conda activate sws3026-bonus
cd D:\26spring_NUS_SOC_SWS\project\demo_0715
```

### 2. 安装依赖

```powershell
python -m pip install -r requirements.txt
```

### 3. 检查脚本是否能被 Python 正常解析

```powershell
python -m py_compile pose_pipeline.py pose_scoring.py bonus_task1_app.py bonus_task2_app.py analyze_reference.py analyze_segments.py analyze_task2_simulation.py test_pose_scoring.py
```

评分模块单元测试：

```powershell
python test_pose_scoring.py
```

### 4. 运行 Task 1 GUI demo

```powershell
python bonus_task1_app.py
```

GUI 功能：

- `Open Video`：选择其他 reference video。
- `Start` / `Stop` / `Restart`：控制 reference video。
- `Show original video`：关闭后只看白底 skeleton。
- `Main dancer only`：只显示系统选出的主舞者。
- `Detector conf`：调整 YOLO person detection 阈值。
- `Keypoint conf`：调整 keypoint 可见阈值。
- `Save Snapshot`：保存当前带 skeleton 的截图到 `outputs/`。

### 5. 单片段无界面验证

处理前 120 帧，并保存 sample frames 和 metrics：

```powershell
python analyze_reference.py --frames 120 --save-samples 6
```

输出：

| 输出 | 内容 |
|---|---|
| `outputs/task1_summary.json` | detection rate、multi-person rate、平均 visible keypoints、平均推理耗时 |
| `outputs/task1_metrics.csv` | 每帧 persons、main visible keypoints、inference time |
| `outputs/sample_*.jpg` | 带 skeleton 的示例帧 |

### 6. 多片段评估

为了更全面地观察 reference video 的开头、中段和后段，运行：

```powershell
python analyze_segments.py --frames 120 --save-samples 3
```

当前已验证结果：

| 指标 | 数值 |
|---|---:|
| 总处理帧数 | 600 |
| person detection rate | 84.67% |
| multi-person frame rate | 57.50% |
| 主舞者平均可见关键点数 | 14.28 / 17 |
| 平均 YOLO 推理耗时 | 48.84 ms/frame |

输出位置：

| 输出 | 内容 |
|---|---|
| `outputs/task1_segments_summary.json` | 多片段汇总指标 |
| `outputs/segment_start*_frames120/task1_summary.json` | 每个片段的 summary，并记录本次有效 sample 文件名 |
| `outputs/segment_start*_frames120/sample_frame*.jpg` | 每个片段的 skeleton 示例图；以 summary 里的 `sample_files` 为准 |

## Bonus Task 2 运行流程

### 1. 运行 webcam 双 panel 评分 GUI

```powershell
python bonus_task2_app.py
```

默认行为：

- 左侧播放 `dance_example_1.mp4` reference video。
- 右侧打开 webcam `0`。
- 两侧都会显示 body skeleton。
- 右侧显示用户动作相对 reference 的实时 score、feedback 和 combo。

现场展示时，用户需要尽量让上半身或全身进入 webcam 画面。若状态显示 `Find User`，通常是摄像头没有拍到足够多的人体关键点，先后退一点或调整摄像头角度。

如果电脑有多个摄像头，可以指定摄像头编号：

```powershell
python bonus_task2_app.py --user-source 1
```

为了避开 reference video 开头舞者尚未稳定入画的部分，现场展示可以从第 900 帧开始：

```powershell
python bonus_task2_app.py --ref-start-frame 900
```

### 2. 无摄像头时用视频模拟 user input

```powershell
python bonus_task2_app.py --user-source dance_example_1.mp4 --ref-start-frame 900 --user-start-frame 900
```

这不是最终展示模式，但可以检查双 panel、两路 pose detection、评分和反馈是否工作。

### 3. 运行 Task 2 离线评分验证

模拟 user 动作比 reference 晚 8 帧：

```powershell
python analyze_task2_simulation.py --frames 90 --simulated-user-lag 8 --lag-window 15 --save-samples 4
```

当前已验证结果：

| 指标 | 数值 |
|---|---:|
| 处理 frame pairs | 90 |
| lag 后有效评分 pairs | 82 |
| lag 后平均分 | 100.00 |
| lag 命中率 | 100.00% |
| pair processing FPS | 8.55 |

优化后的最终验证：

| 指标 | 数值 |
|---|---:|
| 已知 8 帧 lag 平均分 | 99.30 |
| lag 命中率 | 100.00% |
| pair processing FPS | 12.20 |
| 600 帧错位动作平均分 | 65.54 |
| 错误动作达到 `Super` 的比例 | 0.00% |

运行严格评分对照：

```powershell
python analyze_scoring_robustness.py --samples 30 --wrong-offset 600
```

额外对照评估：

| 场景 | 帧数 | 平均分 | lag 命中率 | 说明 |
|---|---:|---:|---:|---|
| lag 0, window 0 | 60 | 100.00 | 100.00% | 同步输入应直接高分 |
| lag 8, window 3 | 60 | 91.14 | 0.00% | 窗口太小，找不回 8 帧延迟 |
| lag 8, window 15 | 90 | 100.00 | 100.00% | 窗口足够，能恢复延迟匹配 |

输出位置：

| 输出 | 内容 |
|---|---|
| `outputs/task2_simulation/task2_summary.json` | Task 2 模拟评分汇总 |
| `outputs/task2_simulation/task2_metrics.csv` | 每帧 score、feedback、matched frame、lag |
| `outputs/task2_simulation/task2_sim_frame*.jpg` | 左右 panel 合成示例图 |

### 4. Task 2 GUI 控件

- `Open Reference`：选择 reference dance video。
- `Open User Video`：选择一个视频文件模拟用户输入。
- `Use Webcam`：切回 webcam `0`。
- `Start` / `Stop` / `Restart`：控制双路输入。
- `Main dancer only`：只显示主舞者 skeleton，适合多人场景。
- `Mirror user`：镜像 webcam，更接近用户照镜子的体验；如果左右手动作评分感觉反了，可以关闭它再试。
- `Lag frames`：时间匹配窗口，允许用户比 reference 慢若干帧。
- `Save Snapshot`：保存当前左右 panel 合成图到 `outputs/`。

## Task 1 汇报重点

- 使用 `yolov8n-pose.pt` 直接做 pose detection，不训练新模型。
- 主舞者选择基于 bbox 面积、可见关键点数量和画面中心位置。
- 实测中视频开头 detection rate 较低，因为舞者还没有稳定进入画面。
- 中后段 detection 基本稳定，但多人帧比例很高，所以主舞者选择是必要模块。
- 多人、遮挡、快速动作、低分辨率、身体出画会影响 keypoint 稳定性。
- partial body visible 时，系统仍显示可见关键点，但下游评分应只比较双方都可见的关键点。

## Task 2 汇报重点

- 不训练新模型，继续使用 `yolov8n-pose.pt` 做 reference 和 webcam 两路 pose detection。
- 空间对齐：以肩膀/髋部中心平移，用肩宽、髋宽、躯干高度和 body box 尺度归一化。
- 时间对齐：保存最近若干帧 reference poses，用 sliding window 找与当前用户 pose 最像的一帧。
- 相似度：结合置信度加权关键点距离、8 个关节角、10 条肢体方向和共同可见度质量。
- 镜像兼容：自动比较 direct / mirrored pose，避免 webcam 镜像造成左右手错误扣分。
- 主舞者稳定：按上一帧位置、框重叠和可见度连续跟踪，不再每帧重新抢主舞者。
- 反馈：根据 score 显示 `Perfect`、`Super`、`Good`、`Keep Going`、`Miss` 或 `Partial`。
- Combo：连续 `Super` / `Perfect` 会累积 combo，低于阈值会归零。
- GUI 显示分数使用 5 帧移动平均来减少 webcam 抖动；离线 CSV 记录的是原始逐帧分数。
- 注意 CPU-only 情况下两路 YOLO 推理会比 Task 1 慢；展示时可以降低 camera 分辨率、只显示 main dancer、调低 confidence。

