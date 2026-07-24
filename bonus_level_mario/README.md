# Bonus Mario：摄像头姿态控制平台游戏

本目录整合了 Zhangyx Part Three 中 `bonus_level/mario_demo` 的完整 Mario
扩展，并继续作为独立程序运行。它不替代课程要求的 Just Dance 双面板应用。

程序复用项目公共资源：

```text
resources/pose_models/yolov8n-pose.pt
resources/videos/dance_example_1.mp4
```

姿态检测及骨架绘制使用本目录的 `pose_analyzer.py`，不会依赖
`bonus_level/mario_demo` 中的文件。

## 手势控制

当前版本只使用肩膀、手肘和手腕，不要求下半身进入画面：

- 向画面左侧举起或伸出一只手：人物向左跑。
- 向画面右侧举起或伸出一只手：人物向右跑。
- 双手同时举过肩膀：人物跳跃。
- 双手在胸前靠拢：人物下蹲。
- 双手自然放下：回到 `NEUTRAL`。

手势需要连续多帧确认，并包含中值滤波、滞回和落地前跳跃缓冲，用于减少
关键点抖动、误触和落地瞬间丢失跳跃指令。

## 游戏内容

- 完整滚动平台关卡、金币、砖块、管道、坑洞和终点。
- 移动平台、敌人、强化蘑菇、检查点、生命、分数和计时。
- 支持人物朝向、空中水平惯性、单向浮空平台和碰撞反馈。
- 使用项目自带的原创像素素材和 WAV 音效。
- 保留键盘备用操作。

## 运行

在项目根目录执行：

```powershell
conda activate vc_sws3026
python bonus_level_mario\mario_camera_demo.py --check
python bonus_level_mario\mario_camera_demo.py
```

也可以使用根目录入口：

```powershell
.\run_mario.ps1 --check
.\run_mario.ps1
```

GUI 打开后点击 `Start Camera`。如果摄像头不是 0 号：

```powershell
.\run_mario.ps1 --camera 1
```

CPU 推理较慢时：

```powershell
.\run_mario.ps1 --image-size 256
```

键盘备用操作：

- `A/D` 或 `←/→`：左右移动。
- `Space`、`W` 或 `↑`：跳跃。
- `S` 或 `↓`：下蹲。
- `R`：重新开始。

## 测试

```powershell
python -m unittest discover -s bonus_level_mario -p "test_*.py" -v
python bonus_level_mario\mario_camera_demo.py --check
```

如果需要重新生成项目自带的像素素材和音效：

```powershell
python bonus_level_mario\build_assets.py
```

## 限制

- YOLOv8 Pose 只有手腕关键点，不识别手指。
- 肩膀、手肘和手腕需要清晰可见。
- 当前是单人单关 Demo，没有多人模式和关卡编辑器。
