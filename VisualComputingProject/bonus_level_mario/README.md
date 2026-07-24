# Bonus Mario - 摄像头姿态控制平台游戏

这是 Zhangyx 分支的独立创意扩展。它不替代 PDF 要求的 Just Dance
双面板应用，而是在同一套 YOLOv8 Pose 人体关键点能力之上增加平台游戏。

## 关键能力

- 只需肩、肘、腕等上半身关键点，下半身出画仍可操作。
- 单手向左/右控制移动，双手举起跳跃，双手靠拢触发下蹲。
- 连续帧确认、滞回和落地动作缓冲，抑制关键点抖动造成的误触。
- 完整横版关卡、金币、砖块、管道、敌人、检查点、生命和终点。
- 自带原创像素素材与音效，并保留键盘备用操作。

共享资源：

```text
VisualComputingProject/resources/pose_models/yolov8n-pose.pt
VisualComputingProject/resources/videos/dance_example_1.mp4
```

## 运行

在仓库根目录执行：

```powershell
python VisualComputingProject/bonus_level_mario/mario_camera_demo.py --check
python VisualComputingProject/bonus_level_mario/mario_camera_demo.py
```

打开界面后点击 `Start Camera`。摄像头不是 0 号时：

```powershell
python VisualComputingProject/bonus_level_mario/mario_camera_demo.py --camera 1
```

CPU 推理较慢时可使用 `--image-size 256`。

键盘备用操作：`A/D` 或左右方向键移动，`Space/W/上` 跳跃，
`S/下` 下蹲，`R` 重开。

## 验证

```powershell
python -m unittest discover -s VisualComputingProject/bonus_level_mario -p "test_*.py" -v
python VisualComputingProject/bonus_level_mario/mario_camera_demo.py --check
```

`--check` 会读取共享视频并运行一次 YOLO/主舞者/手势链路，不打开 GUI。
