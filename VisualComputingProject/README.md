# SWS3026 Visual Computing - Unified Baseline

本目录是三位开发者分支合并后的统一入口，严格对应
`Visual_Computing_Project.pdf` 的 Beginner、Expert 和 Bonus 三层任务。

## 目录

```text
VisualComputingProject/
|-- beginner_level/            # Haar + LBF 68 点实时检测与鲁棒性分析
|-- expert_level/              # keypoint-only 表情分类、PCA/K-fold、实时特效
|-- bonus_level/               # Just Dance 双面板、时空对齐和评分
|-- bonus_level_action_game/   # Sherry 的 3D pose runner
|-- bonus_level_mario/         # Zhangyx 的摄像头 Mario 平台游戏
|-- resources/                 # 统一模型、FER zip、参考视频
`-- project_materials/         # starter/reference 材料
```

完整的来源、模型取舍和架构见仓库根目录 `merge.md`；逐页 presentation 结构、
讲解重点和实验证据见 `../PPT_GUIDELINES.md`。

## 环境

```powershell
conda env create -f environment_setup/environment.yml
conda activate vc_sws3026
python -m pip install -r environment_setup/requirements.txt
```

不要同时安装 `opencv-python` 与 `opencv-contrib-python`；LBF 依赖后者提供的
`cv2.face`。

## 运行

Beginner：

```powershell
.\run_beginner_level.ps1
```

Expert：

```powershell
.\run_expert_level.ps1 inspect
.\run_expert_level.ps1 demo --mirror
.\run_expert_level.ps1 train --cv-folds 5 --workers 4
```

默认 demo 使用已集成的 Zhangyx Geometry-SVM；历史 FER test Macro-F1 为
0.4633，单样本分类为 18.08 ms。正式统一重训仍应使用 train 内
Stratified K-fold，并在选模后只评估一次 official test。

Bonus Just Dance：

```powershell
.\run_bonus_level.ps1
```

Bonus Mario：

```powershell
.\run_mario.ps1 --check
.\run_mario.ps1
```

3D Runner：

```powershell
python VisualComputingProject/bonus_level_action_game/main.py
```

所有 webcam/GUI demo 均需在本机运行。按 `q`/`Esc` 或使用 GUI Stop
正常释放摄像头。
