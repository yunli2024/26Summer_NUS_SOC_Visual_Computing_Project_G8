# Visual Computing Project 统一运行命令

完整架构、模型取舍和答辩故事见仓库根目录 `merge.md`。

```powershell
conda activate vc_sws3026

# Beginner
.\run_beginner_level.ps1

# Expert 资源/模型检查、实时 demo、PCA + K-fold 训练
.\run_expert_level.ps1 inspect
.\run_expert_level.ps1 demo --mirror
.\run_expert_level.ps1 train --cv-folds 5 --workers 4

# Bonus Just Dance
.\run_bonus_level.ps1 --check
.\run_bonus_level.ps1

# Bonus Mario
.\run_mario.ps1 --check
.\run_mario.ps1

# Sherry 3D Runner
python VisualComputingProject/bonus_level_action_game/main.py
```

所有 webcam/GUI 程序必须在本机运行。若脚本找不到 Python，请先激活环境。
