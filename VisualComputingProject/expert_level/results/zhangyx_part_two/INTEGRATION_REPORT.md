# Zhangyx Part Two 集成说明

本目录保存 GitHub `zhangyx-project-upload` 最新 Part Two 的正式实验结果，并将
其中的完整 174-D Geometry-SVM 作为统一项目的默认实时模型。

## 统一入口中的位置

- 运行模型：`../../models/keypoint/current/expression_classifier.joblib`
- 兼容几何特征：`../../src/expression_features.py`
- 实时入口：`../../main.py demo --mirror`
- K-fold 统一重训：`../../main.py train --cv-folds 5 --workers 4`

模型只接收 LBF 68 点产生的 136 维眼对齐坐标；38 个距离、比例、角度和不对称
特征由保存的 sklearn Pipeline 在内部计算，不读取人脸像素。

## 公平对比

| 模型 | 输入 | Accuracy | Macro-F1 | 单样本预测 |
|---|---|---:|---:|---:|
| HGB | 136 coordinates | 0.4313 | 0.3914 | 23.41 ms |
| Base RBF-SVM | 136 coordinates | 0.4631 | 0.4347 | 5.17 ms |
| PCA95 + RBF-SVM | 12 PCs | 0.4032 | 0.3613 | 2.68 ms |
| Tuned RBF-SVM | 136 coordinates | 0.4684 | 0.4589 | 5.31 ms |
| Geometry-SVM（默认） | 136 + 38 geometry | 0.4731 | **0.4633** | 18.08 ms |
| `drop_eyes` ablation | 136 + 28 geometry | **0.4737** | 0.4625 | 4.90 ms |

默认选择 Geometry-SVM 是因为项目以 Macro-F1 和类别平衡为主，而不是只追求
Accuracy。`drop_eyes` 的 Accuracy 略高，但 Macro-F1 略低；它保留为消融证据，
不应被表述为“眼部特征无用”。

## 可直接用于 PPT 的证据

- `06_per_class_f1_comparison.png`：六类方案的 per-class F1 对比。
- `geometry_svm_confusion_matrix.png`：最终混淆矩阵。
- `geometry_svm_failure_cases.png`：最终模型高置信错误。
- `02_geometry_rescues.png`：显式几何纠正坐标模型的案例。
- `03_pca_information_loss.png`：PCA 丢失判别信息的案例。
- `04_geometry_regressions.png`：增加几何量后反而退化的案例。
- `05_all_models_wrong.png`：keypoint-only 表征上限、标签歧义和坏图。
- `group_ablation_validation.png`：五组几何特征 add/drop 消融。

所有数值均来自相邻 JSON/CSV；不要把本次合并的 smoke test 当成新的正式 FER
结果。统一重训的正式结果应另存到 `results/keypoint_cv/`。
