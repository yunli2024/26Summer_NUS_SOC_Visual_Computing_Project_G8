# Expert Level - Keypoint-only Expression Classification

本目录严格遵循项目 PDF：分类器输入仅由 68 个 facial keypoints 派生，
不会把完整人脸图像、眼睛 ROI 或嘴部 ROI 输入分类器。

## 当前可运行主模型（以 GitHub Part Two 为主）

`models/keypoint/current/expression_classifier.joblib` 是 Zhangyx Part Two 的
最终 Geometry-SVM：

- 输入：眼中心对齐后的 136 维坐标；模型 Pipeline 内追加 38 维关键点几何量；
- FER-2013 测试集 Macro-F1：0.4633，Accuracy：0.4731；
- 单样本分类预测：约 18.08 ms，满足小于 30 ms 的目标；
- 摄像头：YuNet/Haar 检脸 + LBF 关键点 + 多脸跟踪 + 概率时序平滑；
- 表情触发 sparkle、burst、red tint、blue tint 等实时特效。

原 Liyunzang RBF-SVM 仍可从合并前的 Git 历史恢复，原 Zhangyx HGB 保存在
`models/keypoint/legacy_hgb/`。模型对比、混淆矩阵、失败案例和消融图位于
`results/zhangyx_part_two/`。

上述指标来自 GitHub 分支的历史正式训练结果，不是本次合并重新训练所得。

## 统一改进训练

新的 `train` 默认执行：

1. 从 FER zip 的官方 `train`/`test` split 读取数据；
2. 每张图运行 Haar + LBF，得到 68 个关键点；
3. 使用眼中心消除平移、旋转与尺度，并加入眼/眉/嘴几何特征；
4. 仅在 train split 上做 5-fold `StratifiedKFold`；
5. 比较 `PCA + Linear SVM`、`PCA + Logistic Regression` 和 HGB；
6. 按 Macro-F1 均值、折间稳定性、balanced accuracy 和 `<30 ms` 约束选模；
7. 选模完成后只在官方 test split 上评估一次；
8. 输出 confusion matrix、抽取失败和高置信误分类案例。

Accuracy 只作为参考记录，不参与主要选模目标。

## 使用

```powershell
python VisualComputingProject/expert_level/main.py inspect
python VisualComputingProject/expert_level/main.py demo --mirror
python VisualComputingProject/expert_level/main.py train --cv-folds 5 --workers 4
```

快速验证训练链路可限制每类样本数：

```powershell
python VisualComputingProject/expert_level/main.py train `
  --max-train-per-class 100 --max-test-per-class 50 --cv-folds 3
```

重新训练会覆盖 `models/keypoint/current/expression_classifier.joblib`。因此正式
重训前请先保留当前模型；历史模型均已在 `models/keypoint/legacy_*` 归档。

## 为什么保留两条训练证据

- GitHub Part Two 提供完整的 HGB、基础 SVM、PCA-SVM、调参 SVM、
  Geometry-SVM 和几何分组消融，适合讲模型差异与失败案例。
- 统一 `train` 入口补上 train split 内的 Stratified K-fold，并把 PCA 放在候选
  Pipeline 内；官方 test 只用于选模后的最终一次评估，满足合并后的实验规范。
- 当前 demo 优先使用历史表现最好的 keypoint-only Geometry-SVM；若正式 K-fold
  重训选出更稳定且小于 30 ms 的模型，再用统一训练输出替换它。

## 关于历史 ROI-CNN

Sherry 分支的 ROI-CNN 文件和评估证据仍保留，作为
“高图像分类分数但违反 keypoint-only 任务定义”的 analysis case。它不再由
`main.py` 暴露，也不是统一 baseline 的候选模型。
