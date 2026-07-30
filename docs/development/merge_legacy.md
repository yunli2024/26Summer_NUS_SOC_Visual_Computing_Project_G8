# Legacy Integration Record

> Historical only. Paths, metrics, test counts, and runtime claims in this
> file may describe an earlier integration candidate. Use `merge.md` and the
> root README for the maintained architecture.

# 三分支合并说明：Unified Visual Computing Baseline

## 1. 合并结论

本仓库已将三位开发者的独立实现整理成一个统一 baseline：

| 本地分支 | 来源 | 定位 |
|---|---|---|
| `liyunzang_project` | 原团队 `main` | Liyunzang 原始实现，保留不改 |
| `sherry-project-upload` | Sherry 上传分支 | Sherry 原始实现，保留不改 |
| `zhangyx-project-upload` | Zhangyx 上传分支 | Zhangyx 原始实现，保留不改 |
| `main` | 三分支融合 | 后续开发的唯一 baseline |

`main` 采用统一目录、统一资源和统一入口。原分支仍可随时 checkout，用于追溯
作者代码、报告与历史指标。

## 2. PDF 要求与合并后的对应关系

| PDF 要求 | 合并后的实现 |
|---|---|
| Beginner：webcam 实时显示 | `beginner_level/main.py` |
| Haar 检脸和 bounding box | `beginner_level/src/face_detector.py` |
| LBF 68 facial keypoints | `beginner_level/src/landmark_detector.py` |
| 光照、转头、遮挡鲁棒性分析 | `beginner_level/BEGINNER_LEVEL_REPORT_BILINGUAL.md` |
| Expert 必须以 keypoints 分类 | `expert_level/src/keypoint_features.py` |
| FER train/test 划分 | 直接读取共享 FER zip 中官方 split |
| 约 30 ms/次预测 | 训练和实时程序都测单样本分类延迟 |
| confusion matrix / failure cases | `expert_level/results/` 和新训练输出 |
| 表情驱动实时特效 | `expert_level/src/expression_effects.py` |
| Bonus：reference + webcam 双 panel | `bonus_level/src/app.py` |
| 人体 keypoint / 主舞者选择 | `pose_detector.py` + `dancer_tracker.py` |
| 空间与时间对齐 | `pose_normalizer.py` + `temporal_alignment.py` |
| 相似度、分数、Perfect/Super/Good | `pose_similarity.py` + `score_manager.py` |
| 创意 bonus activity | 3D runner + Mario 平台游戏 |

最关键的边界是：Expert 默认分类链路不再使用完整图像或 ROI 像素。Haar/YuNet
只负责找脸，LBF 负责产生 68 个点，分类器看到的只有关键点坐标和关键点几何量。

## 3. 总体目录与共享资源

```text
project/
|-- beginner_level/
|-- expert_level/
|   |-- models/keypoint/current/
|   |-- results/keypoint_cv/            # 正式重训后生成
|   |-- results/legacy_liyunzang_svm/
|   |-- results/legacy_zhangyx_hgb/
|   |-- src/
|   `-- tests/
|-- bonus_level/
|-- bonus_level_mario/
|-- resources/
|   |-- expression_data/facial_expression_dataset.zip
|   |-- face_models/
|   |   |-- haarcascade_frontalface_default.xml
|   |   |-- lbfmodel.yaml
|   |   `-- face_detection_yunet_2023mar.onnx
|   |-- pose_models/yolov8n-pose.pt
|   `-- videos/
`-- docs/
```

共享资源只保留一套 canonical path，避免各目录重复模型。两个分支提供的参考视频
内容不同，因此都保留在 `resources/videos/`。

## 4. Beginner Level

### 4.1 架构

```text
webcam frame
  -> raw / CLAHE / gamma preprocessing
  -> Haar face candidates
  -> ROI continuity + candidate de-duplication
  -> LBF 68-point fitting
  -> loose geometry sanity check
  -> box/keypoint temporal smoothing
  -> bounding box + color-coded landmarks + head pose + FPS
```

### 4.2 保留的优点

主要保留 Sherry 分支的增强版 Beginner：

- 完整满足 PDF 指定的 Haar + `cv2.face.createFacemarkLBF()` 路线；
- CLAHE/gamma 改善暗光和低对比度；
- 前一帧 ROI 搜索、短时 box cache 和 IoU 去重降低框闪烁；
- LBF 成功后再做宽松几何验证，过滤明显假脸/塌缩关键点；
- 68 点分区域平滑，下颌、眉鼻、眼嘴使用不同参数；
- `solvePnP` head pose 加鼻尖 2D yaw fallback，改善左右转头显示；
- 保留失败场景和 improvement 分析，不只展示成功画面。

### 4.3 Analysis -> Improvement 答辩故事

1. 最初 Haar 在暗光、侧脸和遮挡下容易漏检，单帧框还会跳动。
2. 只调 `scaleFactor/minNeighbors` 会在 recall 与 false positive 之间来回牺牲。
3. 加入 CLAHE 改善光照，但仍不能解决时间上的闪烁。
4. 因此增加 ROI continuity、短时缓存和 IoU 去重，处理视频连续性问题。
5. Haar 候选不能直接当作真脸；再用 LBF 是否能形成合理的 68 点做二次验证。
6. 严格几何规则又会误杀正常侧脸，因此最终改为“宽松 sanity check +
   head pose 仅用于显示”，把问题从硬拒绝改为稳定追踪。
7. Expert 实时模式额外提供 YuNet 作为替代 detector。YuNet 是学习式 detector，
   对姿态/背景更稳；但 Beginner 仍保留 Haar，以满足 PDF 指定的教学任务和公平比较。

## 5. Expert Level

### 5.1 当前可运行模型

当前 `models/keypoint/current/expression_classifier.joblib` 采用 Zhangyx GitHub
Part Two 最新完整 Geometry-SVM：

```text
68 LBF landmarks
  -> eye-center translation / eye-line rotation / inter-eye scaling (136)
  -> brow/eye/nose/mouth/global geometry (38)
  -> StandardScaler
  -> class-balanced tuned RBF SVM
```

历史报告数字：

| 指标 | Zhangyx Geometry-SVM |
|---|---:|
| FER test Macro-F1 | 0.4633 |
| FER test accuracy（仅参考） | 0.4731 |
| 单样本分类预测 | 18.08 ms |

实时工程继续使用统一版 YuNet/Haar、LBF、多脸跟踪、landmark EMA、概率 EMA、
低置信 `uncertain`、连续帧确认和 expression effects。保存模型中的 38 维几何
转换通过 `expert_level/src/expression_features.py` 兼容加载；模型和 detector 都
在循环外加载。

### 5.2 三个历史方案为何不能直接照搬

| 方案 | 优点 | 不足 | 合并决策 |
|---|---|---|---|
| Zhangyx：眼对齐坐标 + Geometry-SVM | Macro-F1 0.4633；分类 18.08 ms；模型消融和失败分析完整 | 使用 train 内固定 stratified holdout 选模 | 当前演示模型与主要 exploration 证据 |
| Liyunzang：153 维几何 + RBF-SVM | Macro-F1 0.4198；分类约 3.29 ms；实时稳定性好 | 固定超参数；RBF 全量重训慢；box normalization 对旋转不完全不变 | 历史快速基线 |
| Sherry：ROI-CNN ensemble | 历史 Macro-F1 0.6630，遮挡测试完整 | 分类输入是 face/eyes/mouth 像素；landmarks 只用于裁 ROI，违反 PDF keypoint-only | 仅保留为反例/研究历史，不进入默认入口 |

这里不能因为 ROI-CNN 的 accuracy/Macro-F1 更高就选它。项目评分关注的是是否
解决指定问题；用图像纹理绕开 keypoint-only 定义属于任务错位。

### 5.3 统一改进模型

新训练入口：`expert_level/src/train_keypoint.py`。

特征：

1. 计算左右眼中心；
2. 将眼中心中点平移到原点；
3. 将眼线旋转为水平；
4. 按眼间距缩放；
5. 保留 68x2 对齐坐标；
6. 加入眼开合、嘴开合、眉眼距离、鼻/下巴距离和局部角度；
7. 全程只使用 keypoints。

训练与选模：

```text
official FER train split
  -> fixed stratified train/validation holdout
  -> compare HGB, coordinate SVM, PCA95-SVM and Geometry-SVM
  -> tune C/gamma using validation Macro-F1
  -> refit selected model on all train
  -> evaluate exactly once on official test split
```

PCA 的作用不是为了“堆算法”，而是：

- 关键点 x/y 和多个几何量高度相关；
- LBF 在低清 FER 图像上会产生共同抖动方向；
- PCA 只在训练子集上拟合，避免 validation leakage；
- 保留 98% 方差并记录实际 component 数，减少冗余和线性模型计算量。

固定 validation 只从 official train split 中划分。官方 test 不参与调参或选模，
避免 test leakage。主要按 validation Macro-F1 选模，同时检查类别平衡和单样本
预测是否满足 30 ms。Accuracy 会被写入报告，但不能作为唯一结论；必须结合
per-class recall、confusion matrix 和 failure cases 解释。

正式训练将生成：

- `expert_metrics.json`
- `confusion_matrix.csv`
- `extraction_failures.csv`
- `misclassification_cases.csv`
- `svm_tuning_results.csv`
- 重新训练后的 `models/keypoint/current/expression_classifier.joblib`

合并阶段只对小样本执行了完整 smoke run，确认 PCA、选模、最终评估和
报告输出可以走通。没有把 smoke 数字伪装成正式实验结果。答辩前应在统一环境
执行一次全量训练，并把正式结果填入 presentation。

### 5.4 建议的答辩叙事

1. 第一版用纯坐标 + HGB，解决了 RBF-SVM 全量训练过慢的问题，但实时单样本
   余量不大，而且没验证模型选择是否稳定。
2. 第二版用几何特征 + RBF-SVM，Macro-F1 和延迟更好；后续通过 train 内固定
   validation 调整参数，box normalization 仍没有显式消除旋转。
3. ROI-CNN 的指标显著更高，却是在解决 image classification，不是题目要求的
   keypoint classification，所以将它降级为“错误但有启发的对照实验”。
4. 最终统一版回到问题定义：眼对齐 + 几何 keypoints，并将 PCA 作为受控对照，
   使用 train 内 validation 完成选模。
5. 最终判断不只看 Accuracy：看 Macro-F1、各类 recall、confusion、failure
   cases 和实时延迟。重点解释 fear/sad/neutral 为什么难，以及
   keypoints 不含皱纹/纹理这一表征上限。

## 6. Bonus Level：Just Dance

### 6.1 主链路

```text
reference video + webcam
  -> YOLOv8 Pose (17 COCO points)
  -> primary dancer selection
  -> confidence filter + missing-point handling + EMA
  -> mirror left/right correction
  -> torso-center translation + body-scale normalization
  -> one-way recent reference search (0-0.8 s reaction lag)
  -> position + joint angle + 0.4 s motion evidence
  -> coverage penalty + anti-static factor
  -> score EMA + combo + Perfect/Super/Good/Miss/Hold/Move
```

主舞者不是简单取“第一个人”。`dancer_tracker.py` 综合人物面积、画面中心、
与上一帧的连续性和 keypoint confidence，降低多人场景下身份切换。

空间对齐通过躯干中心平移和 shoulder/hip body scale 缩放消除镜头距离与站位；
镜像 webcam 后交换左右 COCO index。

时间对齐只把当前用户帧与过去 0.8 秒内的 reference 比较，不使用未来动作。
单帧 pose 由 normalized position 0.35 与 joint angle 0.65 组成，再乘 visibility
coverage penalty。拥有约 0.4 秒历史后，最终分由 pose 0.55 与 motion 0.45
组成；reference 在动而玩家静止时额外乘 anti-static factor。真实编舞停顿通过
EMA、确认和迟滞显示为 `Hold` 并停止计分。分数再做 EMA，避免反馈在
Perfect/Miss 间闪烁；结算同时记录 average/median/P90 matched lag。

### 6.2 保留内容

- Sherry：模块化双面板 app、主舞者 tracking、时序对齐、组合评分、声音反馈。
- Liyunzang：Task 1/2 分段和 scoring robustness 报告，位于
  `bonus_level/docs/branch_evidence/liyunzang/`。
- Zhangyx：Task 1/2 报告，位于
  `bonus_level/docs/branch_evidence/zhangyx/`。

这些报告用于展示“检测/评分失败 -> 调整主舞者选择、归一化、时间窗口、平滑”的
过程，而不是只报一个总分。

## 7. 创意功能

### 7.1 Zhangyx Mario

位置：`bonus_level_mario/`。

这是必须保留的 Zhangyx 独特 feature：

- 上半身关键点即可玩；
- 单手左右、双手举起跳、双手靠拢蹲；
- 连续帧确认、滞回、落地 jump buffer；
- 完整平台关卡、金币、砖块、管道、敌人、检查点、生命、音效；
- 22 个原创视觉/音频素材；
- 键盘 fallback；
- 28 个手势/游戏 mechanics 单元测试。

Mario 是 Bonus creative extension，不替代 PDF 的 Just Dance Task 1/2。

## 8. 安装与运行

### 8.1 环境

```powershell
conda env create -f environment_setup/environment.yml
conda activate vc_sws3026
python -m pip install -r environment_setup/requirements.txt
```

LBF 需要 `opencv-contrib-python` 的 `cv2.face`。不要同时安装普通
`opencv-python`，否则可能出现包覆盖。

### 8.2 Beginner

```powershell
python beginner_level/main.py
```

### 8.3 Expert

检查数据、模型和 feature version：

```powershell
.\run_expert_level.ps1 inspect
```

实时 demo：

```powershell
.\run_expert_level.ps1 demo --mirror
```

正式 PCA + 5-fold 训练：

```powershell
.\run_expert_level.ps1 train --cv-folds 5 --workers 4
```

小样本 smoke：

```powershell
.\run_expert_level.ps1 train `
  --max-train-per-class 100 `
  --max-test-per-class 50 `
  --cv-folds 3
```

### 8.4 Just Dance

```powershell
.\run_bonus_level.ps1 --check
.\run_bonus_level.ps1
```

### 8.5 Mario

```powershell
.\run_mario.ps1 --check
.\run_mario.ps1
```

## 9. 已完成的合并验证

一键复验（不打开 webcam GUI）：

```powershell
.\check_merge.ps1
```

- PDF：6 页已提取并逐页核对，特别确认 Expert 的 keypoint-only 限制。
- Beginner：`check_part2_setup.py` 全部 33 项通过。
- Expert keypoint feature：translation/rotation/scale invariance、finite vector、
  Zhangyx 136->174 维几何 Pipeline 兼容性通过。
- Expert realtime stability：face track、landmark EMA、YuNet background rejection、
  probability state machine 通过。
- Expert offline E2E：FER zip -> Haar/LBF -> 136 keypoint vector ->
  174-D Geometry-SVM prediction 通过。
- Expert training smoke：PCA、模型比较、选模、held-out test、
  confusion/failure 输出全部通过。
- Bonus：3 个 temporal-alignment tests 通过。
- Mario：28 个 unit tests 通过；`--check` 实际检测到 1 人和 17/17 keypoints。
- Just Dance：`--check` 确认共享 YOLO、reference video、对齐窗口和评分权重。
- 所有 Python 文件通过 `compileall`。

无法在当前自动化环境验证的部分只有真实 webcam/GUI 人机体验。答辩前应在演示
电脑上逐一打开 Beginner、Expert、Just Dance、Mario/3D runner，并录制一段
暗光、转头、遮挡、多人和动作延迟的演示证据。

## 10. 后续开发规则

1. 后续只从 `main` 开新分支，不在三个原作者分支上继续堆修改。
2. Expert 新模型必须声明 `feature_version`，并证明输入没有图像像素。
3. PCA 必须放在 sklearn `Pipeline` 内，确保每个 fold 只在训练部分拟合。
4. test split 不参与调参；正式报告区分 CV 与 final test。
5. 不以 Accuracy 作为唯一结论。至少同时报告 Macro-F1、per-class recall、
   confusion、failure cases、折间方差和单样本延迟。
6. webcam 稳定性属于问题本身：记录检测失败、identity switch、关键点抖动和
   label flicker，不要用静态图片的高分掩盖实时失败。
7. Mario 和 3D runner 是创意扩展，不能替代 Bonus Task 1/2 的双面板舞蹈评分。
