# Visual Computing Project PPT Guidelines

> 主题：Real-time Video Analysis and Rendering  
> 依据：`Visual_Computing_Project.pdf`  
> 建议主线：**问题观察 -> 失败原因 -> 可验证改动 -> 对照结果 -> 最终整合**

这份 PPT 不应只是三个 demo 的功能清单。最有说服力的叙事是：我们先完成课程
baseline，然后用真实失败现象推动迭代；每次优化都说明解决了什么、引入了什么
新代价、如何验证，最后形成一个从 facial landmarks、expression classification
到 body movement scoring 的统一实时系统。

## 一、先统一汇报口径

### 1. PDF 的硬要求与项目证据

| Level | PDF 要求 | 本项目对应证据 |
|---|---|---|
| Beginner | 本地 webcam、Haar 人脸框、LBF 68 点、实时显示 | `beginner_level/main.py` |
| Beginner | 光照、头部角度、遮挡鲁棒性与替代 detector | `docs/robustness_test.md`，Expert 中的 YuNet 对照 |
| Expert | 只用 keypoints，不用 full image 分类 | 136 维眼对齐坐标 + 38 维 landmark geometry |
| Expert | test evaluation、confusion matrix、failure cases | `results/zhangyx_part_two/` |
| Expert | 每次预测小于约 30 ms | Geometry-SVM 历史单样本 18.08 ms |
| Expert | webcam prediction + expression effects | `expert_level/main.py demo --mirror` |
| Bonus | reference 与 webcam 双 panel、双方 skeleton | `bonus_level/main.py` |
| Bonus | 空间/时间对齐、相似度、屏幕 feedback | `bonus_level/docs/SCORING_INTEGRATION.md` |
| Bonus | 解释 scoring techniques and metrics | 本指南第 14-16 页建议 |

### 2. 哪些数字可以直接讲

- GitHub Part Two 正式历史结果可以讲，但要称为“Part Two historical full-run
  results”或“已集成实验结果”，不要说是本次合并重新训练。
- FER 数据完整性已确认：train 28,709，test 7,178，七类齐全。
- 默认 Geometry-SVM：Accuracy 0.4731、Macro-F1 0.4633、单样本分类
  18.08 ms。
- 自动测试已验证评分公式的确定性边界，但不能代替真实 webcam FPS、真实延迟
  和实际鲁棒性测试。
- Beginner 的 robustness table 目前是待填写模板。没有实测数字时只能讲定性
  观察，不能虚构成功率或 FPS。
- PDF 中 cross-validation 是 optional；本项目合并规范更严格，正式统一训练应在
  train split 内做 Stratified K-fold，并把 PCA 作为候选，test 只评估一次。

## 二、建议的 16 页主 Deck + 2 页可选扩展

Slide 17 可作为评分验证 appendix；Slide 18 可与结论页合并。如果时间只有
10 分钟，可把第 2/3 页、第 6/7 页、第 15/16 页分别合并。

### Slide 1 - Title + 10 秒 Demo Hook

标题建议：

> From Facial Landmarks to Motion-Aware Dance Scoring

画面放三张并列截图：

1. Beginner 的 face box + 68 points；
2. Expert 的 emotion + effect；
3. Bonus 的 reference/webcam skeleton + score。

开场话术：

> “Our project is one continuous keypoint pipeline. We first learned where
> keypoints fail, then used facial geometry for real-time expression
> classification, and finally extended the same ideas of normalization and
> temporal stability to full-body dance scoring.”

不要在第一页讲模型细节，先让老师知道最终系统可以现场运行。

### Slide 2 - Assignment Requirements -> Our System

用三行 flow 表达：

```text
Beginner: webcam -> Haar face -> LBF 68 landmarks
Expert: landmarks -> normalized geometry -> expression -> effect
Bonus: YOLO pose -> spatial/temporal alignment -> grade + feedback
```

右侧写三个贯穿项目的研究问题：

- Detection 怎样在光照、角度、遮挡下保持稳定？
- Keypoints 丢失纹理后，分类器之间为什么会产生差异？
- Dance score 怎样既容忍自然反应延迟，又防止“站着不动也得高分”？

### Slide 3 - Integration Decisions and Conflict Resolution

这一页说明你们不是机械拼接代码：

| 冲突 | 选择 | 原因 |
|---|---|---|
| Beginner 本地模块版 vs GitHub Part One | 以本地 Beginner 为主 | ROI continuity、cache、平滑和异常小框抑制更完整 |
| Expert 本地旧 SVM vs GitHub Part Two | 默认用 Part Two Geometry-SVM | Macro-F1 更高，仍为 keypoint-only，18.08 ms |
| Bonus 本地模块 GUI vs GitHub Part Three 单文件 | GUI 保留本地结构，评分采用 Part Three | 兼顾工程可维护性与更合理的运动/延迟评分 |
| PCA 降维 vs 完整几何 | PCA 保留为候选而非默认 | 速度更快，但历史 Macro-F1 明显下降 |

一句总结：

> “We chose algorithms by task compliance, Macro-F1, stability and latency,
> not by whichever branch had the highest raw accuracy.”

## 三、Beginner：讲清楚问题如何推动迭代

### Slide 4 - Baseline: Haar + LBF

展示最初流程：

```text
frame -> grayscale -> Haar candidate box -> LBF fit -> 68 landmarks
```

解释依赖关系：

- Haar 失败：没有 face box，LBF 无法工作；
- box 偏移：LBF 下颌、嘴和眼部点一起漂移；
- 单帧检测：框和关键点会闪烁。

这页必须明确保留课程指定的
`cv2.CascadeClassifier`、`cv2.face.createFacemarkLBF()` 和 68 点输出。

### Slide 5 - What Failed in Development

建议用“现象 - 原因 - 证据截图”三列：

| 现象 | 可能原因 | 应保存的证据 |
|---|---|---|
| 暗光漏检 | Haar contrast 不足 | raw 与 CLAHE 同位置截图 |
| 侧脸/抬头漏检 | frontal Haar 的姿态先验 | yaw 较大时 DETECTED/LOST 对比 |
| 遮嘴后 landmarks 漂移 | LBF 局部特征被遮挡 | cover-mouth 截图 |
| 框一帧出现一帧消失 | detector 是逐帧判断 | DETECTED/CACHED/LOST 连续截图 |
| 转头时出现鼻子大小的小框 | Haar 把局部结构当成人脸 | 96x96 异常框截图 |
| 强增强后反而变糊 | gamma/CLAHE 放大噪声与反光 | 开/关 enhancement 对比 |

不要说“CLAHE always improves detection”。项目开发实际发现，过强增强会让正常
场景更不稳定，这正是很好的 exploration evidence。

### Slide 6 - Iteration Timeline

建议画成六步：

```text
Baseline Haar + LBF
  -> CLAHE / gamma experiments
  -> previous-box ROI + periodic full scan
  -> short cache + IoU de-duplication
  -> LBF geometry sanity check
  -> relax over-strict rejection + sudden-shrink rejection
```

每一步讲一个 trade-off：

- CLAHE：暗部局部对比提高，但可能放大噪声；
- ROI/cache：减少闪烁，但短时间可能显示过期框；
- geometry sanity check：过滤假脸，但过严会误杀侧脸；
- landmark EMA：稳定，但会产生轻微视觉延迟；
- sudden-shrink rejection：阻止局部小框抢占 track，但阈值过高可能忽略真实远脸。

最重要的开发洞察：

> “鲁棒性不只是换 detector；video continuity 本身也是信息。”

### Slide 7 - Robustness Comparison and Alternative Detector

同一个人、同一位置、每个场景约 20 秒，记录：

- detection success rate；
- DETECTED / CACHED / LOST 数量；
- average FPS；
- landmark jitter/drift；
- false positive；
- raw vs CLAHE；
- Haar vs YuNet。

使用现有模板：

`VisualComputingProject/beginner_level/docs/robustness_test.md`

至少展示正常光、暗光、左右转头、遮嘴四个场景。替代 detector 可使用 Expert
实时入口中的 YuNet，比较时说明：

- YuNet 更适合多姿态和复杂背景；
- Haar 更轻、更符合 Beginner 指定 baseline；
- detector 改善不能修复 LBF 自身在严重遮挡下的拟合问题。

如果答辩前没有定量完成，表格写“qualitative observation”，不要填写猜测百分比。

## 四、Expert：模型差异、失败案例、最终整合

### Slide 8 - Why Keypoint-only and How Features Are Built

左侧放 68 点图，右侧放特征流程：

```text
68x2 LBF coordinates
  -> eye midpoint to origin
  -> rotate eye line to horizontal
  -> divide by inter-eye distance
  -> 136 normalized coordinates
  -> +38 landmark-derived geometry
  -> 174-D Geometry-SVM
```

38 个 geometry 分成 brow、eyes、nose、mouth、global 五组。强调它们仍然只由
keypoints 计算，没有眼睛 ROI、嘴部像素或完整人脸纹理。

解释预处理目的：

- translation invariance；
- in-plane rotation invariance；
- scale invariance；
- 显式表达 mouth opening、brow-eye distance、eye aspect ratio 等关系。

### Slide 9 - Fair Model Comparison

主表建议直接使用：

| Model | Feature | Accuracy | Macro-F1 | Single prediction |
|---|---|---:|---:|---:|
| HGB | 136 coords | 0.4313 | 0.3914 | 23.41 ms |
| Base RBF-SVM | 136 coords | 0.4631 | 0.4347 | 5.17 ms |
| PCA95 + SVM | 12 PCs | 0.4032 | 0.3613 | **2.68 ms** |
| Tuned SVM | 136 coords | 0.4684 | 0.4589 | 5.31 ms |
| Geometry-SVM | 136 + 38 | 0.4731 | **0.4633** | 18.08 ms |
| `drop_eyes` | 136 + 28 | **0.4737** | 0.4625 | 4.90 ms |

讲解顺序：

1. HGB 能学非线性，但这个特征空间中不如 RBF-SVM；
2. 基础 SVM 已显著提高 Macro-F1；
3. PCA 最快，但无监督方差不等于分类信息；
4. 调参主要改善类别平衡；
5. 几何特征进一步提高 Macro-F1；
6. `drop_eyes` Accuracy 略高但 Macro-F1 略低，因此不取代默认模型。

这里一定要说“Accuracy 不是唯一选择指标”。

### Slide 10 - Why PCA and Geometry Behave Differently

建议左右对照：

**PCA**

- 优点：12 PCs，2.68 ms，训练更快；
- 问题：保留 95% variance 不代表保留 95% discriminative information；
- 结果：Macro-F1 从基础 SVM 的 0.4347 降到 0.3613。

**Explicit Geometry**

- 优点：直接表达眉、眼、嘴的相对关系；
- 问题：部分比率会放大 LBF jitter，且特征之间可能冗余；
- 结果：修复 458 个 tuned-coordinate SVM 的错误，也引入 424 个新错误。

配图：

- `03_pca_information_loss.png`
- `02_geometry_rescues.png`
- `04_geometry_regressions.png`

结论不能写成“geometry always better”；正确结论是平均表现更好，但存在样本级
trade-off。

### Slide 11 - Confusion Matrix and Per-class Difference

主图：

`geometry_svm_confusion_matrix.png`

旁边放 `06_per_class_f1_comparison.png`。

讲解重点：

- happy、surprise 的嘴形和开口几何更明显；
- fear、sad、neutral 在 landmark space 中重叠更大；
- disgust 样本少，指标波动更大；
- keypoint-only 缺少 wrinkles、cheek tension、skin texture 等信息。

不要只朗读对角线。选择一到两个最明显的 confusion pair，解释它为什么发生。

### Slide 12 - One Failure Case Deep Dive

推荐用 `05_all_models_wrong.png` 中的 #1047：

```text
true label: disgust
HGB / SVM / PCA-SVM: neutral
Tuned-SVM / Geometry-SVM: angry
```

分析分四层：

1. 输入质量：48x48，眼镜、侧向光照和低对比；
2. landmark fit：68 点可以拟合，但无法表达鼻皱、皮肤纹理；
3. representation limit：disgust 与 angry/neutral 的几何可能接近；
4. data uncertainty：FER label 也可能有主观性，不应把每个错误都归因于 classifier。

提出的改进必须仍尊重任务：

- 更严格的 landmark fit quality filtering；
- temporal expression evidence；
- class-balanced calibration；
- K-fold 检查改进是否稳定；
- 不要偷偷改成 full-image CNN 作为主模型。

也可把 #6568 放 appendix，展示坏图/非人脸如何暴露 dataset quality 问题。

### Slide 13 - Final Expert Integration

讲清楚“选择模型”和“工程整合”是两件事：

```text
GitHub Geometry-SVM
  + local YuNet/Haar switch
  + LBF landmark extraction
  + multi-face tracking
  + landmark/probability EMA
  + confidence and label-switch hysteresis
  + expression-driven effects
```

为什么最终 demo 用 Geometry-SVM：

- keypoint-only，符合题目；
- 历史 Macro-F1 最高；
- 18.08 ms 小于 30 ms；
- 能直接嵌入现有实时稳定化和特效 pipeline。

实验协议改进：

- 统一 `train` 在 official train 内做 Stratified K-fold；
- PCA 在每个 fold 内拟合，防止 leakage；
- 按 Macro-F1 mean、fold std、balanced accuracy 和 latency 选模；
- official test 只在选模后评估一次。

若还没有跑完整统一 K-fold，这页必须标注：

> “Pipeline verified by smoke tests; full K-fold result pending.”

不能把历史 holdout 结果改称为 K-fold 结果。

## 五、Bonus：评分如何构成，以及 delay 如何分析

### Slide 14 - Pose Pipeline and Detection Challenges

流程：

```text
reference + webcam
  -> YOLOv8n Pose (17 points)
  -> main dancer tracking
  -> confidence mask + EMA
  -> 12 body joints
  -> spatial normalization
  -> delay search + motion comparison
```

必须讨论的 detection challenges：

- 多人：面积、画面中心、上一帧连续性和 confidence 综合选择主舞者；
- 遮挡/出画：只比较双方共同可见关节；
- 快动作：YOLO keypoint jitter 与 motion blur；
- partial body：coverage penalty，至少四个共同身体点才评分；
- mirror：比较 direct 与 anatomically mirrored candidate。

### Slide 15 - How the Grade Is Constructed

建议把公式分三层动画展示。

第一层，单帧 pose：

\[
S_{pose}=(0.35S_{position}+0.65S_{angle})(0.60+0.40C)
\]

- position：归一化关节欧氏距离的 Gaussian similarity；
- angle：八个肘/肩/髋/膝角度差；
- wrist/ankle 权重 1.35；
- \(C=N_{common}/12\) 是 coverage。

第二层，0.4 秒 motion：

\[
S_{motion}=0.70S_{vector}+0.30S_{activity}
\]

第三层，最终相似度：

\[
S=(0.55S_{pose}+0.45S_{motion})\times antiStatic
\]

Grade：

| Similarity | Feedback | Base points |
|---:|---|---:|
| `>=85` | Perfect | 1000 |
| `>=70` | Super | 700 |
| `>=55` | Good | 400 |
| `<55` | Miss | 0 |

每秒最多一次正式计分，避免高 FPS 设备占优势。Combo bonus 有上限。`Sync` 和
真正的 `Hold` 只显示、不计分；参考在动而玩家不动时显示 `Move!`。

“最合理”不是说权重绝对正确，而是说明每个设计目标：

- angle 对身材和摄像头距离更稳定，所以占 pose 的 65%；
- position 保留手脚落点信息；
- motion 防止静态姿势作弊；
- coverage 防止少数可见点偶然匹配拿满分；
- capped combo 防止连击完全淹没动作质量。

### Slide 16 - Delay: Search, Risk, and Analysis

当前玩家时间 \(t\) 只搜索：

\[
t-0.8 \le t_{reference}\le t
\]

并报告：

\[
lag=t_{player}-t_{reference}\ge0
\]

必须强调“不使用未来参考帧”。这比对称搜索更符合 reaction delay 的定义。

为什么不能无限扩大窗口：

- 窗口越大，越容易从旧动作中“挑中”相似姿势，虚高分数；
- 窗口太小，又会惩罚正常人的反应时间；
- motion window 和 anti-static factor 用来减少只挑到相似静态姿势的问题。

建议答辩前做 delay sweep：

| 人为延迟 | Window | Median matched lag | P90 lag | Average score | Window-edge rate |
|---:|---:|---:|---:|---:|---:|
| 0.0 s | 0.8 s |  |  |  |  |
| 0.3 s | 0.8 s |  |  |  |  |
| 0.6 s | 0.8 s |  |  |  |  |
| 1.0 s | 0.8 s |  |  |  |  |

分析口径：

- median：典型反应时间；
- P90：较慢动作的尾部延迟；
- window-edge rate：多少 match 卡在 0.8 s 边缘；
- 1.0 s 人为延迟应明显掉分，否则窗口/相似度过宽；
- 如果所有人的 lag 都贴近 0.8 s，应检查播放线程和时间戳，而不是直接加大窗口。

### Slide 17 - Validation and Ablation for the Scoring System

若时间允许，作为 Bonus 的额外一页；否则放 appendix。

至少比较：

| Test | Expected observation |
|---|---|
| identical pose + motion | 接近 Perfect |
| same pose, player static | reference moving 时低于 Good，显示 Move! |
| reference hold | Hold，不计 Miss |
| translated/scaled skeleton | 分数应基本不变 |
| wrong arm | position/angle/motion 都应下降 |
| 0.4 s delayed correct move | lag 约 0.4 s，仍可高分 |
| future-only matching case | 不得选择未来 reference |
| partial occlusion | coverage 下降，仍可在足够点时评分 |

自动测试位于：

`VisualComputingProject/bonus_level/tests/test_temporal_alignment.py`

真实视频测试还要报告 YOLO inference time，因为 scoring 很快不代表整条 pipeline
能达到 30 FPS。CPU 上可采用参考视频离线 pose cache 或降低 inference resolution
作为进一步优化。

### Slide 18 - Final Demo and Takeaways

如果保留 16 页，把这页与 Slide 17 合并。

现场顺序：

1. Beginner：正脸 -> 左右转 -> 遮嘴，展示 DETECTED/CACHED/LOST；
2. Expert：neutral -> happy -> surprise，展示 label 稳定化和 effect；
3. Bonus：正常跟跳 -> 故意慢约 0.4 s -> 完全静止，展示 lag、score 与 Move!。

最后三句 takeaways：

- Detection failure often propagates into every downstream task.
- Keypoint geometry improves invariance and interpretability, but cannot recover
  texture that was never represented.
- A fair interactive score needs spatial normalization, temporal tolerance and
  anti-cheating motion evidence together.

## 六、PPT 可直接使用的本地素材

Expert：

- `VisualComputingProject/expert_level/results/zhangyx_part_two/06_per_class_f1_comparison.png`
- `VisualComputingProject/expert_level/results/zhangyx_part_two/geometry_svm_confusion_matrix.png`
- `VisualComputingProject/expert_level/results/zhangyx_part_two/geometry_svm_failure_cases.png`
- `VisualComputingProject/expert_level/results/zhangyx_part_two/02_geometry_rescues.png`
- `VisualComputingProject/expert_level/results/zhangyx_part_two/03_pca_information_loss.png`
- `VisualComputingProject/expert_level/results/zhangyx_part_two/04_geometry_regressions.png`
- `VisualComputingProject/expert_level/results/zhangyx_part_two/05_all_models_wrong.png`
- `VisualComputingProject/expert_level/results/zhangyx_part_two/group_ablation_validation.png`

Beginner 需要你本地补采：

- normal light；
- dark light raw/CLAHE；
- left/right head turn；
- mouth/eye occlusion；
- sudden small false box before/after；
- Haar/YuNet comparison。

Bonus 需要你本地补采：

- 双 panel skeleton；
- Perfect/Super/Good/Miss；
- Sync/Hold/Move!；
- 状态栏中的 positive lag；
- 结算页的 average/median/P90 delay。

## 七、答辩前执行清单

### 环境与静态检查

```powershell
conda activate vc_sws3026
python VisualComputingProject\expert_level\main.py inspect
python VisualComputingProject\bonus_level\main.py --check
python -m unittest discover -s VisualComputingProject\expert_level\tests -v
python VisualComputingProject\expert_level\tests\test_keypoint_features.py
python VisualComputingProject\expert_level\tests\test_realtime_stability.py
python -m unittest discover -s VisualComputingProject\bonus_level\tests -v
```

### 三个现场入口

```powershell
.\run_beginner_level.ps1
.\run_expert_level.ps1 demo --mirror
.\run_bonus_level.ps1
```

### 正式 Expert K-fold（耗时，答辩前单独安排）

```powershell
.\run_expert_level.ps1 train --cv-folds 5 --workers 4
```

先用小样本确认流程：

```powershell
.\run_expert_level.ps1 train `
  --max-train-per-class 100 `
  --max-test-per-class 50 `
  --cv-folds 3
```

注意：正式重训会覆盖当前默认模型。先备份
`models/keypoint/current/expression_classifier.joblib`，并把正式输出与历史 Part
Two 结果分开保存。

## 八、常见答辩追问

### 为什么不直接使用 full-image CNN？

题目明确要求以 facial keypoints 分类。ROI-CNN 即使分数更高，也改变了任务定义；
它可以作为 appearance upper-bound 或反例，但不能作为默认 Expert 解法。

### 为什么 Geometry-SVM 不是 Accuracy 最高的 `drop_eyes`？

项目以 Macro-F1 和类别平衡为主。`drop_eyes` Accuracy 0.4737 略高，但
Macro-F1 0.4625 略低于完整 Geometry-SVM 的 0.4633。差异很小，正确结论是两者
存在速度/类别平衡 trade-off，不是眼部特征无用。

### 为什么 PCA 变差？

PCA 保留的是总体方差，不知道哪些低方差方向对表情分类重要。历史 PCA95 只保留
12 个主成分，速度提升，但 Macro-F1 降到 0.3613。

### 为什么 dance score 不只比较当前 frame？

单帧无法区分“跟着动作移动”和“碰巧摆出相似姿势”。加入 0.4 秒 motion 后，
可以比较关节位移方向和活动幅度，并惩罚 reference 在动、player 静止的情况。

### 0.8 秒 delay window 会不会让人作弊？

会有 candidate cherry-picking 风险，因此只允许搜索过去 reference、不看未来，
限制窗口为 0.8 秒，并把 motion similarity 和 anti-static factor 放进最终分。
还应通过 median/P90 和 window-edge rate 检查窗口是否过宽。

### 为什么没有声称 30 FPS？

Expert 的 18.08 ms 是 classifier-only 历史单样本延迟，不包括 detector、LBF、
tracking 和 rendering。Bonus 的 YOLO pose 更可能成为瓶颈。最终只能根据本机
实测 end-to-end FPS 声称实时性能。
