# Visual Computing Final Presentation Guidelines

本文件依据以下资料重构：

- `project-eval-2.pdf`：最终 Presentation 的官方评价标准；
- `Visual_Computing_Project.pdf`：Beginner、Expert、Bonus 的项目任务说明；
- 当前仓库中的真实代码、模型、实验指标、混淆矩阵、失败案例和测试结果。

本指南只指导 PPT 制作，不修改现有 PowerPoint。

---

## 1. 官方规则优先级

### 1.1 Stage 1

- 形式：使用 slides 进行口头汇报；
- 时间：**10 分钟 presentation + 3 分钟 Q&A**；
- **严格禁止现场运行 demo 或程序**；
- 只能展示程序截图或预先录制的视频；
- Instructor 只根据 Stage 1 的汇报内容评分。

因此，Stage 1 中不要：

- 打开终端；
- 连接 webcam 后现场运行程序；
- 切换到 Python、VS Code 或文件管理器；
- 依赖网络链接播放视频；
- 把“现场 demo”安排进 10 分钟讲稿。

所有程序效果都应该提前录制并嵌入 PPT，同时准备静态截图作为视频播放失败时的备份。

### 1.2 Stage 2

- 所有组在 Stage 1 结束后展示 Bonus Dance Analysis；
- Stage 2 是同学投票的 popularity contest；
- Stage 2 不计入 Instructor 给出的课程项目分数。

Stage 2 仍应准备完整可运行的 Dance demo，但不要占用 Stage 1 的口头汇报时间进行
现场操作。

---

## 2. 汇报的 communication job

> By the end of the 10-minute presentation, the instructors should understand
> how each observed failure led to a measurable pipeline improvement, and how
> the final system satisfies the official Beginner, Expert and Bonus criteria.
>
> 10 分钟结束时，老师应当清楚看到：每一个失败现象如何推动了一次可验证的改进，
> 以及最终系统如何覆盖 Beginner、Expert 和 Bonus 的官方评分要求。

整场汇报使用一条连续的 exploration narrative：

```text
Observed failure
    -> Cause analysis
    -> Technical change
    -> Before/after evidence
    -> Remaining limitation
```

不要把汇报讲成：

```text
We used Haar.
We used SVM.
We used YOLO.
```

应该讲成：

```text
Haar failed during head rotation.
We found that unstable and shrinking face boxes were the immediate cause.
We added ROI search, candidate rejection and short-term caching.
The improved recording shows a more continuous box, although large side poses
remain difficult.
```

---

## 3. 官方 16 项标准与页面对应

| Official item | 官方要求摘要 | 主讲页面 | 当前证据状态 |
|---:|---|---:|---|
| 1 | 展示 Haar detector failure cases | 2 | 需要补录真实失败截图/视频 |
| 2 | 解释 Task 3 如何改进 detector | 3 | 代码和双语报告已存在 |
| 3 | 展示 improved robust detector | 3 | 需要补录 before/after |
| 4 | 解释表情分类 preprocessing、features、classifier | 4–10 | 已有代码与真实指标 |
| 5 | 展示改进前 confusion matrix 和 failure images | 11 | 已有 HGB matrix/failure cases |
| 6 | 解释如何改进，并展示改进后 matrix | 10–11 | 已有 Geometry-SVM matrix |
| 7 | 展示 Expert Task 2 程序 | 12–13 | 有效果预览；需要补录真实 webcam |
| 8 | 说明 additional expressions 及训练方法 | 4 | 当前七类中将 `neutral` 作为额外类别 |
| 9 | 选择两个 video effects，解释 detection 到 rendering | 12 | 代码完整；选择 Happy 与 Sad |
| 10 | 展示 flicker/jerky result、原因和修复 | 13 | smoothing 已实现；需要补录 A/B |
| 11 | 展示 Bonus Task 1 程序 | 14 | 已有 contact sheet 和 annotated MP4 |
| 12 | 解释 keypoint detection/rendering challenges 和解决方法 | 14 | 已有报告和片段分析 |
| 13 | 展示 Dance program | 15 | 需要补录双面板 webcam 视频 |
| 14 | 解释评分、动作对齐、distance/similarity | 15–16 | 代码、报告和测试已存在 |
| 15 | 解释 numeric/text score、时间变化、overall score | 17 | 代码已存在；需要录制完整回合 |
| 16 | 展示 dancing 之外的动作、对象和评分 | 18 | Mario 已实现；需要补录实际画面 |

### 3.1 关于官方“6 classes”和当前“7 classes”

官方评价文件要求解释六类基础表情，并询问是否检测 additional expressions。

当前项目实际训练七类：

```text
angry, disgust, fear, happy, sad, surprise + neutral
```

最安全、最清楚的表述是：

> We trained the six basic emotion classes required by the task and added
> neutral as a seventh FER category.
>
> 我们训练了任务要求的六类基础表情，并将 neutral 作为 FER 数据中的第七个额外类别。

`neutral` 不是后处理规则，而是与其他六类一起：

1. 使用官方 train/test 文件夹；
2. 运行相同的 LBF landmark extraction；
3. 使用相同的归一化和 geometry features；
4. 进入同一个 class-balanced classifier 进行训练。

不要说我们额外训练了 tongue detector；当前没有对应数据和模型。

---

## 4. 推荐主讲结构：18 页，目标讲述时间约 9 分钟

18 页与现有 PPT 页数接近，同时能够逐项覆盖官方标准。

| Slide | Section | Narrative claim | 建议时间 |
|---:|---|---|---:|
| 1 | Opening | One pipeline evolved from facial points to interactive body movement | 10 s |
| 2 | Beginner | Haar fails predictably under pose, lighting and occlusion | 25 s |
| 3 | Beginner | Pipeline-level stabilisation makes Haar + LBF more robust | 30 s |
| 4 | Expert | Seven-class recognition uses landmarks only | 30 s |
| 5 | Expert model | HGB captured nonlinearity but fragmented smooth geometry | 18 s |
| 6 | Expert model | RBF-SVM was a better fit for continuous facial shape | 18 s |
| 7 | Expert model | PCA gained speed but removed discriminative detail | 22 s |
| 8 | Expert model | Validation-based tuning improved class balance | 22 s |
| 9 | Expert model | Explicit geometry produced the best overall Macro-F1 | 25 s |
| 10 | Expert decision | Model comparison and ablation justify the final choice | 35 s |
| 11 | Expert failure | The final model improves the matrix but does not solve ambiguity | 40 s |
| 12 | Expert effects | Two effects map class predictions to procedural rendering | 30 s |
| 13 | Expert stability | Temporal smoothing reduces label flicker and jerky effects | 30 s |
| 14 | Bonus Task 1 | Tracking, confidence and caching stabilise body keypoints | 35 s |
| 15 | Bonus scoring | The score evolved after pose-only scoring rewarded static players | 45 s |
| 16 | Bonus alignment | Fair comparison requires spatial, motion and delay alignment | 50 s |
| 17 | Bonus feedback | Continuous similarity becomes grades, points and an overall result | 40 s |
| 18 | Extension/close | The same pose interface controls and scores a platform game | 25 s |

总计约 8 分 50 秒，预留约 70 秒处理视频播放、换人和临场停顿。

时间只放在 Speaker Notes 或排练表中，不要显示在观众看到的页面上。

---

# Part A — Beginner Level

## Slide 1 — Keypoints Became an Interaction Interface / 关键点成为交互界面

**Official criteria:** opening only

### English on-slide copy

**Real-time Video Analysis and Rendering**

`Facial landmarks -> expression effects -> dance scoring -> pose-controlled game`

### 中文对应讲稿

> 我们从人脸关键点实时检测开始，随后只使用 landmarks 进行表情分类，再将同样的
> keypoint 思路扩展到全身动作评分和姿态控制游戏。汇报重点是每次失败如何推动
> pipeline 迭代。

### Visual / Evidence

使用一张四阶段横向 montage：

1. 68 facial landmarks；
2. Expert expression effect；
3. Dance 双面板；
4. Mario pose control。

标题页保持简单，不放成员分工、模型指标或完整目录。

---

## Slide 2 — Haar Fails Before LBF Can Fit the Face / Haar 失败会直接阻断 LBF

**Official criteria:** 1

### English on-slide copy

**Haar failure cases**

- Large head rotation -> missed or shrinking face box
- Backlight / low contrast -> unstable detection
- Partial occlusion -> incorrect ROI or lost face
- Internal facial features -> occasional small false positives

**Key observation**

> LBF cannot recover when the upstream face ROI is missing or wrong.

### 中文对应讲稿

> Haar 决定 LBF 在哪里拟合 68 个点。当 Haar 在转头时漏检，或把鼻子、嘴巴等
> 局部区域误当成小脸时，后续 landmarks 也会消失或漂移。因此第一步不是直接
> 更换分类器，而是先找出 detector failure。

### Visual / Evidence

必须放真实截图或 6–8 秒预录视频，至少包含：

1. 正常正脸基线；
2. 右转头时 face box 缩小或丢失；
3. 暗光/背光；
4. 手遮嘴或遮眼。

在每张失败图上只标一项：

`MISS`、`SHRINKING BOX`、`FALSE POSITIVE` 或 `LANDMARK DRIFT`。

### 素材状态

`beginner_level/outputs/` 当前没有可直接使用的真实结果文件，需要补录。

---

## Slide 3 — Robustness Improved Without Replacing Haar / 不更换 Haar 也能提高连续性

**Official criteria:** 2, 3

### English on-slide copy

**Task 3 improvements**

1. Previous-face ROI search with periodic full-frame correction
2. Candidate filtering and sudden-shrink rejection
3. LBF geometry validation before accepting a face
4. Optional CLAHE/gamma preprocessing
5. Short-term box cache and region-aware landmark smoothing

**Result**

> Fewer disappearing boxes and less landmark flicker during short pose changes.

**Remaining limit**

> Large side poses and severe occlusion still exceed the Haar + LBF model range.

### 中文对应讲稿

> 我们没有把未完成的 MediaPipe 或 Dlib 写成已实现方案。当前改进仍然以 Haar +
> LBF 为核心，但改变了运行方式：以上一帧稳定人脸框进行局部搜索，并定期全图
> 校正；拒绝突然缩小且落在人脸内部的小框；只有 LBF 点分布合理时才接受；短时
> 漏检时使用缓存；最后对不同面部区域做时间平滑。

### Visual / Evidence

采用左右 before/after：

- 左：baseline Haar failure；
- 右：相同动作下 improved detector；
- 下方放一条 4–6 秒预录片段。

不要只放代码截图。官方明确要求展示 improved detector 的程序效果。

---

# Part B — Expert Level

## Slide 4 — We Classify Seven Expressions Using Landmarks Only / 七类表情只使用关键点

**Official criteria:** 4, 8

### English on-slide copy

**Six basic emotions + one additional FER category**

`angry · disgust · fear · happy · sad · surprise + neutral`

**Landmark-only pipeline**

```text
FER 48×48 image
 -> resize to 192×192
 -> centred FER face ROI
 -> LBF 68 landmarks
 -> eye-centred translation
 -> eye-line rotation
 -> inter-eye scaling
 -> 136 coordinates
 -> optional 38 geometry descriptors
 -> classifier
```

- Train: 28,709
- Test: 7,178
- Primary selection metric: Macro-F1

### 中文对应讲稿

> 官方要求六类基础表情。我们使用 FER 的七类标签，并将 neutral 作为第七个额外
> 类别。Neutral 使用同一份 train/test split、同样的 landmark extraction 和同一个
> classifier 训练，并不是规则判断。所有分类输入都来自关键点，完整人脸图像只用于
> LBF 提取和展示。

### Visual / Evidence

页面主体使用一条 pipeline。右上角放一张：

`原始 FER 图 -> 68 点 -> eye-aligned 点图`

页面脚注：

`Single fixed stratified 80/20 validation split inside the official train set; no K-fold.`

---

## Slide 5 — HGB Fragmented a Smooth Shape Space / HGB 难以表达平滑的人脸形状空间

**Official criteria:** 4, supports 5–6

### English on-slide copy

**Histogram Gradient Boosting**

- Input: 136 normalised coordinates
- Accuracy: 43.13%
- Macro-F1: 39.14%
- Prediction: 23.41 ms

**Strength**

Captures nonlinear coordinate interactions with boosted trees.

**Limitation**

Axis-aligned tree splits fragment the continuous facial-shape manifold.

### 中文对应讲稿

> HGB 可以学习非线性关系，但树模型依靠分段、轴对齐的切分。我们的输入是连续变化
> 的人脸几何形状，因此这种边界不如平滑的 kernel boundary 合适。它成为后续改进前
> 的 baseline。

### Visual / Evidence

只使用一个简化的 tree boundary 示意图和四个数字。此页控制在约 18 秒。

---

## Slide 6 — RBF-SVM Better Matches Continuous Facial Geometry / RBF-SVM 更适合连续几何

**Official criteria:** 4, supports 6

### English on-slide copy

**Coordinate RBF-SVM**

- Input: 136 normalised coordinates
- Accuracy: 46.31%
- Macro-F1: 43.47%
- Prediction: 5.17 ms

**Why it improved**

- Smooth nonlinear boundary
- Class-balanced training
- Better accuracy, Macro-F1 and latency than HGB

**Remaining problem**

Raw coordinates do not explicitly encode openings, ratios or symmetry.

### 中文对应讲稿

> RBF kernel 通过局部相似性形成平滑的非线性边界，更适合连续的人脸形状空间。
> Class balancing 也降低了 happy 等大类对模型的支配。但是 136 个坐标没有直接
> 表达嘴巴开合、眉眼距离和左右对称性。

---

## Slide 7 — PCA Preserved Variance, Not Expression Evidence / PCA 保留方差却丢失判别信息

**Official criteria:** 4, 6

### English on-slide copy

**PCA95 + RBF-SVM**

- 136 coordinates -> 12 principal components
- Variance retained: 95.265%
- Accuracy: 40.32%
- Macro-F1: 36.13%
- Prediction: 2.68 ms

**Decision**

> Faster inference did not justify the loss of low-variance facial detail.

### 中文对应讲稿

> 我们原本假设 PCA 可以去除相关性和噪声。虽然 12 个主成分保留了 95.265% 总方差，
> 但 Macro-F1 明显下降。原因是 PCA 优先保存总体变化，而眉毛、眼睑、嘴角等对表情
> 有用的细节可能位于低方差方向。

### Visual / Evidence

使用：

`expert_level/level2_report_assets/03_pca_information_loss.png`

页面核心句：

> 95% variance retained did not mean 95% expression information retained.

---

## Slide 8 — Validation Tuning Improved Class Balance / 调参主要改善类别平衡

**Official criteria:** 4, 6

### English on-slide copy

**Tuned coordinate SVM**

- `C = 10`
- `gamma = 2/136 = 0.0147059`
- Accuracy: 46.84%
- Macro-F1: 45.89%
- Prediction: 5.31 ms

**Selection protocol**

- One stratified 80/20 split inside train
- Parameters selected by validation Macro-F1
- Official test evaluated after selection

### 中文对应讲稿

> 我们没有直接在 test set 上找参数。在官方 train 内固定划分 22,967 个 fit samples
> 和 5,742 个 validation samples，以 Macro-F1 选择 C 和 gamma。调参后的主要收益
> 是少数类别和整体类别平衡，而不是只提高 Accuracy。

### Visual / Evidence

使用：

`expert_level/artifacts_svm_tuned/tuning_heatmap.png`

只标出选中的点，不逐格解释 heatmap。

---

## Slide 9 — Explicit Geometry Adds Useful Facial Structure / 显式几何补充表情结构

**Official criteria:** 4, 6

### English on-slide copy

**Full Geometry-SVM**

```text
136 normalised coordinates
+ 38 landmark-derived descriptors
= 174 features
```

- brow: 8
- eyes: 10
- nose: 3
- mouth: 13
- global: 4

**Official-test result**

- Accuracy: 47.31%
- Macro-F1: **46.33%**
- Prediction: 18.08 ms

### 中文对应讲稿

> 几何特征显式表示眉眼距离、眼睛开合、嘴巴长宽比、嘴角曲率和全局对称性。
> 这些特征为 RBF-SVM 加入了与表情结构相关的先验。最终 Macro-F1 最高，而且
> classifier-only latency 仍低于 30 ms。

### Visual / Evidence

用五种颜色标出一张 68 点脸上的 brow、eyes、nose、mouth、global 区域。

必须说 `classifier-only latency`，不能说完整 webcam pipeline 是 18.08 ms。

---

## Slide 10 — Comparison and Ablation Justify the Final Model / 对比与消融支持最终选模

**Official criteria:** 4, 6

### English on-slide copy

| Model | Features | Accuracy | Macro-F1 | Prediction |
|---|---:|---:|---:|---:|
| HGB | 136 | 43.13% | 39.14% | 23.41 ms |
| Coordinate RBF-SVM | 136 | 46.31% | 43.47% | 5.17 ms |
| PCA95 RBF-SVM | 12 | 40.32% | 36.13% | 2.68 ms |
| Tuned coordinate SVM | 136 | 46.84% | 45.89% | 5.31 ms |
| Full Geometry-SVM | 174 | 47.31% | **46.33%** | 18.08 ms |

**Ablation**

- 12 variants: coordinate baseline, five add-one, full geometry, five drop-one
- drop-eyes test Accuracy: 47.37%
- drop-eyes test Macro-F1: 46.25%
- full geometry retained because Macro-F1 was higher

### 中文对应讲稿

> 我们的主指标是 Macro-F1，而不是 Accuracy。消融中，drop-eyes 的 Accuracy 比
> 完整模型高 0.06 个百分点，但 Macro-F1 低 0.09 个百分点，因此没有替换最终模型。
> 这说明 eye geometry 可能有冗余或噪声，但不能得出“眼睛没有用”的结论。

### Visual / Evidence

左侧保留精简表格，右侧使用：

`expert_level/artifacts_svm_geometry_ablation/group_ablation_validation.png`

消融细节放 Q&A backup，不在主讲中解释所有 12 个柱子。

---

## Slide 11 — The Matrix Improved, but Geometry Cannot Resolve Every Expression / 混淆改善但几何仍有边界

**Official criteria:** 5, 6

### English on-slide copy

**Before: HGB baseline**

- Accuracy: 43.13%
- Macro-F1: 39.14%

**After: tuned Geometry-SVM**

- Accuracy: 47.31%
- Macro-F1: 46.33%
- Macro-F1 improvement: **+7.19 percentage points**

**Persistent confusions**

- fear -> angry: 179
- fear -> sad: 175
- sad -> neutral: 219
- sad -> angry: 208

**Why**

Geometric overlap, missing texture cues, landmark errors and label ambiguity.

### 中文对应讲稿

> 从 HGB 到最终模型，改进来自更合适的 RBF boundary、validation tuning 和显式
> geometry features。混淆矩阵整体改善，但 fear、sad、neutral 仍然重叠。
> Landmarks 只能表达形状，无法表达皱纹、皮肤纹理和鼻部细节；此外 FER 的低分辨率
> 和标签歧义也会造成失败。

### Visual / Evidence

三栏布局：

1. `artifacts/confusion_matrix.png`：baseline；
2. `artifacts_svm_geometry/confusion_matrix.png`：final；
3. 放大一个 `artifacts_svm_geometry/failure_cases.png` 中的测试图。

失败案例必须标注：

`True label`、`Predicted label`、`Landmark quality`、`Missing visual cue`。

---

## Slide 12 — Happy and Sad Effects Are Rendered Procedurally / Happy 与 Sad 特效的渲染流程

**Official criteria:** 7, 9

### English on-slide copy

**Common runtime path**

```text
Webcam
 -> Haar face
 -> LBF landmarks
 -> 174-D features
 -> Geometry-SVM class
 -> class-specific renderer
```

**Effect 1 — Happy**

- warm face tint using alpha blending
- eight animated stars around the face
- star positions use `sin/cos(frame phase)`

**Effect 2 — Sad**

- blue face tint
- seven falling rain streaks
- vertical position advances with `frame_index`

### 中文对应讲稿

> 两个特效都不是预制视频。模型先输出类别，然后 renderer 根据 face ROI 和
> `frame_index` 实时生成图形。Happy 用 `cv2.addWeighted` 加暖色 tint，再以
> sin/cos 计算八颗旋转星星的位置；Sad 加蓝色 tint，并让七条雨线随帧号下落。

### Visual / Evidence

只展示两个效果：

- Happy before/after；
- Sad before/after；
- 中间放一条从 prediction 到 rendering 的箭头。

可使用 `artifacts/effects_preview.png` 辅助，但官方还要求程序截图/视频，因此需要
补录真实 webcam 片段。

---

## Slide 13 — Smoothing Reduces Flicker but Adds a Small Lag / 平滑降低闪烁但会增加轻微滞后

**Official criteria:** 7, 10

### English on-slide copy

**Observed poor result**

- face and landmarks jitter between frames
- class prediction switches near the decision boundary
- effect anchors jump when the detected face box moves

**Implemented correction**

- landmark EMA: `alpha = 0.60`
- probability EMA: `alpha = 0.35`
- face matching before probability smoothing
- frame-index-driven animation phase

**Trade-off**

> More stable effects, with a small response delay.

### 中文对应讲稿

> Flicker 主要有三层原因：Haar ROI 抖动、LBF landmarks 抖动，以及模型在相邻
> 类别边界附近逐帧切换。我们先匹配同一张脸，再分别对 landmarks 和 probability
> 做 EMA。动画相位由连续 frame index 驱动，避免每次分类后重新开始。
> 平滑改善稳定性，但也会带来轻微反应滞后。

### Visual / Evidence

录制同一段表情：

1. 按 `S` 关闭 smoothing；
2. 再按 `S` 打开 smoothing；
3. 画面同时保留 expression label 和 effect。

PPT 中放左右两个 5–6 秒视频或同一视频的前后半段，并加：

`Smoothing OFF` / `Smoothing ON`。

---

# Part C — Bonus Level

## Slide 14 — Pose Tracking Must Survive Multiple People and Missing Joints / 姿态检测要处理多人和缺失点

**Official criteria:** 11, 12

### English on-slide copy

**Task 1 pipeline**

```text
Video
 -> YOLOv8n Pose
 -> all people + 17 keypoints
 -> primary dancer tracking
 -> confidence filtering
 -> EMA smoothing
 -> skeleton and pose cache
```

| Challenge | Implemented response |
|---|---|
| multiple people | area + centre + temporal continuity |
| occlusion / missing joints | confidence mask and coverage |
| keypoint jitter | EMA smoothing |
| CPU cost | precompute the reference pose |

**Selected 60-frame observations**

- Dance example: 16.95/17 visible joints; 1.67% multi-person frames
- TikTok sample: 13.77/17 visible joints; 85% multi-person frames

### 中文对应讲稿

> 首帧主要根据人物面积、画面中心和 pose confidence 选择主舞者；后续加入 continuity，
> 减少多人画面中的身份切换。低置信度点不参与骨架或评分，时间 EMA 降低抖动。
> 参考视频提前计算并缓存，游戏时只对 webcam 运行 YOLO。

### Visual / Evidence

使用：

- `task1_results/dance_example_1/contact_sheet.jpg`
- `task1_results/seq_00001_00009_YouTube/contact_sheet.jpg`
- `task2_results/dance_example_1/annotated.mp4` 中 6–8 秒片段。

这些数字是选定片段的 detection availability，不是人工标注 keypoint accuracy。

---

## Slide 15 — Pose-only Scoring Rewarded Static Players / 只看姿势会错误奖励静止玩家

**Official criteria:** 13, 14

### English on-slide copy

**Scoring iteration**

```text
Iteration 1
Normalised pose = angles + positions + coverage
Problem: a static player can match a similar past pose

Iteration 2
Pose + 0.4 s motion + anti-static factor
Problem: a true reference pause should not be punished

Iteration 3
SYNC / HOLD / MOVE! state gate
then PERFECT / GREAT / GOOD / MISS
```

**Current active-motion score**

\[
Score=(0.55Score_{pose}+0.45Score_{motion})F_{static}
\]

### 中文对应讲稿

> 第一版先解决空间问题：比较角度、归一化位置和 coverage。但测试发现，玩家保持普通
> 站姿，也可能匹配到历史窗口中的某个相似姿势。第二版加入 0.4 秒运动向量和活动幅度，
> 并用 anti-static factor 惩罚明显不动的玩家。随后又发现参考舞者真实停顿时不能惩罚
> 玩家，因此第三版增加 SYNC、HOLD 和 MOVE! 状态门控。

### Visual / Evidence

主视觉是一条：

`Problem -> Change -> New problem -> Change`

同时嵌入 Dance 双面板 6–8 秒预录视频，必须能看到：

- 左侧 reference；
- 右侧 webcam；
- 双方 skeleton；
- similarity、feedback 和 score。

---

## Slide 16 — Fair Scoring Aligns Space, Motion and Reaction Time / 公平评分需要空间、运动和反应时间对齐

**Official criteria:** 14

### English on-slide copy

**Spatial alignment**

- root at hip centre
- divide by a robust body scale
- compare 8 joint angles and 12 body-joint positions
- coverage penalises missing evidence

**Similarity**

\[
Score_{pose}=100(0.65S_{angle}+0.35S_{position})F_{coverage}
\]

\[
S_{motion}=0.70S_{vector}+0.30S_{activity}
\]

**Reaction-delay alignment**

- search current and past reference poses, up to 0.8 s
- never search future poses
- use the same real-time motion duration for each candidate
- compare direct and anatomical-mirror candidates

### 中文对应讲稿

> 空间上，我们将骨架平移到髋中心，再按肩宽、髋宽和躯干尺度归一化。姿势分由
> 65% 角度和 35% 位置组成，并用 coverage 处理遮挡。运动分比较最近 0.4 秒内
> 关节位移方向和总体活动量。
>
> 时间上，玩家必然需要反应时间，所以系统在当前和过去最多 0.8 秒的参考动作中
> 选择综合分最高者。它不搜索未来动作，因此不会奖励提前做尚未出现的动作。

### Visual / Evidence

必须使用时间轴区分两个窗口：

```text
0.8 s past-only search:
t-0.8s  <-------------------------------  t
                 best reference ^
current player pose ---------------------^

0.4 s motion window:
previous joints ----------> current joints
```

页面右下角可放 deterministic result：

```text
requested delay 0.8 s
matched lag 0.8 s
score 100.00
```

必须说明该测试只证明索引和匹配逻辑，不是真实玩家 accuracy。

---

## Slide 17 — Similarity Becomes Grades, Points and a Final Result / 相似度如何变成 Grade 和总成绩

**Official criteria:** 15

### English on-slide copy

**State first, grade second**

```text
insufficient history -> SYNC
confirmed reference pause -> HOLD
player too static -> MOVE!
otherwise -> grade by similarity
```

| Similarity | Grade | Event points | Combo |
|---:|---|---:|---|
| 85–100 | PERFECT | 1000 | +1 |
| 70–84.999 | GREAT | 700 | +1 |
| 55–69.999 | GOOD | 400 | +1 |
| below 55 | MISS | 0 | reset |

**During the dance**

- numeric similarity changes at every new webcam inference result
- textual grade and Combo update with each score event

**After the dance**

- Total Score = sum of event points
- Average = mean continuous similarity of score events
- Best Combo = longest successful sequence

### 中文对应讲稿

> 系统先判断特殊状态，再映射 Grade。Similarity 是连续诊断值；PERFECT、GREAT、
> GOOD、MISS 是离散游戏反馈；Total Score 则累加每个事件的积分。回合结束后保留
> Total Score、Average similarity 和 Best Combo。
>
> 当前限制是每个新 YOLO 结果可能产生一次积分事件，所以更快的电脑在相同时间内
> 可能产生更多评分机会。下一版应按固定音乐 beat 或固定参考时间片评分。

### Visual / Evidence

使用一段 8–10 秒录屏，展示：

```text
SYNC -> GOOD -> GREAT -> PERFECT -> MOVE! -> HOLD
```

如果真实录屏无法覆盖全部状态，可使用两段录屏，不要伪造一次连续发生的状态序列。

Grade 的详细问题与改进方向放在 Backup Slide B3–B4。

---

## Slide 18 — Pose Control Extends Beyond Dancing / 姿态交互可以扩展到平台游戏

**Official criteria:** 16 and closing

### English on-slide copy

**Pose-controlled platform game**

| Detected action | Landmark rule | Rendered response |
|---|---|---|
| left / right | wrist extension relative to shoulders | horizontal movement and run sprite |
| jump | both wrists above shoulders | jump physics and jump sprite |
| crouch | wrists close together near chest | crouch state and sprite |

**Features and temporal logic**

- shoulders, elbows and wrists only
- shoulder-width normalisation
- elbow angle and wrist-position features
- 3-frame median, EMA, confirmation and hysteresis
- rule-based recognition; no additional training set

**Game score**

- jump: +10
- coin: +100
- enemy stomp: +250
- checkpoint: +500
- power-up: +1000
- goal: time bonus

### 中文对应讲稿

> 除了舞蹈，我们把相同的 YOLO body keypoints 接入平台游戏。识别只使用肩、肘和
> 手腕，并按肩宽归一化。左右动作使用手腕伸展距离和肘角；双手高于肩膀触发 jump；
> 双手在胸前靠拢触发 crouch。这里没有额外训练集，而是使用可解释的规则、连续帧
> 确认、滤波和 hysteresis。
>
> 手势状态控制人物物理和 sprite，金币、敌人、平台与终点由游戏循环渲染。分数来自
> 通过动作完成的跳跃、收集、踩敌人、检查点和通关时间。

### Visual / Evidence

使用实际 Mario webcam + game 双区域录屏，必须同时看到：

- webcam skeleton；
- 当前 gesture label；
- 游戏人物响应；
- score 变化。

页面最下方用一句话收束整场汇报：

> Our contribution is a failure-driven, keypoint-based interaction pipeline.
>
> 我们的核心贡献是一条由失败分析驱动的关键点交互 pipeline。

---

# Part D — Q&A Backup Slides

以下页面放在主讲 18 页之后，正常情况下不主动讲，但可用于 3 分钟 Q&A。

## Backup Slide B1 — Complete Expert Metrics / Expert 完整指标

| Model | Dim. | Accuracy | Macro-F1 | Prediction |
|---|---:|---:|---:|---:|
| HGB | 136 | 43.13% | 39.14% | 23.41 ms |
| RBF-SVM | 136 | 46.31% | 43.47% | 5.17 ms |
| PCA95 RBF-SVM | 12 | 40.32% | 36.13% | 2.68 ms |
| Tuned coordinate SVM | 136 | 46.84% | 45.89% | 5.31 ms |
| Full Geometry-SVM | 174 | 47.31% | 46.33% | 18.08 ms |
| drop-eyes ablation | 164 | 47.37% | 46.25% | 4.90 ms |

说明：

- Accuracy 只作为辅助；
- 主指标是 Macro-F1；
- latency 为 classifier-only；
- 当前是 single stratified validation split，不是 K-fold。

---

## Backup Slide B2 — How the Ablation Was Run / 消融实验怎样完成

### English on-slide copy

```text
1 coordinates-only baseline
+ 5 add-one-group variants
+ 1 full-geometry variant
+ 5 drop-one-group variants
= 12 candidates
```

- coarse screening: stratified 10k subset
- top three: full 22,967 / 5,742 validation
- fixed `C=10`
- `gamma=2/current feature dimension`
- official test used after selection

**Full-validation Macro-F1**

- drop eyes: 45.12%
- full geometry: 44.58%
- drop global: 44.58%

**Official test Macro-F1**

- full geometry: 46.33%
- drop eyes: 46.25%

### 中文解释

消融显示 eye geometry 可能有冗余，但 single split 的 validation 优势没有在 official
test 上保持。不能说“去掉眼睛一定更好”，也不能说“eye features 没有用”。

---

## Backup Slide B3 — Problems in Each Grade Band / 每个 Grade 的问题和修改方向

| Grade | Current meaning | Current problem | Next modification |
|---|---|---|---|
| PERFECT ≥85 | very close pose and motion | scores near 85 can switch because of noise; threshold is hand-tuned | human-rating calibration; short score smoothing or dwell |
| GREAT 70–84.999 | strong overall match | wide band hides whether pose or motion caused the loss | display pose/motion sub-scores and joint-level hints |
| GOOD 55–69.999 | acceptable attempt | pose error, weak motion and low coverage collapse into one label | separate pose, timing and low-confidence feedback |
| MISS <55 | failed score event | one occlusion or bad temporal match can reset Combo | confidence-aware grace period or brief persistence |

### 中文总结

- `PERFECT`：85 附近容易等级跳变；下一步用真人评分校准并加入短时平滑；
- `GREAT`：区间较宽；下一步展示 pose/motion 子分数；
- `GOOD`：错误来源不清楚；下一步区分姿势、节奏和低置信度；
- `MISS`：一次遮挡也可能重置 Combo；下一步增加基于置信度的短暂容错。

共同问题：

> Grade thresholds are interpretable engineering rules, but they are not learned
> from human dance-quality labels.

---

## Backup Slide B4 — Problems in SYNC, HOLD and MOVE! / 特殊状态的问题和修改方向

| State | Current implementation | Why it exists | Remaining issue / next step |
|---|---|---|---|
| SYNC | wait for usable motion history | motion cannot be measured from one pose | add a visible warm-up countdown |
| HOLD | reference-motion EMA 0.35; enter below 0.07 for 2 samples; exit at 0.10 | correct stillness should not be punished | use time-based dwell instead of inference-sample count |
| MOVE! | \(M_p<\max(0.06,0.45M_r)\); 0 points; Combo reset | prevent static-pose exploitation | separate movement compliance from Average similarity |

HOLD 的迭代逻辑：

```text
single threshold
    -> false HOLD flashes during momentary slow-down

EMA + two-sample confirmation + hysteresis
    -> more stable ACTIVE/HOLD state
```

HOLD 不加分，也不进入 Average，避免同一个静止姿势重复刷 PERFECT。

---

## Backup Slide B5 — Delay and Mirror Validation / 延迟与镜像验证

| Player delay | Allowed lag | Mirror player | Expected observation |
|---:|---:|---|---|
| 0.0 s | 0.0 s | no | synchronous score near 100 |
| 0.8 s | 0.0 s | no | score drops in fast motion |
| 0.8 s | 0.8 s | no | correct past reference recovered |
| 0.8 s | 0.8 s | yes | mirrored delayed pose recovered |

当前确定性检查：

```text
requested delay: 0.8 s
matched lag: 0.8 s
delayed self-match score: 100.00
mirrored delayed self-match score: 100.00
```

该检查证明：

- frame index 与 seconds 换算正确；
- past-window search 正确；
- anatomical mirror 正确。

它不证明：

- 真实玩家评分 accuracy；
- 所有人最合理的 delay 都是 0.8 s；
- 摄像头遮挡和关键点噪声已经解决。

---

## Backup Slide B6 — Current Bonus Limitations / Bonus 当前限制

1. 2D pose 缺少 depth、手指和手掌方向；
2. 65/35、55/45 和 Grade thresholds 是人工工程参数；
3. 0.8 s 历史搜索可能在重复动作中选中碰巧相似的旧姿势；
4. 当前不对较大的 lag 额外扣分；
5. 每个评分事件独立选择最佳 lag，可能出现 lag 跳变；
6. 0.4 s 首尾净位移可能漏掉“移动后回到原位”的轨迹；
7. Total Score 受 YOLO inference event 数量影响；
8. 当前缺少多人真实用户和人工舞蹈质量评分实验。

优先改进顺序：

1. 固定 beat/time-slice scoring；
2. lag regularisation 和时间连续性；
3. 累积 motion path length；
4. 用真实玩家评分校准 weights、sigma 和 Grade thresholds。

---

## 5. 必须补录的 Presentation 素材

因为 Stage 1 禁止现场运行程序，以下素材必须在制作 PPT 前完成。

### 5.1 Beginner

运行：

```powershell
conda activate vc_sws3026
.\run_beginner_level.ps1
```

需要：

- baseline Haar failure：正脸、转头、背光、遮挡；
- improved detector：重复相同动作；
- 同屏显示 face box、68 landmarks、状态和 FPS；
- 每段录制 6–10 秒。

### 5.2 Expert

运行：

```powershell
conda activate vc_sws3026
.\run_expert_level.ps1
```

需要：

- Happy sparkle；
- Sad rain；
- smoothing OFF；
- smoothing ON；
- 表情 label、confidence、landmarks 和 effect 同时可见。

### 5.3 Bonus Dance

运行：

```powershell
conda activate vc_sws3026
.\run_bonus_level.ps1
```

需要：

- 左侧 reference + skeleton；
- 右侧 webcam + skeleton；
- numeric similarity；
- PERFECT/GREAT/GOOD/MISS；
- 至少一个 MOVE! 或 HOLD；
- Total Score、Average、Combo；
- 回合结束后的 Finished / Best Combo 画面。

### 5.4 Mario Extension

运行：

```powershell
conda activate vc_sws3026
.\run_mario.ps1
```

需要：

- webcam skeleton 与 gesture label；
- left/right；
- jump；
- crouch；
- coin 或 enemy interaction；
- score 变化和 final score。

### 5.5 视频格式建议

- 使用 MP4 / H.264；
- 720p 即可，避免 PPT 文件过大；
- 每段 6–10 秒；
- 裁去桌面、终端和无关窗口；
- 在 PowerPoint 中嵌入视频，不使用外部路径链接；
- 设置合适 poster frame；
- 在最终展示电脑上离线测试；
- 每个视频旁保留一张静态截图作为 fallback。

---

## 6. 证据文件路径

### Beginner

- 实现说明：
  `beginner_level/BEGINNER_LEVEL_REPORT_BILINGUAL.md`
- 鲁棒性记录模板：
  `beginner_level/docs/robustness_test.md`
- 改进 detector：
  `beginner_level/src/face_detector.py`
- landmark smoothing：
  `beginner_level/src/landmark_detector.py`

### Expert

- HGB baseline matrix：
  `expert_level/artifacts/confusion_matrix.png`
- HGB failure cases：
  `expert_level/artifacts/failure_cases.png`
- final matrix：
  `expert_level/artifacts_svm_geometry/confusion_matrix.png`
- final failure cases：
  `expert_level/artifacts_svm_geometry/failure_cases.png`
- PCA analysis：
  `expert_level/level2_report_assets/03_pca_information_loss.png`
- per-class F1：
  `expert_level/level2_report_assets/06_per_class_f1_comparison.png`
- tuned SVM heatmap：
  `expert_level/artifacts_svm_tuned/tuning_heatmap.png`
- ablation：
  `expert_level/artifacts_svm_geometry_ablation/group_ablation_validation.png`
- effect preview：
  `expert_level/artifacts/effects_preview.png`
- effect implementation：
  `expert_level/task2_realtime.py`

### Bonus

- Task 1 contact sheets：
  `bonus_level/task1_results/`
- annotated reference video：
  `bonus_level/task2_results/dance_example_1/annotated.mp4`
- dance scoring：
  `bonus_level/dance_scoring.py`
- Grade、Total Score、Average、Combo：
  `bonus_level/just_dance_app.py`
- delay/mirror test tool：
  `bonus_level/scoring_video_tester.py`
- detailed report：
  `bonus_level/LEVEL3_DANCE_SCORING_DETAILED_REPORT.md`
- Mario gesture recognition：
  `bonus_level_mario/gesture_controller.py`
- Mario rendering/scoring：
  `bonus_level_mario/mario_camera_demo.py`

---

## 7. PPT 视觉与信息密度规则

### 7.1 字体

- Deck title：至少 50 pt；
- Slide title：至少 35 pt；
- Subheading/callout：至少 24 pt；
- Body：至少 16 pt；
- 不要为了塞入更多内容而缩小字号。

### 7.2 每页只承担一个 narrative job

使用结论式标题：

- 推荐：`PCA Preserved Variance, Not Expression Evidence`
- 不推荐：`PCA Model`

- 推荐：`Pose-only Scoring Rewarded Static Players`
- 不推荐：`Bonus Scoring`

### 7.3 图表

- 两张 confusion matrix 必须使用相同大小和颜色尺度；
- 在矩阵旁写出 takeaway，不让观众自己寻找结论；
- 表格中高亮 Macro-F1，不只高亮 Accuracy；
- failure case 应放大一个案例，不要把 20 张缩略图全部塞入一页；
- 视频必须显示程序结果，而不是运行命令。

### 7.4 中英文

推荐：

- PPT 可见文字以英文为主；
- 中文放在标题第二行、少量副标题或 Speaker Notes；
- 本指南中的中文讲稿不需要全部复制到页面。

---

## 8. 禁止出现或需要谨慎使用的表述

不要说：

- “Stage 1 我们会现场运行 demo。”官方严格禁止。
- “Instructor 会根据 Stage 2 Dance demo 打分。”Stage 2 是 popularity contest。
- “Expert 使用 full face image 作为 classifier input。”
- “我们只训练了六类。”当前模型是七类。
- “Neutral 是后处理规则。”Neutral 是训练类别。
- “我们额外训练了 tongue expression。”当前没有。
- “47.31% Accuracy 是项目最终分数。”
- “Geometry-SVM 对每一个类别都最好。”
- “我们完成了 K-fold cross-validation。”当前没有。
- “18.08 ms 是完整 webcam pipeline latency。”
- “PCA 保留 95% 方差，所以保留了 95% 表情信息。”
- “Drop-eyes 证明眼睛特征没有用。”
- “TikTok 数据的 keypoint accuracy 是 100%。”
- “0.8 s 是系统固定把视频延迟。”
- “Delay self-match 100 分证明真实玩家评分准确率是 100%。”
- “HOLD 表示玩家偷懒。”HOLD 表示参考舞者已确认停顿。
- “Grade 是机器学习模型直接输出的概率。”
- “Grade thresholds 已经通过真人实验校准。”当前没有。
- “不同电脑的 Total Score 可以直接公平比较。”
- “Mario gesture classifier 使用了训练集。”当前是 rule-based。
- “YOLOv8 Pose 是我们自己训练的。”

---

## 9. 推荐开场与收尾

### 9.1 开场

**English**

> Our project asks one question across three levels: how can noisy visual
> keypoints become a stable and explainable interaction signal?

**中文**

> 我们三个 level 实际上在回答同一个问题：有噪声的视觉关键点怎样变成稳定、可解释
> 的交互信号？

### 9.2 收尾

**English**

> The main result is not a single detector or classifier. It is a sequence of
> failure-driven decisions: stabilise the keypoints, choose models with
> class-balanced evidence, and align body movement in both space and time.

**中文**

> 我们的主要结果不是某一个 detector 或 classifier，而是一系列由失败分析推动的
> 决策：先稳定关键点，再用类别平衡证据选模，最后在空间和时间上对齐身体动作。

---

## 10. 提交前最终检查

### Official coverage

- [ ] Haar failure screenshots/video
- [ ] Task 3 improvement explanation
- [ ] robust detector before/after
- [ ] Expert preprocessing, features and classifier
- [ ] six basic expressions + neutral additional category
- [ ] baseline confusion matrix and failure image
- [ ] final confusion matrix and improvement explanation
- [ ] Expert Task 2 screenshot/video
- [ ] two effects explained from detection to rendering
- [ ] flicker cause and smoothing A/B
- [ ] Bonus Task 1 screenshot/video
- [ ] keypoint challenges and solutions
- [ ] Dance app screenshot/video
- [ ] spatial, motion and delay alignment
- [ ] numeric/text score over time
- [ ] final Total Score, Average and Best Combo
- [ ] Mario actions, features, rendering and score

### Presentation safety

- [ ] 主讲不超过 10 分钟
- [ ] 不安排 Stage 1 现场运行程序
- [ ] 所有视频已嵌入 PPT
- [ ] 所有视频可在展示电脑离线播放
- [ ] 每段视频都有静态 fallback
- [ ] 没有虚构 K-fold、real-user accuracy 或人工校准结果
- [ ] Q&A backup slides 放在主讲页之后
