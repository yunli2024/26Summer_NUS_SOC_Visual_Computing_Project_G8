# Visual Computing Project PPT Guidelines

本指南基于：

- `Visual_Computing_Project.pdf` 的 Beginner、Expert、Bonus 要求；
- 本地 Beginner Level；
- Zhangyx `part two` 覆盖后的 `expert_level`；
- Zhangyx `part three` 覆盖后的 `bonus_level`；
- 仓库中已经存在的 metrics、confusion matrix、failure cases、contact sheet
  和 deterministic tests。

PPT 的主线不是“我们做了三个程序”，而是：

> 我们从 facial landmarks 的实时稳定性出发，将 keypoints 转换成可解释的
> 表情几何特征，再扩展到全身姿态、空间归一化、运动比较和反应延迟补偿。
> 每一层都经历了问题观察、对照实验、失败分析和迭代。

## 1. 汇报原则

### 1.1 每一部分都讲 exploration loop

统一使用以下顺序：

1. Requirement：课程要求解决什么问题。
2. Baseline：最初方案是什么。
3. Observation：实际看到什么失败。
4. Hypothesis：为什么会失败。
5. Experiment：比较或修改了什么。
6. Evidence：用什么图、表、指标证明。
7. Decision：最终保留什么方案。
8. Limitation：仍然没有解决什么。

### 1.2 区分已有结果与现场结果

- 仓库 JSON/报告中的指标可以作为历史实验结果。
- Webcam FPS、实时画面稳定性和摄像头兼容性必须在展示电脑上再次确认。
- 不要把分类器单次 inference time 当成完整 webcam pipeline latency。
- 不要把无 ground truth 的 pose detection availability 当成 keypoint accuracy。

### 1.3 验证协议必须诚实

Zhangyx Part Two 的正式模型使用：

- 官方 `train`/`test` split；
- 在官方 train 内做一次固定的 stratified 80/20 validation split；
- 用 validation Macro-F1 选择参数；
- 选模完成后才在官方 test 上评估一次。

当前结果不是 5-fold cross-validation。课程 PDF 将 cross-validation 标为
optional，因此可以展示现有结果，但答辩中必须说
`single stratified validation split`，不能说成 K-fold。

如果最终合并 rubric 明确要求 Stratified K-fold，应在提交前补做并更新本文件，
不要虚构尚未运行的 K-fold 数字。

## 2. 推荐 PPT 结构

建议 18 页左右。时间不足时优先保留标有“核心”的页面。

## Slide 1 - Title

标题：

`Real-time Video Analysis and Rendering`

副标题：

`From Facial Landmarks to Expression Effects and Dance Scoring`

内容：

- 课程、组号、成员；
- 一张最终系统拼图：Beginner 人脸点、Expert 表情特效、Bonus 双面板评分。

讲解重点：

> 我们关注的不只是最终效果，而是 keypoint pipeline 在真实条件下为什么失败，
> 以及怎样通过归一化、模型比较和时空对齐逐步提高可用性。

## Slide 2 - Project Map and Research Questions

用三层流程图：

```text
Webcam face
  -> 68 facial landmarks
  -> expression geometry/classification
  -> visual effects
  -> 17 body keypoints
  -> spatial + temporal alignment
  -> dance score and feedback
```

三个研究问题：

1. Face detector 与 LBF 在光照、姿态、遮挡下是否稳定？
2. 只使用 landmarks 时，不同模型和特征为什么表现不同？
3. 不同人体位置、尺度、镜像和反应延迟下，如何公平评分？

## Slide 3 - Beginner Requirement and Baseline

课程要求：

- 本地 webcam；
- Haar face box；
- LBF 68 landmarks；
- 实时显示和正常退出；
- 鲁棒性观察与替代/改进方案。

展示：

- pipeline 图；
- 正常光照正脸截图；
- FPS overlay。

讲解：

> Haar 决定在哪里找脸，LBF 只能在成功人脸框中拟合。因此 landmarks 失败不一定
> 是 LBF 本身，也可能是上游 detector 没有给出正确 ROI。

## Slide 4 - Beginner Failure Matrix

建议现场记录同一个人四种场景：

| 场景 | Face box | 68 points | Jitter | 观察 |
| --- | --- | --- | --- | --- |
| 正常正脸 |  |  |  |  |
| 暗光/背光 |  |  |  |  |
| 左右转头 |  |  |  |  |
| 手遮嘴/遮眼 |  |  |  |  |

不要只写“效果不好”，要指出失败层级：

- detector miss；
- false-positive face；
- face box 抖动；
- LBF 点漂移；
- 快速运动造成 temporal jitter。

## Slide 5 - Beginner Iteration

建议讲成 before/after：

- preprocessing：raw、CLAHE、gamma、CLAHE + gamma；
- detector 参数：`scaleFactor`、`minNeighbors`、最小人脸尺寸；
- 单脸稳定跟踪；
- landmark smoothing；
- alternative detector 的计划或实测结果。

如果没有完成 alternative detector 实验，必须写成 future work，不能写成已验证。

结论形式：

> CLAHE 主要改善局部低对比，但可能放大噪声；更严格的 Haar 参数降低误检，
> 但会提高漏检。最终参数是 precision、recall 和实时性的折中。

## Slide 6 - Expert Data and Keypoint Representation

数据：

| Split | Images |
| --- | ---: |
| Train | 28,709 |
| Test | 7,178 |
| Total | 35,887 |

类别不平衡例子：

- `happy` train：7,215；
- `disgust` train：436；
- 相差约 16.5 倍。

表示流程：

1. FER 48x48 放大到 192x192；
2. 使用 centered FER crop 给 LBF 提供人脸区域；
3. 计算左右眼中心；
4. 平移到眼中心中点；
5. 旋转使眼线水平；
6. 用 inter-eye distance 缩放；
7. 展平为 136 coordinates；
8. 最终模型再加入 38 个 landmark-only geometry descriptors。

必须强调：

> Centered ROI 是 FER crop 专用策略；实时 webcam 仍使用真实 face detector。

## Slide 7 - Why Keypoint Extraction Needed Iteration

问题：

- FER 只有 48x48；
- Haar 在低清、紧裁剪 FER 图上经常漏检；
- 失败不等于图中没有脸。

探索：

- `haar`；
- `haar-fallback`；
- `center`。

最终选择：

- 离线 FER 使用 centered region；
- 实时 webcam 使用 Haar + LBF。

局限：

- centered region 可能接受错误 landmark fit；
- 应增加 landmark geometry quality rejection。

## Slide 8 - Expert Model Ladder（核心）

使用仓库真实指标：

| Experiment | Features | Accuracy | Macro-F1 | Single prediction |
| --- | ---: | ---: | ---: | ---: |
| HGB | 136 | 43.13% | 39.14% | 23.41 ms |
| RBF-SVM | 136 | 46.31% | 43.47% | 5.17 ms |
| PCA95 + RBF-SVM | 12 PCA comps | 40.32% | 36.13% | 2.68 ms |
| Tuned coordinate SVM | 136 | 46.84% | 45.89% | 5.31 ms |
| Tuned geometry SVM | 174 | 47.31% | 46.33% | 18.08 ms |

讲解不要只说谁最高：

- HGB：非线性但在连续几何空间中边界不如 RBF-SVM 合适；
- RBF-SVM：适合连续、重叠的 landmark geometry；
- PCA：更快，但 95% variance 不等于保留表情判别信息；
- Geometry-SVM：显式表达嘴角、眼睛开合、眉眼距离、对称性。

## Slide 9 - PCA Failure Analysis

核心发现：

- PCA 将 136 coordinates 压到 12 components；
- prediction 降到 2.68 ms；
- Macro-F1 从基础 SVM 的 43.47% 降到 36.13%。

推荐证据：

- `expert_level/level2_report_assets/03_pca_information_loss.png`
- `expert_level/artifacts_svm_pca95/confusion_matrix.png`

讲解：

> PCA 优先保存总体方差，但表情可能由低方差的嘴角、眼睑或眉毛变化决定。
> 因此“保留 95% 方差”仍可能丢失判别信息。

## Slide 10 - Geometry Iteration and Ablation

最终输入：

```text
136 normalized coordinates + 38 geometry descriptors = 174 features
```

相对 tuned coordinate SVM：

- Accuracy：+0.47 percentage points；
- Macro-F1：+0.45 points；
- `fear` F1：+2.18 points；
- `sad` F1：+1.88 points；
- `happy`、`angry`、`disgust` 并非全部改善。

消融候选：

- 去掉 eye geometry 后 Accuracy 47.37%；
- Macro-F1 46.25%，略低于最终完整 geometry 模型；
- 因主指标是 Macro-F1，所以未替换最终模型。

结论：

> Feature engineering 有整体收益，但不是每一组特征、每一个类别都受益。

## Slide 11 - Confusion Matrix and One Failure Case（核心）

展示：

- `expert_level/artifacts_svm_geometry/confusion_matrix.png`
- `expert_level/artifacts_svm_geometry/failure_cases.png`

真实混淆：

- fear -> angry：179；
- fear -> sad：175；
- fear -> neutral：141；
- sad -> neutral：219；
- sad -> angry：208；
- sad -> fear：206。

建议深挖一个案例，而不是摆很多缩略图：

1. true label / predicted label；
2. landmarks 是否合理；
3. geometry 为什么相似；
4. 缺失了什么 texture cue；
5. 是模型问题、keypoint 问题还是 label ambiguity。

可用案例类型：

- fear 与 surprise/angry 几何重叠；
- disgust 依赖鼻纹、皱眉和纹理；
- neutral 标签图中实际有笑容；
- 极端裁剪、水印或无效人脸。

## Slide 12 - Expert Final Model and Real-time Effects

最终模型：

- class-balanced RBF-SVM；
- `C=10`；
- `gamma=0.0114943`；
- 174 landmark-derived features；
- 18.08 ms classifier-only benchmark；
- 满足 PDF 的 `<30 ms` 分类器目标。

特效映射：

| Expression | Effect |
| --- | --- |
| Happy | warm tint + sparkles |
| Surprise | orange tint + star |
| Angry | red tint + action rays |
| Sad | blue tint + rain |
| Fear | purple tint + echo boxes |
| Disgust | green tint + bubbles |
| Neutral | cyan corner markers |

稳定化：

- landmark EMA alpha 0.60；
- probability EMA alpha 0.35；
- 可现场用 `S` 切换 smoothing 做 before/after。

必须说明：

> 18.08 ms 仅是 classifier prediction，不包含 Haar、LBF、绘制和摄像头读取。

## Slide 13 - Bonus Task 1: Pose Exploration

Pipeline：

```text
video
 -> YOLOv8n Pose
 -> all people + 17 COCO keypoints
 -> primary dancer selection
 -> confidence filtering
 -> EMA smoothing
 -> skeleton + metrics + pose cache
```

主舞者选择：

- 首帧：area 60%、center 30%、confidence 10%；
- 后续：continuity 45%、area 30%、center 15%、confidence 10%。

已保存实验：

| Clip | Frames | Primary dancer | Multi-person | Avg visible joints |
| --- | ---: | ---: | ---: | ---: |
| dance example | 60 | 100% | 1.67% | 16.95/17 |
| TikTok sample | 60 | 100% | 85% | 13.77/17 |

这些数字只描述选定片段，不是整个 TikTok dataset accuracy。

证据：

- `bonus_level/task1_results/.../contact_sheet.jpg`
- 对应 `analysis.json`

## Slide 14 - Bonus Runtime Architecture

Part Three 的关键优化：

1. reference video 预处理；
2. 保存 annotated MP4 与 `pose_cache.npz`；
3. 游戏时左侧读取缓存；
4. 只有 webcam 运行 YOLO；
5. camera worker thread 与 Tkinter GUI 分离；
6. GUI 只消费最新 inference result，避免积压。

讲解：

> CPU 环境下同时运行两路 YOLO 会严重降低实时性。预缓存 reference side
> 将运行期成本从两路 inference 降为一路。

## Slide 15 - Spatial Alignment and Pose Score（核心）

使用肩到脚踝的 12 个 body joints，不使用 5 个 face keypoints。

归一化：

1. 以 hip center 为 root；
2. hip 不可见时退化到 shoulder center 或可见 body centroid；
3. scale 综合 shoulder width、hip width、torso length 和 body extent；
4. 所有点执行 `(point - root) / scale`。

单帧分：

```text
PoseRaw = 0.65 * AngleSimilarity + 0.35 * PositionSimilarity
PoseScore = 100 * PoseRaw * (0.60 + 0.40 * Coverage)
```

- 8 个 elbow/shoulder/hip/knee angles；
- angle sigma：32 degrees；
- normalized position sigma：0.55；
- wrists/ankles 权重 1.35；
- 至少 4 个共同 body joints 才评分。

设计理由：

- angles 对平移和统一缩放稳定；
- positions 保留手脚整体位置；
- coverage 防止少量可见点得到过高置信度。

## Slide 16 - Motion, Grade and Anti-static（核心）

0.4 s motion window：

```text
Motion = 0.70 * VectorAgreement + 0.30 * ActivityAgreement
Final = (0.55 * PoseScore + 0.45 * 100 * Motion) * AntiStaticFactor
```

`AntiStaticFactor` 范围约 0.45-1.00。参考在动而玩家活动不足时：

- 显示 `MOVE!`；
- 0 points；
- combo reset。

Grade：

| Similarity | Feedback | Points | Combo |
| --- | --- | ---: | --- |
| insufficient history | SYNC | 0 | unchanged |
| confirmed reference hold | HOLD | 0 | unchanged |
| player too static | MOVE! | 0 | reset |
| 85-100 | PERFECT | 1000 | +1 |
| 70-84.99 | GREAT | 700 | +1 |
| 55-69.99 | GOOD | 400 | +1 |
| below 55 | MISS | 0 | reset |

为什么 HOLD 不加分：

- 避免同一个静止姿势被反复刷 PERFECT；
- HOLD 表示 reference 停顿，不表示玩家偷懒。

## Slide 17 - Delay, Mirror and HOLD Stabilization（核心）

时间对齐：

- 玩家与当前 reference 以及过去最多 0.8 s 的 reference poses 比较；
- 选择综合分最高的过去帧；
- 只搜索当前和过去，不搜索未来；
- 0.8 s 是 maximum lag，不是固定延迟。

镜像：

- x coordinate 取反；
- 同时交换 anatomical left/right joints；
- direct 与 mirrored candidate 取更高分。

HOLD 防闪烁：

- reference activity EMA alpha 0.35；
- 连续 2 个评分样本 `<0.07` 才进入 HOLD；
- smoothed activity `>=0.10` 才退出；
- 0.07-0.10 是 hysteresis band。

验证证据：

- 10 个 deterministic scoring tests；
- 0.8 s delayed self-match 得分 100；
- mirrored delayed self-match 得分 100。

## Slide 18 - Limitations, Next Iteration and Demo

Expert limitations：

- single validation split，不是 K-fold；
- FER -> webcam domain shift；
- landmarks 缺少 wrinkle、texture、depth；
- centered FER ROI 可能接受错误 LBF fit；
- 18.08 ms 不等于 end-to-end latency。

Bonus limitations：

- 2D pose 不包含 depth/hand/finger；
- score weights 是人工工程设计；
- 0.4 s 净位移可能漏掉“移动后回到原位”；
- total points 事件数受 YOLO inference rate 影响；
- 0.8 s lag 窗口可能过度放宽节奏。

下一轮优先级：

1. Expert 补做 Stratified K-fold 稳定性；
2. 添加 landmark quality rejection；
3. Bonus 固定 beat/time-slice 评分，提高不同电脑间公平性；
4. 累计 motion path length，而非只看窗口端点；
5. 用人工评分片段校准 sigma、weights 和 grade thresholds。

现场 Demo 顺序：

1. Beginner：raw 与 CLAHE、遮挡/转头；
2. Expert：smoothing on/off、不同表情特效；
3. Bonus：先 reference skeleton，再 webcam、delay、mirror、score；
4. 若时间足够，再展示 Mario extension。

## 3. 推荐图表和证据路径

### Expert

- 最终混淆矩阵：
  `VisualComputingProject/expert_level/artifacts_svm_geometry/confusion_matrix.png`
- 最终失败案例：
  `VisualComputingProject/expert_level/artifacts_svm_geometry/failure_cases.png`
- 六模型真实图片对照：
  `VisualComputingProject/expert_level/level2_report_assets/`
- 模型指标：
  各 `artifacts*/metrics.json`
- 详细解释：
  `VisualComputingProject/expert_level/LEVEL2_TRAINING_DETAILED_REPORT.md`

### Bonus

- 单人 contact sheet：
  `VisualComputingProject/bonus_level/task1_results/dance_example_1/contact_sheet.jpg`
- 多人 contact sheet：
  `VisualComputingProject/bonus_level/task1_results/seq_00001_00009_YouTube/contact_sheet.jpg`
- 实验 JSON：
  对应目录的 `analysis.json`
- 评分公式和答辩 Q&A：
  `VisualComputingProject/bonus_level/LEVEL3_DANCE_SCORING_DETAILED_REPORT.md`

## 4. 禁止出现的表述

不要说：

- “我们用了 full face image 做 Expert classifier。”
- “Accuracy 47% 就是项目得分。”
- “18.08 ms 代表整个 webcam pipeline。”
- “我们做了 5-fold cross-validation。”当前没有。
- “TikTok dataset 全部测试准确率是 100%。”
- “0.8 s 是系统固定把画面延迟。”
- “HOLD 表示玩家没有动。”
- “YOLO pose 是我们自己训练的。”
- “所有模型失败都是模型能力不足。”数据标签和 landmark fit 也可能有问题。

## 5. 一分钟总结模板

> Beginner 中，我们发现 facial landmarks 的稳定性强依赖 face detector、
> 光照和姿态，因此比较了 preprocessing 与检测稳定化策略。Expert 中，我们
> 严格只使用 68 个 landmarks。眼中心对齐得到 136 coordinates，显式几何特征
> 扩展到 174 维。六组实验显示 PCA 虽更快却丢失判别信息；最终 Geometry-SVM
> 得到 46.33% Macro-F1，分类器单次预测 18.08 ms。失败主要集中在
> fear/sad/neutral 的几何重叠、纹理信息缺失和 FER 标签噪声。Bonus 中，我们
> 使用 YOLOv8 Pose 提取 17 点并跟踪主舞者。评分前先做身体中心与尺度归一化，
> 再组合角度、位置、0.4 秒运动和可见点覆盖率；0.8 秒历史搜索补偿自然反应
> 延迟，镜像匹配和 HOLD hysteresis 提高实际可玩性。我们的核心贡献是把每个
> failure observation 转换成可解释、可验证的迭代。
