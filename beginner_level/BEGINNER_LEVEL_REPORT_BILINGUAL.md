# Beginner Level Face Recognition and Landmark Demo Report / Beginner Level 人脸识别与关键点 Demo 报告

## Git Commands / Git 常用命令

中文：以下命令可用于查看修改、提交代码，并上传到 remote GitHub 仓库。请将 `<remote-github-url>` 替换成你自己的 GitHub 仓库地址，例如 `https://github.com/username/repository.git`。

English: The following commands can be used to inspect changes, commit code, and upload the project to a remote GitHub repository. Replace `<remote-github-url>` with your own GitHub repository URL, for example `https://github.com/username/repository.git`.

```powershell
# 1. Check current repository status / 查看当前仓库状态
git status

# 2. View file changes / 查看具体修改内容
git diff

# 3. Stage all changed files / 暂存所有修改文件
git add .

# 4. Commit changes / 提交修改
git commit -m "Update beginner level face landmark demo and report"

# 5. Check existing remotes / 查看已有远程仓库
git remote -v

# 6. Add remote GitHub repository if not configured / 如果还没有远程仓库，则添加 GitHub remote
git remote add origin <remote-github-url>

# 7. Rename current branch to main if needed / 如有需要，将当前分支改名为 main
git branch -M main

# 8. Push to GitHub remote / 上传到 GitHub
git push -u origin main
```

中文：如果 `origin` 已经存在但地址不对，可以使用下面的命令修改 remote 地址。

English: If `origin` already exists but points to the wrong URL, update it with the command below.

```powershell
git remote set-url origin <remote-github-url>
git push -u origin main
```

## Project Overview / 项目概述

中文：本项目的 Beginner Level 实现了基于摄像头的实时人脸检测与 68 点人脸关键点标注。程序先从摄像头读取视频帧，再使用 Haar Cascade 检测人脸框，随后将检测到的人脸区域传给 OpenCV LBF Facemark 模型，得到每张人脸的 68 个关键点。Demo 会在画面上绘制人脸框、关键点、关键点编号、FPS 和当前检测状态。

English: The Beginner Level project implements real-time webcam-based face detection and 68-point facial landmark annotation. Each frame is captured from the camera, processed by a Haar Cascade face detector, and then passed to OpenCV's LBF Facemark model to estimate 68 landmarks for each detected face. The demo displays face boxes, keypoints, selected keypoint indices, FPS, and detection status.

## Latest Synchronized Changes / 最新同步修改说明

中文：根据本次修改要求，报告和 demo 已同步更新两点。第一，针对“脸部框选不够稳定、整个框不断刷新”的问题，改进版通过短时缓存上一帧有效人脸框、过滤异常候选框、抑制脸部内部误检，以及对连续帧中的关键点做平滑处理，降低人脸框和关键点的闪烁。第二，demo 视频图像中已经加入 68 个 keypoint 的显示：每个检测到的人脸都会显示 LBF 模型输出的 68 个关键点，并按下颌、眉毛、鼻子、眼睛、嘴巴等区域使用不同颜色标注，部分代表性点位还会显示编号。

English: According to the latest requested changes, both the report and the demo have been synchronized in two ways. First, to address the unstable and constantly refreshing face box, the improved version uses short-term caching of the previous valid face box, filters abnormal candidate boxes, suppresses internal facial-part false positives, and smooths landmarks across continuous frames to reduce flickering. Second, the demo video now displays the 68 keypoints: every detected face shows the 68 LBF landmarks, color-coded by jaw, eyebrows, nose, eyes, and mouth, with representative keypoint indices labeled on the frame.

中文：目录结构也已同步调整：原来的 `beginner_level/improved` 版本已经替换到 `beginner_level` 根目录中，因此现在直接运行 `beginner_level\main.py` 就是改进后的版本，不再需要进入单独的 `improved` 子目录。

English: The folder structure has also been synchronized: the previous `beginner_level/improved` version has been moved into the `beginner_level` root. Therefore, running `beginner_level\main.py` now starts the improved version directly, without using a separate `improved` subfolder.

中文：参考 `expert_level` 后，本次进一步增强了 Beginner Level 的人脸与 68 点关键点检测。检测器现在会结合上一帧稳定人脸框做 ROI 局部搜索，并与全图 Haar 检测结果合并、去重，这可以在用户左右转头或短时间偏离正脸时提高连续性。LBF 拟合前也会轻微扩展人脸框，让关键点模型获得更完整的下巴、额头和嘴部上下文。

English: After referencing `expert_level`, the Beginner Level face and 68-point landmark detector has been further strengthened. The detector now uses the previous stable face box as a local ROI search region, merges it with full-frame Haar detections, and removes duplicate boxes. This improves continuity when the user turns the head left or right or briefly moves away from a frontal pose. The face box is also slightly expanded before LBF fitting so the landmark model receives more complete chin, forehead, and mouth context.

中文：为了增强“转头的人脸判断”，项目新增了基于 68 个关键点的头部姿态估计，会计算 yaw、pitch、roll，并在画面中的人脸框上显示朝向标签。对于合理范围内的左右转头或俯仰，系统会继续认为它是同一张人脸；但如果姿态估计极端异常，则会把它作为可疑结果过滤掉。

English: To improve head-turn judgment, the project now estimates head pose from the 68 landmarks by calculating yaw, pitch, and roll, then displays a pose label near the face box. Reasonable left/right turns or pitch changes are still accepted as the same face, while extremely abnormal pose estimates are treated as suspicious and filtered out.

中文：为了规避非人脸误检测，系统现在不再只依赖 Haar Cascade 的候选框。每个候选框必须通过 LBF 68 点拟合，并通过宽松的关键点几何检查，包括关键点是否大体位于人脸框附近、关键点是否没有塌缩成一小团、整体宽高分布是否合理。未通过这些基础验证的候选框会被标记为 `REJECTED`，不会被绘制或缓存为下一帧的人脸。

English: To avoid false detections on non-face regions, the system no longer trusts Haar Cascade boxes alone. Each candidate must pass LBF 68-point fitting and loose landmark geometry checks, including whether the landmarks mostly stay near the face box, whether they do not collapse into a tiny cluster, and whether their overall width/height spread is plausible. Candidates that fail these basic checks are marked as `REJECTED` and are neither drawn nor cached for the next frame.

中文：随后根据实际测试反馈，Beginner Level 的关键点验证标准已参考 `expert_level/src` 调整为更宽松的策略。系统现在更接近 expert 版本的原则：只要 Haar 候选框能够成功拟合出 68 个 LBF keypoints，并且这些点没有明显塌缩、没有大面积跑出人脸框附近、整体宽高分布合理，就接受为正常人脸。头部 yaw/pitch/roll 现在主要用于显示转头方向，不再作为严格拒绝条件；眼睛、鼻子、下巴的严格纵向顺序也不再用于过滤，以避免正常低头、仰头、侧转头时被错误拒绝。

English: After practical testing feedback, the Beginner Level landmark validation has been relaxed based on the style used in `expert_level/src`. The system now follows the expert version more closely: if a Haar candidate can successfully produce 68 LBF keypoints, and the points are not collapsed, not largely outside the nearby face region, and have a plausible overall width/height spread, it is accepted as a normal face. Head-pose yaw/pitch/roll is now mainly used for displaying the turning direction rather than as a hard rejection rule. The strict vertical ordering of eyes, nose, and chin is also no longer used as a filter, preventing normal looking-down, looking-up, or side-turning faces from being rejected.

中文：根据 demo 画面镜像的问题，摄像头帧现在会在检测和显示前进行一次水平翻转，将镜像画面调整回正常方向。默认配置为 `UNMIRROR_CAMERA = True`；如果某台摄像头本身已经输出正常方向，可以运行时加入 `--keep-mirror` 保留原始画面。

English: To fix the mirrored demo video, each camera frame is now horizontally flipped before detection and display, restoring the normal viewing direction. The default configuration is `UNMIRROR_CAMERA = True`; if a camera already outputs a normal non-mirrored image, the `--keep-mirror` option can be used to keep the original feed.

中文：本次根据 `expert_level/src` 的实时检测方式继续优化 Beginner Level。Beginner Level 现在不再每帧都完全依赖全图 Haar 检测，而是参考 expert demo 使用“ROI 局部搜索为主、定期全图检测校准”的策略：上一帧稳定人脸框会作为下一帧的局部搜索区域，每隔 `FULL_DETECT_INTERVAL` 帧再进行一次全图扫描。这样可以减少人脸框不断刷新造成的跳动，同时在转头、俯仰头和短时检测失败时保持更好的连续性。

English: Based on the realtime recognition pipeline in `expert_level/src`, the Beginner Level detector has been optimized further. It no longer relies on full-frame Haar detection every frame. Instead, it follows the expert demo's strategy of using ROI local search as the primary path and full-frame detection as periodic correction: the previous stable face box is used as the next frame's local search region, while a full-frame scan is performed every `FULL_DETECT_INTERVAL` frames. This reduces box refresh jitter and improves continuity during head turns, pitch changes, and short detection failures.

中文：预处理也参考 expert 版本进行了增强。现在支持 `raw`、`clahe`、`gamma`、`clahe-gamma` 四种实时预处理模式，默认使用 `clahe`，以提升暗光、局部阴影和对比度不足情况下的人脸与五官关键点识别效果。运行时可以通过 `--preprocess raw|clahe|gamma|clahe-gamma` 切换。

English: Preprocessing has also been strengthened based on the expert version. The demo now supports four realtime preprocessing modes: `raw`, `clahe`, `gamma`, and `clahe-gamma`, with `clahe` as the default. This improves face and facial-feature keypoint detection under dim lighting, local shadows, and low contrast. The mode can be changed with `--preprocess raw|clahe|gamma|clahe-gamma`.

中文：68 个 keypoint 的稳定性也进一步参考 expert 版本优化。原来的简单整体平滑改为“分区域平滑”：下颌线、眉毛/鼻子、眼睛/嘴巴使用不同的平滑系数；同时限制单个关键点每帧最大跳变，并约束下颌线不要漂出人脸框过远。显示层还增加了额外平滑，使五官 keypoint 在视频中更稳定。

English: The 68-keypoint stability has also been improved using the expert version as a reference. The previous simple global smoothing has been replaced with region-aware smoothing: the jawline, brows/nose, and eyes/mouth use different smoothing factors. Each landmark's maximum per-frame jump is limited, and the jawline is constrained so it does not drift too far outside the face box. An additional display-only smoothing layer makes the visible keypoints more stable in the video.

中文：对于转头识别，系统继续使用基于 68 点的 yaw、pitch、roll 姿态估计，并参考 expert demo 增加了 pose-aware 的下颌轮廓显示修正。当用户左右转头时，LBF 模型有时会保留偏正脸的下颌模板，导致轮廓漂浮；现在显示层会根据 yaw 角度适度收缩和上移下颌轮廓，使侧脸时的关键点显示更贴近实际可见脸部。

English: For head-turn recognition, the system continues to estimate yaw, pitch, and roll from the 68 landmarks, and now adds the expert demo's pose-aware jawline display adjustment. When the user turns left or right, LBF may keep a frontal jaw template and make the contour float away from the visible face. The display layer now narrows and slightly lifts the jawline according to yaw, making side-face landmark visualization closer to the actual visible face.

中文：根据左转头时 keypoints 对不上的问题，姿态估计又增加了 2D yaw fallback。实际测试中，左转时 `solvePnP` 有时会把 yaw 低估，同时给出接近 160 度的异常 pitch，导致侧脸下颌显示修正没有触发。现在系统会额外检查鼻尖相对人脸框中心的水平偏移；当 3D 姿态估计明显异常或 yaw 被低估时，会用这个 2D yaw 作为兜底，使左转和右转都能触发 pose-aware keypoint 修正。

English: To fix the mismatch when turning the head left, a 2D yaw fallback has been added to pose estimation. In testing, `solvePnP` sometimes underestimated yaw during left turns while producing an abnormal pitch near 160 degrees, so the side-face jawline correction was not triggered. The system now also checks the horizontal offset of the nose tip relative to the face-box center. When the 3D pose estimate is clearly unstable or yaw is underestimated, this 2D yaw is used as a fallback so both left and right head turns can trigger pose-aware keypoint adjustment.

中文：根据张大嘴巴时口腔区域被误检测成第二张脸的问题，检测阶段新增了嵌套小脸过滤。系统会先按面积保留大的人脸框；如果另一个候选框大部分落在该大脸内部、面积明显更小，并且中心位于大脸下半部，则认为它更可能是嘴巴/下巴区域的误检，而不是第二张真实人脸。这样可以保留多人场景，同时减少张嘴时出现的口腔假人脸框和重复 keypoints。

English: To address the false detection caused by a widely opened mouth, nested small-face filtering has been added during detection. The system keeps larger face boxes first; if another candidate lies mostly inside a larger face, is much smaller, and has its center in the lower half of that larger face, it is treated as a mouth/chin false positive rather than a second real face. This keeps multi-person support while reducing fake mouth-region face boxes and duplicate keypoints when the mouth is open.

中文：针对张大嘴时下嘴唇 keypoints 错位的问题，嘴部区域现在加入了 open-mouth 动态处理。系统会检测外嘴唇和内嘴唇的张开比例；当嘴巴明显张开时，嘴部关键点使用更高的平滑更新系数和更大的单帧位移上限，让下嘴唇能更快跟随真实位置。同时，下嘴唇相关点位会受到轻量几何约束，避免它们被平滑拖在上嘴唇、牙齿或口腔内部过高的位置。

English: To fix lower-lip keypoint drift when the mouth opens widely, open-mouth dynamic handling has been added for the mouth region. The system checks the opening ratio of the outer and inner lips. When the mouth is clearly open, mouth landmarks use a higher smoothing update factor and a larger per-frame movement limit, allowing the lower lip to follow the real position faster. A light geometric constraint is also applied to lower-lip landmarks so they are not dragged too high toward the upper lip, teeth, or inside-mouth region.

中文：针对正脸时右侧脸颊/下颌线 keypoints 被拉到人脸框边缘的问题，下颌线约束进一步增强。除了原有的人脸框范围约束外，现在还会根据眼距、鼻子和嘴部中心估算真实脸部内部宽度，限制下颌线左右两侧不要过度漂向背景或耳朵区域。这样可以减少人脸框偏宽时 cyan 下颌点外移的问题。

English: To fix cheek and jawline keypoints drifting toward the face-box edge in frontal views, the jawline constraint has been strengthened. In addition to the original face-box bounds, the system estimates the internal face width from eye distance and the nose/mouth center, then prevents jawline points from drifting too far into the background or ear area. This reduces cyan jawline displacement when the detected face box is wider than the visible face.

中文：针对灯光较暗时识别效果变差的问题，视频帧现在会先进行低光增强，再进入人脸检测和 68 点 keypoint 拟合。增强流程在 BGR 视频帧上执行：先转换到 HSV 空间，对亮度 V 通道进行 gamma 提亮，再使用 CLAHE 增强局部对比度，最后与原图按比例融合。这样显示画面、人脸框检测和 LBF keypoint 拟合都使用增强后的图像，而不是只对灰度检测图做 CLAHE。状态栏会显示 `video-lowlight(...)` 或 `video-enhanced`，便于确认当前是否触发低光增强。

English: To improve recognition under dim lighting, each video frame is now enhanced before face detection and 68-point keypoint fitting. The enhancement runs on the BGR frame: it converts the frame to HSV, applies gamma brightening to the V channel, applies CLAHE for local contrast, and blends the enhanced result with the original frame. Therefore, the displayed image, Haar face detection, and LBF keypoint fitting all use the enhanced image, instead of applying CLAHE only to the grayscale detection image. The status text shows `video-lowlight(...)` or `video-enhanced` so it is clear when low-light enhancement is active.

中文：根据低光增强后画面变模糊、识别不稳定的问题，低光预处理已改为更保守的清晰增强。新版不再强力拉高 HSV 的 V 通道，而是在 YCrCb 色彩空间中只增强亮度 Y 通道，降低 gamma 和 CLAHE 强度，并降低增强图与原图的融合比例。同时加入轻量 unsharp mask 锐化，尽量保留眼睛、嘴唇、鼻梁等 LBF 需要的局部边缘纹理，减少过度提亮带来的噪声和糊感。

English: After observing blur and instability from the previous low-light enhancement, the preprocessing has been changed to a more conservative clarity-focused enhancement. Instead of strongly boosting the HSV V channel, the new version enhances only the Y luminance channel in YCrCb space, lowers the gamma and CLAHE strength, and reduces the blend ratio between the enhanced frame and the original frame. A light unsharp mask is also applied to preserve local edge details around the eyes, lips, and nose bridge, reducing noise amplification and softness caused by over-enhancement.

中文：为了方便在 demo 中直接比较不同预处理效果，现在支持运行时键盘切换。按 `1` 使用 `raw`，按 `2` 使用 `clahe`，按 `3` 使用 `gamma`，按 `4` 使用 `clahe-gamma`，按 `v` 开关整帧低光视频增强，按 `m` 开关镜像修正，按 `r` 重置检测缓存和平滑状态，按 `q` 退出。每次切换都会清空上一模式的缓存，避免旧的人脸框或 keypoint 平滑结果影响对比。

English: To compare preprocessing effects directly in the demo, runtime keyboard switching is now supported. Press `1` for `raw`, `2` for `clahe`, `3` for `gamma`, `4` for `clahe-gamma`, `v` to toggle full-frame low-light video enhancement, `m` to toggle mirror correction, `r` to reset detection cache and smoothing state, and `q` to quit. Each switch clears the previous mode's cache so old face boxes or smoothed keypoints do not affect the comparison.

| Mode / 模式 | Best Use / 适用场景 | Advantage / 优点 | Risk / 风险 |
|---|---|---|---|
| `raw` | Bright, stable lighting / 明亮稳定光照 | Preserves original image, least processing noise / 保留原图，噪声最少 | Weak under dim light / 暗光下较弱 |
| `clahe` | Uneven or dim lighting / 光照不均或偏暗 | Improves local contrast for eyes, nose, mouth / 提升眼鼻嘴局部对比 | Can amplify noise slightly / 可能轻微放大噪声 |
| `gamma` | Globally dark image / 整体偏暗 | Brightens the whole grayscale image / 整体提亮灰度图 | May wash out highlights / 可能让亮部变平 |
| `clahe-gamma` | Very dark but still detailed image / 很暗但仍有纹理 | Strongest grayscale enhancement / 灰度增强最强 | Highest noise/over-enhancement risk / 噪声和过增强风险最高 |
| `video enhance` (`v`) | Low-light live demo / 暗光实时演示 | Enhances BGR frame before detection, landmarks, and display / 在检测、关键点、显示前增强整帧 | If too dark/noisy, may still reduce stability / 极暗或高噪声下仍可能不稳定 |

## Implementation Logic / 实现原理与逻辑

中文：整体流程是“视频输入 -> 灰度预处理 -> 人脸检测 -> 人脸框筛选/稳定 -> 关键点拟合 -> 可视化输出”。基础版直接对灰度图运行 Haar 分类器，并用 LBF 模型拟合关键点；改进版在此基础上加入了边界 padding、候选框过滤、短时缓存、多脸检测和关键点平滑。

English: The overall pipeline is "video input -> grayscale preprocessing -> face detection -> face box filtering/stabilization -> landmark fitting -> visualization." The original version runs Haar detection on the grayscale frame and fits LBF landmarks directly. The improved version adds border padding, candidate filtering, short-term face caching, multi-face support, and landmark smoothing.

中文：Haar Cascade 通过滑动窗口和多尺度扫描寻找可能的人脸区域，返回 `(x, y, w, h)` 格式的人脸框。LBF Facemark 依赖这些人脸框进行局部特征回归，输出下颌、眉毛、鼻子、眼睛、嘴巴等区域的 68 个关键点。

English: The Haar Cascade searches for likely face regions with a sliding-window multi-scale scan and returns boxes in `(x, y, w, h)` format. The LBF Facemark model then performs local feature regression inside each face box and outputs 68 keypoints covering the jaw, eyebrows, nose, eyes, and mouth.

## Current Demo Keypoint Annotation / 当前 Demo 的关键点标注

中文：项目确实已经识别了 keypoints。现在 demo 中的关键点可视化进一步增强：68 个点会按脸部区域使用不同颜色显示，并对若干代表性关键点进行编号标注，例如下颌边缘、鼻梁/鼻尖、眼角、嘴角和下唇点。这比单纯绘制红点更容易展示“模型确实检测到了 facial landmarks”。

English: Yes, the project detects keypoints. The demo has now been enhanced so the 68 landmarks are color-coded by facial region, with representative indices labeled around the jaw, nose bridge/tip, eye corners, mouth corners, and lower lip. This makes the landmark detection result clearer than drawing plain red dots only.

中文：根据界面文字遮挡人脸的问题，HUD 显示已改为左上角小型半透明状态面板，只保留状态、FPS、人脸数量、框大小、预处理模式和失败帧数等必要信息。原来横跨画面的长提示文字和人脸框上方的姿态文字默认关闭，避免遮挡人脸、五官和关键点展示。

English: To solve the issue where on-screen text blocked the face, the HUD has been changed to a compact semi-transparent status panel in the top-left corner. It now only shows essential information such as state, FPS, face count, box size, preprocessing mode, and missed frames. The long full-width help text and pose text above the face box are disabled by default so they no longer cover the face, facial features, or keypoints.

中文：根据低光视频增强导致正常人脸无法稳定识别的问题，视频增强逻辑已调整。原因是之前增强帧被直接送入 Haar 和 LBF 检测链路，gamma、CLAHE 和锐化会在正常背光或眼镜反光场景中放大噪声，反而降低人脸框和 keypoint 的稳定性。现在 `ENHANCE_VIDEO_FRAME` 默认关闭；即使手动按 `v` 开启，也只影响显示画面，不再改变检测和关键点识别使用的原始帧。识别链路继续使用稳定的灰度/CLAHE 预处理。

English: To address the problem where low-light video enhancement made normal face detection unstable, the enhancement pipeline has been changed. Previously, the enhanced frame was passed directly into Haar and LBF detection; gamma, CLAHE, and sharpening could amplify noise and glasses reflections in normal backlit scenes, making face boxes and keypoints less stable. `ENHANCE_VIDEO_FRAME` is now off by default. Even when manually enabled with `v`, it only affects the displayed preview and no longer changes the original frame used for detection and landmark fitting. The recognition pipeline continues to use stable grayscale/CLAHE preprocessing.

中文：针对向右转头时容易把五官局部识别成小人脸的问题，新增了 sudden-shrink track rejection。系统会把当前候选框与上一帧稳定人脸框比较：如果候选框面积突然变得明显更小、中心仍落在上一帧人脸附近，并且有一定比例位于上一帧脸框内部，就认为它更可能是鼻子/嘴巴/眼睛区域的局部误检，而不是真实新脸。此时程序会拒绝这个小框，并短暂回退到上一帧稳定人脸框，避免右转时出现 `96x96` 之类的小框和错误 keypoints。

English: To fix the issue where right head turns could make the detector treat facial features as a small face, sudden-shrink track rejection has been added. The system compares the current candidate with the previous stable face box: if the candidate suddenly becomes much smaller, its center is still near the previous face, and part of it lies inside the previous box, it is treated as a local nose/mouth/eye false positive rather than a real new face. The small box is rejected and the demo briefly falls back to the previous stable face box, preventing `96x96`-style boxes and incorrect keypoints during right turns.

## Optimizations Already Made / 已经完成的优化

中文：第一，针对人脸框刷新过快和不稳定的问题，改进版加入了短时缓存机制。当当前帧检测失败时，程序会在少量帧内继续显示上一帧的人脸框，避免人脸框立刻消失造成闪烁。

English: First, to reduce unstable and rapidly flickering face boxes, the improved version uses a short-term cache. If detection fails in the current frame, the previous valid face box is displayed for a few frames instead of disappearing immediately.

中文：第二，针对俯仰头或左右转头时人脸框容易丢失的问题，改进版在人脸检测前给灰度图增加了边界 padding，并放宽/调整了检测参数。这可以缓解脸靠近画面边缘、姿态变化导致局部特征不完整的问题。

English: Second, to improve detection when the head tilts, pitches, or turns, the improved version pads the grayscale image before detection and tunes detection parameters. This helps when the face is near the frame edge or when pose changes make some facial features less clear.

中文：第三，针对初版只能处理一个人的问题，改进版现在默认支持多张人脸。检测器会保留所有通过面积、比例和误检过滤的人脸框，并对每张脸分别执行关键点拟合。若需要演示单人稳定跟踪，也可以通过 `--single-face` 参数切回单人模式。

English: Third, to address the original single-person limitation, the improved version now supports multiple faces by default. The detector keeps all boxes that pass area, aspect-ratio, and false-positive filters, then fits landmarks for each face. The `--single-face` option can still be used when a one-person tracking demo is preferred.

中文：第四，加入了候选框过滤逻辑。程序会过滤掉面积过小、面积过大、宽高比异常的候选框，从而减少把鼻子、嘴巴、局部阴影等误识别成人脸的情况。

English: Fourth, candidate filtering was added. Boxes that are too small, too large, or have abnormal aspect ratios are removed, reducing false positives where the nose, mouth, shadows, or partial regions are mistaken for faces.

中文：第五，加入了内部误检抑制。当检测结果突然缩小并且几乎完全落在上一帧人脸框内部时，程序会认为它更可能是脸部内部区域的误检，而不是新的人脸。

English: Fifth, inner false-positive suppression was added. If a new box suddenly shrinks and lies almost entirely inside the previous face box, it is treated as a likely internal facial-part false positive rather than a new face.

中文：第六，加入了关键点指数平滑。对于连续帧中位置相近的人脸，关键点坐标会与上一帧结果进行加权融合，从而减少眼角、嘴角等点位的抖动。

English: Sixth, landmark exponential smoothing was added. For faces that match across consecutive frames, current landmark coordinates are blended with the previous result, reducing jitter around points such as eye corners and mouth corners.

中文：第七，增加了 CLAHE 可选预处理。CLAHE 可以提升局部对比度，在光照不均匀或人脸偏暗时有机会改善检测效果。

English: Seventh, optional CLAHE preprocessing was added. CLAHE improves local contrast and can help detection under uneven lighting or when the face is relatively dark.

中文：第八，改进了可视化信息。界面显示 FPS、检测状态、原始候选数量、保留候选数量、人脸框大小、失败帧数和关键点说明，有助于调试和展示项目效果。

English: Eighth, visualization was improved. The interface now shows FPS, detection state, raw and filtered candidate counts, face box size, failed-frame count, and keypoint information, which helps both debugging and project presentation.

## Remaining Limitations / 仍然存在的限制

中文：当前系统仍然依赖 Haar Cascade 和 LBF 模型，因此在大角度侧脸、严重遮挡、强背光、快速运动、低分辨率和复杂背景下可能失败。LBF 关键点拟合也依赖可靠的人脸框，如果人脸框不准确，关键点会随之漂移。

English: The current system still depends on Haar Cascade and LBF models, so it may fail under large side poses, heavy occlusion, strong backlighting, fast motion, low resolution, or complex backgrounds. LBF landmark fitting also depends on accurate face boxes; if the box is inaccurate, the landmarks may drift.

## Future Improvements / 后续可以改进的地方

中文：后续可以用更鲁棒的深度学习检测器替代 Haar Cascade，例如 RetinaFace、MediaPipe Face Detection 或 YOLO-face，以增强侧脸、多人和复杂光照下的稳定性。

English: A future improvement would be replacing Haar Cascade with a more robust deep-learning detector, such as RetinaFace, MediaPipe Face Detection, or YOLO-face, to improve stability for side faces, multiple people, and difficult lighting.

中文：可以加入人脸 ID 跟踪，例如基于 IoU、Kalman Filter 或 SORT 的跟踪逻辑，让多人场景中每张脸保持稳定身份，而不是只按检测框顺序处理。

English: Face ID tracking could also be added, using IoU matching, a Kalman filter, or SORT-style tracking, so each person keeps a stable identity in multi-face scenes instead of being processed only by detection order.

中文：可以加入姿态估计，根据 68 个关键点计算头部 yaw、pitch、roll，并在界面上展示头部朝向。这能更直接地说明项目对俯仰和左右转头的处理效果。

English: Head pose estimation could be added by computing yaw, pitch, and roll from the 68 landmarks. Displaying these values would make the handling of nodding and left/right head turns more explicit.

中文：可以进一步优化性能，例如降低处理分辨率、隔帧检测、检测与关键点拟合分线程执行，或只在跟踪失败时重新运行完整检测。

English: Performance can be improved by processing at a lower resolution, detecting every few frames, running detection and landmark fitting in separate threads, or only re-running full detection when tracking fails.

中文：可以增加实验评估表，对不同光照、距离、遮挡、姿态、人数和背景复杂度分别记录成功率、FPS 和失败原因，让项目报告更有证据支撑。

English: An evaluation table could be added to record success rate, FPS, and failure cases under different lighting, distances, occlusions, poses, numbers of people, and background complexity. This would make the report more evidence-based.

## How To Run / 运行方式

```powershell
python .\beginner_level\main.py
python .\beginner_level\main.py --clahe
python .\beginner_level\main.py --preprocess clahe-gamma
python .\beginner_level\main.py --single-face
python .\beginner_level\main.py --keep-mirror
```

Runtime keys / 运行时按键：

```text
1 raw
2 clahe
3 gamma
4 clahe-gamma
v toggle video enhancement
m toggle mirror correction
r reset detection cache
q quit
```
