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

中文：目录结构也已同步调整：原来的 `beginner_level/improved` 版本已经替换到 `beginner_level` 根目录中，因此现在直接运行 `VisualComputingProject\beginner_level\main.py` 就是改进后的版本，不再需要进入单独的 `improved` 子目录。

English: The folder structure has also been synchronized: the previous `beginner_level/improved` version has been moved into the `beginner_level` root. Therefore, running `VisualComputingProject\beginner_level\main.py` now starts the improved version directly, without using a separate `improved` subfolder.

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

中文：本次根据 `VisualComputingProject/expert_level/src` 的实时检测方式继续优化 Beginner Level。Beginner Level 现在不再每帧都完全依赖全图 Haar 检测，而是参考 expert demo 使用“ROI 局部搜索为主、定期全图检测校准”的策略：上一帧稳定人脸框会作为下一帧的局部搜索区域，每隔 `FULL_DETECT_INTERVAL` 帧再进行一次全图扫描。这样可以减少人脸框不断刷新造成的跳动，同时在转头、俯仰头和短时检测失败时保持更好的连续性。

English: Based on the realtime recognition pipeline in `VisualComputingProject/expert_level/src`, the Beginner Level detector has been optimized further. It no longer relies on full-frame Haar detection every frame. Instead, it follows the expert demo's strategy of using ROI local search as the primary path and full-frame detection as periodic correction: the previous stable face box is used as the next frame's local search region, while a full-frame scan is performed every `FULL_DETECT_INTERVAL` frames. This reduces box refresh jitter and improves continuity during head turns, pitch changes, and short detection failures.

中文：预处理也参考 expert 版本进行了增强。现在支持 `raw`、`clahe`、`gamma`、`clahe-gamma` 四种实时预处理模式，默认使用 `clahe`，以提升暗光、局部阴影和对比度不足情况下的人脸与五官关键点识别效果。运行时可以通过 `--preprocess raw|clahe|gamma|clahe-gamma` 切换。

English: Preprocessing has also been strengthened based on the expert version. The demo now supports four realtime preprocessing modes: `raw`, `clahe`, `gamma`, and `clahe-gamma`, with `clahe` as the default. This improves face and facial-feature keypoint detection under dim lighting, local shadows, and low contrast. The mode can be changed with `--preprocess raw|clahe|gamma|clahe-gamma`.

中文：68 个 keypoint 的稳定性也进一步参考 expert 版本优化。原来的简单整体平滑改为“分区域平滑”：下颌线、眉毛/鼻子、眼睛/嘴巴使用不同的平滑系数；同时限制单个关键点每帧最大跳变，并约束下颌线不要漂出人脸框过远。显示层还增加了额外平滑，使五官 keypoint 在视频中更稳定。

English: The 68-keypoint stability has also been improved using the expert version as a reference. The previous simple global smoothing has been replaced with region-aware smoothing: the jawline, brows/nose, and eyes/mouth use different smoothing factors. Each landmark's maximum per-frame jump is limited, and the jawline is constrained so it does not drift too far outside the face box. An additional display-only smoothing layer makes the visible keypoints more stable in the video.

中文：对于转头识别，系统继续使用基于 68 点的 yaw、pitch、roll 姿态估计，并参考 expert demo 增加了 pose-aware 的下颌轮廓显示修正。当用户左右转头时，LBF 模型有时会保留偏正脸的下颌模板，导致轮廓漂浮；现在显示层会根据 yaw 角度适度收缩和上移下颌轮廓，使侧脸时的关键点显示更贴近实际可见脸部。

English: For head-turn recognition, the system continues to estimate yaw, pitch, and roll from the 68 landmarks, and now adds the expert demo's pose-aware jawline display adjustment. When the user turns left or right, LBF may keep a frontal jaw template and make the contour float away from the visible face. The display layer now narrows and slightly lifts the jawline according to yaw, making side-face landmark visualization closer to the actual visible face.

## Implementation Logic / 实现原理与逻辑

中文：整体流程是“视频输入 -> 灰度预处理 -> 人脸检测 -> 人脸框筛选/稳定 -> 关键点拟合 -> 可视化输出”。基础版直接对灰度图运行 Haar 分类器，并用 LBF 模型拟合关键点；改进版在此基础上加入了边界 padding、候选框过滤、短时缓存、多脸检测和关键点平滑。

English: The overall pipeline is "video input -> grayscale preprocessing -> face detection -> face box filtering/stabilization -> landmark fitting -> visualization." The original version runs Haar detection on the grayscale frame and fits LBF landmarks directly. The improved version adds border padding, candidate filtering, short-term face caching, multi-face support, and landmark smoothing.

中文：Haar Cascade 通过滑动窗口和多尺度扫描寻找可能的人脸区域，返回 `(x, y, w, h)` 格式的人脸框。LBF Facemark 依赖这些人脸框进行局部特征回归，输出下颌、眉毛、鼻子、眼睛、嘴巴等区域的 68 个关键点。

English: The Haar Cascade searches for likely face regions with a sliding-window multi-scale scan and returns boxes in `(x, y, w, h)` format. The LBF Facemark model then performs local feature regression inside each face box and outputs 68 keypoints covering the jaw, eyebrows, nose, eyes, and mouth.

## Current Demo Keypoint Annotation / 当前 Demo 的关键点标注

中文：项目确实已经识别了 keypoints。现在 demo 中的关键点可视化进一步增强：68 个点会按脸部区域使用不同颜色显示，并对若干代表性关键点进行编号标注，例如下颌边缘、鼻梁/鼻尖、眼角、嘴角和下唇点。这比单纯绘制红点更容易展示“模型确实检测到了 facial landmarks”。

English: Yes, the project detects keypoints. The demo has now been enhanced so the 68 landmarks are color-coded by facial region, with representative indices labeled around the jaw, nose bridge/tip, eye corners, mouth corners, and lower lip. This makes the landmark detection result clearer than drawing plain red dots only.

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
python .\VisualComputingProject\beginner_level\main.py
python .\VisualComputingProject\beginner_level\main.py --clahe
python .\VisualComputingProject\beginner_level\main.py --preprocess clahe-gamma
python .\VisualComputingProject\beginner_level\main.py --single-face
python .\VisualComputingProject\beginner_level\main.py --keep-mirror
```
