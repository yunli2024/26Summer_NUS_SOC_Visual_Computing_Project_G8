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
python .\VisualComputingProject\beginner_level\improved\main.py
python .\VisualComputingProject\beginner_level\improved\main.py --clahe
python .\VisualComputingProject\beginner_level\improved\main.py --single-face
```
