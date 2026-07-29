# AGENTS.md

本项目是 NUS SOC SWS3026 Visual Computing group project，主题为
`Real-time Video Analysis and Rendering`。后续 agent 必须先阅读本文件，再阅读
`docs/course_materials/Visual_Computing_Project.pdf` 和对应难度目录中的 starter code / dataset / model。

## 工作语言

- 默认用中文和用户沟通。
- 代码、变量名、文件名、图表标题可以使用英文。
- 给用户的实现说明要直接、可执行，避免空泛计划。

## 硬性限制

- 本项目必须在本地运行，不能依赖 Google Colab 或 Jupyter Notebook，因为需要访问 laptop webcam。
- 不要假设有高性能 GPU。项目说明明确说不强制使用 deep network；如果使用深度模型，必须保证用户电脑能训练和实时运行。
- 实时 webcam/demo 是核心交付，不要只做离线 notebook 或静态分析。
- 表情分类必须使用 facial keypoints 作为输入，不要直接把完整人脸图像喂给分类器来替代项目要求。
- Expert level 的实时预测目标是每次预测小于 30 ms，以支持约 30 FPS。
- Bonus level 的 TikTok dataset 主要用于探索和改进 body movement detection pipeline，不要求用它训练机器学习或深度学习模型。
- 不要批量删除文件或目录。禁止使用 `del /s`、`rd /s`、`rmdir /s`、`Remove-Item -Recurse`、`rm -rf`。如需删除文件，只能一次删除一个明确路径的文件；如需批量删除，停止并让用户手动处理。

## 当前资料结构

- `docs/course_materials/Visual_Computing_Project.pdf`：项目总说明，所有实现必须与此对齐。
- `beginner_level/`：Haar + LBF 68 点实时检测。
- `expert_level/`：keypoint-only 表情分类、PCA/validation
  训练、实时跟踪和特效。
- `bonus_level/`：Just Dance 双面板、时空对齐与评分。
- `bonus_level_mario/`：Zhangyx 的姿态控制平台游戏。
- `resources/face_models/`：Haar、LBF 与 YuNet。
- `resources/expression_data/facial_expression_dataset.zip`：
  完整 FER-style `train`/`test` 七类数据。
- `resources/pose_models/yolov8n-pose.pt`：YOLOv8
  nano pose 模型。
- `resources/videos/`：参考舞蹈视频。
- `docs/`：课程说明、展示材料、开发记录和原始项目资料。

## 推荐项目目标

优先做成一个可以现场展示的系统，而不是零散脚本。最小完整路线：

1. Beginner：实时 webcam 人脸框 + 68 facial landmarks。
2. Expert：基于 landmarks 的 facial expression classifier + webcam 表情文字/特效。
3. Bonus：reference video 与 webcam 双 panel pose detection + 简单动作相似度评分和屏幕反馈。

每一层都要能解释方法、失败情况、改进策略和 demo 效果。

## Beginner Level 要求

目标：实现实时 facial keypoints detection，并讨论 face detector 鲁棒性。

必须完成：

- 从 `beginner/starter.py` 出发，使用 `cv2.VideoCapture(0)` 打开 webcam。
- 使用 `cv2.CascadeClassifier("haarcascade_frontalface_default.xml")` 检测每帧人脸。
- 对检测到的人脸画 bounding box。
- 使用 OpenCV facemark LBF：
  - `cv2.face.createFacemarkLBF()`
  - `facemark.loadModel("lbfmodel.yaml")`
  - 预测并绘制 68 个 facial landmarks。
- 在实时视频窗口中显示检测结果。
- 支持按键退出，并正确 `release()` camera、`destroyAllWindows()`。

必须分析：

- 光照变化、头部角度、部分遮挡时的表现。
- Haar cascade + LBF 的失败原因。
- 至少提出一种改进方案，例如 preprocessing、调 detector 参数，或尝试 Dlib / MediaPipe 等替代检测器。
- 项目文档鼓励至少实验一个替代 face detector，并解释为什么可能更好。

最终效果：

- 摄像头画面中能实时显示 face bounding box 和 facial landmarks。
- 能展示成功案例和若干失败/鲁棒性观察。

## Expert Level 要求

目标：使用 facial keypoints 而不是 full image 做实时 facial expression classification，并基于表情设计 video effects。

数据准备：

- 使用 `expert/facial_expression_dataset.zip` 中的 FER-style 数据。
- 训练/测试划分应使用数据集已有 `train` / `test` 目录；需要 validation 时再从 train 内部分出。
- 当前已解压目录不完整，必须先确认七类数据都可访问。

Task 1 必须完成：

- 对每张表情图像运行 Beginner/Part I 的 facial keypoint detection pipeline。
- 从 landmarks 构造分类输入。推荐至少做以下预处理之一：
  - 按人脸框归一化 x/y 坐标。
  - 减去中心点并按脸宽/脸高缩放。
  - 构造相对距离、角度或关键区域几何特征。
  - 将 keypoints 画成白底黑点图再训练轻量模型。
- 训练分类器。可选 SVM、Random Forest、MLP、轻量 CNN 等，但必须满足实时预测约束。
- 使用 test set 评估并报告：
  - accuracy 或 macro F1。
  - confusion matrix。
  - failure cases。
  - 遇到的挑战和改进策略。
- 合并后的统一训练在 train split 内划分 stratified validation，并将 PCA
  作为候选 pipeline 的一部分；test split 只用于最终一次评估。
- 选模以 Macro-F1、类别平衡指标和实时延迟为主，accuracy
  只作为参考，不能作为项目“得分”或唯一结论。

Task 2 必须完成：

- 将表情分类器接入 webcam pipeline。
- 在实时画面上显示预测表情。最低要求可以用 `cv2.putText`。
- 设计至少一种表情驱动 video effect，例如 smile 触发 sparkle overlay、surprise 触发 stars、neutral 显示普通状态等。
- 特效应随表情实时变化，尽量保持画面流畅。

最终效果：

- webcam 实时检测 face landmarks。
- classifier 根据 landmarks 输出表情类别。
- 屏幕上显示预测结果，并触发对应 visual effect。

## Bonus Level 要求

目标：从 face keypoint 扩展到 full/partial body movement detection，并做一个 Just Dance 类互动应用。

Task 1 必须完成：

- 从 `bonus/danceapp.py` 出发运行 GUI。
- 先使用左侧 reference video panel。
- 对 reference video 做 body keypoint detection 并可视化 skeleton。
- 可使用现有 `yolov8n-pose.pt` 和 Ultralytics YOLO pose pipeline。
- 分析 keypoint detection 的挑战，例如：
  - 多人场景下如何确定主舞者。
  - 遮挡、快速动作、低分辨率、出画时如何处理。
  - partial body visible 时如何保持合理输出。
- 不需要评估整个 dataset，可以挑几个例子测试并在 presentation 中报告发现。

Task 2 必须完成：

- 同时对 reference video 和 webcam input 运行 pose detection。
- GUI 左右 panel 都显示 body movement / skeleton。
- 设计自己的 scoring system 比较用户动作和 reference dancer。
- 评分系统必须解释空间与时间对齐：
  - 空间：可按 torso/hip/shoulder 归一化、平移到共同中心、按身体尺度缩放。
  - 时间：考虑 webcam 动作相对 reference video 的 lag，可用固定延迟、滑动窗口或动态匹配。
  - 相似度：可基于关键点欧氏距离、cosine similarity、关节角度差、可见关键点加权平均等。
- 在屏幕显示反馈，例如 `Perfect`、`Super`、`Good`、score、combo 等。
- 可以加入额外互动玩法，例如 webcam 中弹出目标并让玩家用手部关键点触碰得分。

最终效果：

- 一个可展示的 Just Dance-like app：左侧参考舞蹈，右侧 webcam，双方都有 pose skeleton。
- 系统实时给出动作匹配分数和反馈。
- presentation 中能清楚解释所用 keypoint detector、alignment 方法、similarity metric 和反馈机制。

## 实现建议

- 优先保证 demo 可运行，再逐步优化准确率和美观。
- 复用 starter code，但应把可复用逻辑拆成清晰模块，例如：
  - face detection / landmark extraction
  - feature normalization
  - classifier training / saving / loading
  - webcam inference
  - video effects
  - pose detection
  - pose scoring
- 模型和数据路径使用相对路径，并在脚本启动时检查文件是否存在，报错信息要清楚。
- 对 webcam 脚本提供 `q` 或 GUI Stop 按钮退出。
- 不要在每帧里重复加载模型；模型必须在循环外加载。
- 对实时 pipeline 记录或显示 FPS，至少在开发阶段确认速度。
- 如使用 OpenCV LBF，注意 `cv2.face` 通常需要 `opencv-contrib-python`，不是普通 `opencv-python`。若运行时报 `cv2 has no attribute face`，应安装/切换到 `opencv-contrib-python`。

## 验证清单

提交或汇报前，至少验证：

- 环境能本地启动，camera 能被 `cv2.VideoCapture(0)` 打开。
- Beginner：实时窗口中出现 face box 和 68 landmarks。
- Beginner：至少记录一种失败场景和一种改进方向。
- Expert：完整七类数据可访问，训练/测试划分清楚。
- Expert：分类器输入是 keypoints 或 keypoint-derived representation，不是 full image。
- Expert：输出 confusion matrix 和 failure cases。
- Expert：webcam 中实时显示表情预测和至少一个 visual effect。
- Expert：单次 prediction 接近或优于 30 ms 目标。
- Bonus：`danceapp.py` 或改造版 GUI 能打开 reference video 和 webcam。
- Bonus：左右 panel 都能显示 pose/skeleton。
- Bonus：屏幕显示 score/feedback，且 scoring metric 可以解释。

## Presentation 重点

- Beginner：展示实时 facial landmarks，说明鲁棒性测试和替代 detector/改进思路。
- Expert：说明为什么用 keypoints 表征表情，展示 confusion matrix、失败案例、实时表情特效。
- Bonus：展示双 panel dance app，说明主舞者选择、pose alignment、temporal lag 处理和 scoring metric。
- 不要只报结果，要解释观察到的问题以及如何解决。

## 与用户协作方式

- 用户通常需要 actionable guidance。遇到问题时先给可执行命令、检查点和下一步，不要只讲概念。
- 如果需要下载额外模型或安装包，先说明目的、大小/风险和替代方案。
- 如果当前资料缺失或不完整，明确指出，不要编造数据或结果。
- 对 webcam、GUI、实时 FPS 相关问题，优先让用户本地运行并反馈报错/截图；agent 不能假设自己的环境有可用摄像头。
