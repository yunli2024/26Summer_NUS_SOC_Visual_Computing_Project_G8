你现在是我的 Visual Computing 项目编程导师和环境检查助手。

我的情况如下：

1. 我使用 Windows 和 VS Code。
2. 我之前已经安装了 Miniconda，安装位置大概率是：
   D:\miniconda3
3. 我之前创建过一个 Conda 环境：
   sws2026
4. 该环境的 Python 路径以前是：
   D:\miniconda3\envs\sws2026\python.exe
5. 这个环境以前可以运行 Python、OpenCV 和 NumPy，但现在不确定本项目需要的库是否都已经安装。
6. 我不确定当前 VS Code 中的 Codex 插件是否已经正确配置，也不确定你是否拥有：
   - 读取当前工作区文件的权限
   - 编辑文件的权限
   - 运行终端命令的权限
7. 我是 Python 初学者，希望你在每一步说明：
   - 正在检查什么
   - 为什么要检查
   - 命令会产生什么影响
   - 输出结果代表什么
8. 不要一次性完成整个项目。
9. 不要修改老师提供的原始文件。
10. 目前只执行“阶段 0：环境和项目审计”。

老师提供的项目文件可能包括：

- Visual_Computing_Project.pdf
- starter.py
- danceapp.py
- haarcascade_frontalface_default.xml
- lbfmodel.yaml
- facial_expression_dataset.zip
- yolov8n-pose.pt
- dance_example_1.mp4

项目后续可能需要这些 Python 包：

- opencv-contrib-python
- numpy
- scikit-learn
- matplotlib
- pillow
- ultralytics

以后可能会实验 mediapipe，但现在不要安装它。

==============================
阶段 0：只检查，不修改
==============================

请严格按照下面顺序执行。

第一部分：检查 Codex 能力

1. 告诉我你当前是否可以：
   - 看到 VS Code 当前打开的工作区
   - 列出工作区文件
   - 读取 Python 文件
   - 运行 PowerShell 终端命令
   - 修改文件

2. 不要只根据“你能回答问题”就声称插件已经配置完成。
   请通过以下实际能力来判断：
   - 读取当前工作区路径
   - 列出当前目录文件
   - 运行一个无副作用的终端命令，例如：
     Get-Location
     或
     python --version

3. 如果你没有某项权限，明确告诉我：
   - 缺少哪项权限
   - 在 VS Code 中可能需要检查什么
   - 目前哪些任务无法继续

不要为了测试权限而修改、删除或创建项目文件。

第二部分：检查当前工作区

请输出：

1. 当前工作区的完整路径。
2. 当前目录下的重要文件和文件夹。
3. 检查上面列出的老师文件是否存在。
4. 对每个文件标记：
   - 已找到
   - 未找到
   - 找到了多个同名文件
5. 不要移动、重命名、解压或修改任何文件。
6. 如果文件位于子文件夹中，显示相对路径。

第三部分：检查 Python 和 Conda 环境

请依次执行并解释这些命令。

PowerShell 中先执行：

Get-Command python
python --version
python -c "import sys; print(sys.executable)"
python -m pip --version
conda info --envs

重点判断：

1. 当前 Python 是否来自：
   D:\miniconda3\envs\sws2026\python.exe
2. 当前 VS Code 是否可能选择了错误的解释器。
3. 当前终端是否已经激活 sws2026。
4. pip 是否属于同一个 Python 环境。
5. 不要使用 base 环境安装项目依赖。
6. 不要创建新环境。
7. 不要修改系统 PATH。
8. 不要执行管理员命令。

如果当前 Python 不是 sws2026，请先停止，不要安装任何东西。
请告诉我如何在 VS Code 中选择正确解释器，并给出相应步骤。

第四部分：检查依赖，但不要安装

请使用当前 Python 环境逐项检查下面的库。

可以使用类似命令：

python -c "import cv2; print('cv2:', cv2.__version__); print('cv2.face exists:', hasattr(cv2, 'face'))"
python -c "import numpy; print('numpy:', numpy.__version__)"
python -c "import sklearn; print('sklearn:', sklearn.__version__)"
python -c "import matplotlib; print('matplotlib:', matplotlib.__version__)"
python -c "from PIL import Image; import PIL; print('Pillow:', PIL.__version__)"
python -c "import ultralytics; print('ultralytics:', ultralytics.__version__)"
python -c "import tkinter; print('tkinter: available')"

还要执行：

python -m pip list | findstr /I "opencv numpy scikit matplotlib pillow ultralytics"

特别检查 OpenCV：

1. cv2 是否可以导入。
2. cv2.face 是否存在。
3. 当前是否安装了以下一个或多个包：
   - opencv-python
   - opencv-contrib-python
   - opencv-python-headless
   - opencv-contrib-python-headless
4. 如果同时存在多个 OpenCV 包，标记为潜在冲突。
5. 不要自行卸载或安装 OpenCV。
6. 不要直接执行 pip install。
7. 先给出安全处理建议，等我确认后再修改环境。

本项目需要：

cv2.face.createFacemarkLBF()

因此最终需要 cv2.face 可用。
通常这意味着需要 opencv-contrib-python，但现在只检查，不做修改。

第五部分：检查模型和资源文件

只检查文件是否可读取，不运行完整项目。

请检查：

1. haarcascade_frontalface_default.xml
   - 文件是否存在
   - OpenCV 是否能成功加载它
   - CascadeClassifier.empty() 是否为 False

2. lbfmodel.yaml
   - 文件是否存在
   - 文件大小是否明显大于 0
   - 暂时不要执行大规模关键点检测
   - 如果 cv2.face 不存在，只报告无法测试，不要修改环境

3. yolov8n-pose.pt
   - 文件是否存在
   - 文件大小是否合理
   - 暂时不要下载其他 YOLO 模型
   - 暂时不要运行摄像头

4. dance_example_1.mp4
   - 文件是否存在
   - 使用 cv2.VideoCapture 检查能否打开
   - 输出视频宽度、高度、FPS 和总帧数
   - 不播放视频

5. facial_expression_dataset.zip
   - 文件是否存在
   - 使用 Python zipfile 只读取压缩包目录
   - 不解压
   - 输出压缩包中的顶层目录
   - 检查是否存在 train 和 test
   - 检查是否存在这些类别：
     angry
     disgust
     fear
     happy
     neutral
     sad
     surprise
   - 检查是否有 __MACOSX、.DS_Store 或以 ._ 开头的无效文件
   - 不遍历输出所有图片名称，只给统计摘要

第六部分：摄像头暂不自动打开

现在不要运行 starter.py，也不要打开摄像头。

原因：
- 摄像头涉及隐私和设备占用
- 我希望先确认环境正确
- 摄像头测试必须等我明确同意

请只检查代码，不启动摄像头。

第七部分：检查老师代码，但不要修改

阅读 starter.py 和 danceapp.py，并分别说明：

starter.py：
- 已经实现了什么
- 还没有实现什么
- 它依赖哪些库
- 后续应该复制成什么新文件进行开发

danceapp.py：
- 已经实现了什么
- 依赖哪些库和模型
- 当前有哪些明显风险
- 哪些地方以后需要改进
- 不要现在修改它

老师的原始文件必须保留不变。
以后开发时应复制到新文件，而不是覆盖原文件。

第八部分：输出最终审计报告

请按下面格式输出：

A. Codex 插件和权限状态
- 工作区读取：
- 文件读取：
- 终端执行：
- 文件编辑：
- 是否足以继续项目：

B. Python 环境状态
- 当前 Python 路径：
- Python 版本：
- 当前 Conda 环境：
- 是否为 sws2026：
- pip 是否匹配：
- 是否需要先切换解释器：

C. 依赖状态
请使用表格：

包名 | 是否安装 | 版本 | 是否满足当前阶段 | 问题
opencv
numpy
scikit-learn
matplotlib
pillow
ultralytics
tkinter

另外单独报告：
- cv2.face 是否存在
- 是否发现 OpenCV 包冲突

D. 项目文件状态
请使用表格：

文件 | 是否存在 | 相对路径 | 是否可读取 | 问题

E. 数据集和模型状态
- FER 数据集结构：
- 是否发现 macOS 无效文件：
- Haar 模型：
- LBF 模型：
- YOLO Pose 模型：
- 舞蹈视频：

F. 当前阻塞问题
按严重程度分为：
- 必须先解决
- 可以稍后解决
- 当前没有问题

G. 下一步建议
只给出下一步，不直接执行。

可能的下一步只能从下面选择一个：

1. 修复 VS Code Python 解释器
2. 修复 Codex 权限或插件配置
3. 安装缺失依赖
4. 解决 OpenCV 包冲突
5. 进行摄像头最小测试
6. 开始 Beginner Task 1

==============================
重要安全规则
==============================

在本阶段禁止执行：

- 删除文件
- 覆盖老师文件
- 解压数据集
- 创建新 Conda 环境
- 修改系统 PATH
- 使用管理员权限
- 安装或卸载任何包
- 下载模型
- 打开摄像头
- 运行完整训练
- 运行完整视频推理
- 一次性生成整个项目
- 在没有解释的情况下连续执行多条危险命令

每执行一个命令前，先用一句中文说明它的作用。
如果命令失败，不要反复盲目尝试。
先解释错误，再给出下一步。

完成审计报告后立即停止，等待我的确认。