你现在只负责为当前 Visual Computing 项目搭建一个全新的、独立的 Conda 环境。

不要使用我之前的 sws2026 环境，因为它正在被另一个项目使用，我不希望不同项目之间发生依赖冲突。

我的电脑环境：

- 操作系统：Windows
- 开发工具：VS Code
- Miniconda 安装位置：D:\miniconda3
- 当前项目需要一个全新的独立环境
- 新环境名称统一使用：vc_sws3026
- 建议 Python 版本：3.11

本次任务仅限于：

1. 创建新的 Conda 环境
2. 安装本项目所需 Python 库
3. 检查 OpenCV 的 cv2.face 是否可用
4. 验证所有依赖能否正常导入
5. 输出环境搭建结果

不要执行项目代码，不要打开摄像头，不要处理数据集，不要运行模型推理，不要修改老师提供的任何文件。

==================================================
一、操作安全要求
==================================================

必须遵守以下规则：

1. 不使用 sws2026 环境。
2. 不向 Conda base 环境安装项目依赖。
3. 不删除任何已有 Conda 环境。
4. 不修改 Windows 系统 PATH。
5. 不修改 PowerShell 执行策略。
6. 不使用管理员权限。
7. 不修改当前项目中的 Python 文件。
8. 不打开摄像头。
9. 不运行 starter.py 或 danceapp.py。
10. 不下载额外的预训练模型。
11. 每执行一个命令前，用一句中文说明命令的作用。
12. 如果命令失败，先分析错误，不要连续重复执行相同命令。

==================================================
二、检查 Conda 是否可用
==================================================

先执行：

conda --version
conda info --envs

检查环境列表中是否已经存在：

vc_sws3026

处理规则：

- 如果不存在，则继续创建。
- 如果已经存在，不要删除或覆盖它。
- 如果已存在，请先检查它的 Python 路径和依赖状态。
- 如果它明显是其他项目使用的环境，请停止并报告，不要自行删除。
- 绝对不要修改 sws2026。

==================================================
三、创建全新的 Conda 环境
==================================================

如果 vc_sws3026 不存在，执行：

conda create -n vc_sws3026 python=3.11 pip tk -y

创建完成后执行：

conda info --envs

确认新环境已经出现。

新环境的 Python 路径应当是：

D:\miniconda3\envs\vc_sws3026\python.exe

不要依赖 PowerShell 中的 conda activate 是否成功。

后续优先使用 Python 的绝对路径执行所有命令：

$PY = "D:\miniconda3\envs\vc_sws3026\python.exe"

然后验证：

Test-Path $PY
& $PY --version
& $PY -c "import sys; print(sys.executable)"
& $PY -m pip --version

必须确认：

- Python 路径属于 vc_sws3026
- pip 也属于 vc_sws3026
- 没有使用 base
- 没有使用 sws2026

如果 Python 路径不正确，立即停止，不要安装依赖。

==================================================
四、升级基础安装工具
==================================================

只在 vc_sws3026 环境中执行：

& $PY -m pip install --upgrade pip setuptools wheel

完成后输出：

& $PY -m pip --version

==================================================
五、安装项目依赖
==================================================

本项目需要以下主要依赖：

- numpy
- opencv-contrib-python
- scikit-learn
- matplotlib
- pillow
- ultralytics
- joblib

其中必须使用：

opencv-contrib-python

原因是项目需要：

cv2.face.createFacemarkLBF()

不要主动安装 headless 版本，因为本项目需要 OpenCV 图形窗口和摄像头功能。

禁止主动安装：

- opencv-python-headless
- opencv-contrib-python-headless
- mediapipe
- librosa
- madmom
- jupyter
- tensorflow
- CUDA
- cuDNN

先安装除 OpenCV 之外的依赖：

& $PY -m pip install numpy scikit-learn matplotlib pillow joblib ultralytics

然后检查当前是否已经因为 ultralytics 安装了 OpenCV：

& $PY -m pip list | findstr /I "opencv"

再安装或补充 OpenCV contrib：

& $PY -m pip install opencv-contrib-python

不要在没有检查的情况下反复卸载和重装 OpenCV。

==================================================
六、检查 OpenCV 状态
==================================================

执行：

& $PY -m pip list | findstr /I "opencv"

记录安装了哪些 OpenCV 包及版本。

然后执行：

& $PY -c "import cv2; print('OpenCV version:', cv2.__version__); print('cv2.face exists:', hasattr(cv2, 'face')); print('cv2.imshow exists:', hasattr(cv2, 'imshow'))"

必须验证：

1. cv2 可以导入。
2. cv2.face 存在。
3. cv2.imshow 存在。

继续验证 LBF 接口：

& $PY -c "import cv2; detector = cv2.face.createFacemarkLBF(); print('createFacemarkLBF: PASS')"

验收标准：

- cv2.face exists 为 True
- cv2.imshow exists 为 True
- createFacemarkLBF 创建成功

如果 cv2.face 不存在：

1. 先输出当前所有 OpenCV 包。
2. 分析是否存在标准版、contrib 版和 headless 版冲突。
3. 不要盲目卸载。
4. 只处理确认存在冲突的 OpenCV 包。
5. 最终必须保留一个能够提供 cv2.face 和 cv2.imshow 的非 headless OpenCV 安装。
6. 不要改动其他环境。

==================================================
七、检查所有依赖
==================================================

逐项执行：

& $PY -c "import numpy; print('numpy:', numpy.__version__)"
& $PY -c "import cv2; print('opencv:', cv2.__version__)"
& $PY -c "import sklearn; print('scikit-learn:', sklearn.__version__)"
& $PY -c "import matplotlib; print('matplotlib:', matplotlib.__version__)"
& $PY -c "import PIL; print('pillow:', PIL.__version__)"
& $PY -c "import ultralytics; print('ultralytics:', ultralytics.__version__)"
& $PY -c "import joblib; print('joblib:', joblib.__version__)"
& $PY -c "import tkinter; print('tkinter:', tkinter.TkVersion)"
& $PY -c "import torch; print('torch:', torch.__version__); print('CUDA available:', torch.cuda.is_available())"

注意：

- CUDA available 返回 False 不代表失败。
- 本项目允许使用 CPU。
- 不要安装 CUDA 版本的 PyTorch。
- Ultralytics 自动安装的 PyTorch 可以保留。

==================================================
八、检查依赖一致性
==================================================

执行：

& $PY -m pip check

如果出现问题：

1. 列出具体冲突。
2. 判断是否会影响本项目。
3. 只修复 vc_sws3026 环境。
4. 不修改其他 Conda 环境。
5. 不为了追求最新版本而升级全部软件包。

然后输出完整安装列表：

& $PY -m pip list

==================================================
九、保存环境记录
==================================================

在当前工作区中新建文件夹：

environment_setup

只允许创建环境记录文件，不要修改老师代码。

保存完整版本记录：

& $PY -m pip freeze | Out-File -Encoding utf8 "environment_setup\requirements-lock.txt"

使用 Conda 导出环境：

conda env export -n vc_sws3026 --from-history | Out-File -Encoding utf8 "environment_setup\environment.yml"

再创建：

environment_setup\environment_summary.txt

内容应包含：

- 环境名称
- Python 完整路径
- Python 版本
- pip 版本
- OpenCV 版本
- cv2.face 是否可用
- NumPy 版本
- scikit-learn 版本
- Matplotlib 版本
- Pillow 版本
- Ultralytics 版本
- PyTorch 版本
- Tkinter 是否可用
- CUDA 是否可用
- pip check 结果

不要创建项目代码文件。

==================================================
十、最终报告
==================================================

完成后，用中文按照下面格式输出：

A. 新环境信息

- 环境名称：
- 环境路径：
- Python 路径：
- Python 版本：
- pip 路径：
- 是否确认未使用 sws2026：
- 是否确认未向 base 安装依赖：

B. 依赖状态

使用表格：

依赖 | 是否安装成功 | 版本 | 验证结果 | 备注

至少包括：

- numpy
- opencv
- cv2.face
- scikit-learn
- matplotlib
- pillow
- ultralytics
- joblib
- tkinter
- torch

C. OpenCV 检查

- 当前安装的 OpenCV 包：
- OpenCV 版本：
- cv2.face 是否存在：
- cv2.imshow 是否存在：
- createFacemarkLBF 是否成功：
- 是否发现 headless 包：
- 是否发现潜在 OpenCV 冲突：

D. 环境健康状态

- pip check 结果：
- CPU 是否可以运行项目：
- CUDA 是否可用：
- CUDA 是否为项目必要条件：

E. 创建的环境记录

- environment_setup\requirements-lock.txt
- environment_setup\environment.yml
- environment_setup\environment_summary.txt

F. 最终结论

明确回答：

1. vc_sws3026 是否创建成功。
2. 项目主要依赖是否安装完成。
3. 人脸关键点所需的 cv2.face 是否可用。
4. Ultralytics 是否可用。
5. 是否已经可以进入后续项目开发。

完成环境搭建和报告后立即停止。

禁止继续执行：

- 摄像头测试
- 人脸检测
- LBF 模型加载
- YOLO 模型推理
- 视频播放
- 数据集解压
- 表情分类训练
- 修改 starter.py
- 修改 danceapp.py
- 编写项目功能代码