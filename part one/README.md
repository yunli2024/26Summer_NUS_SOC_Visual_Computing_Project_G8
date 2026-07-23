# Part One: Real-time Facial Landmarks

This program uses a Haar cascade to find faces and OpenCV's LBF model to locate 68 facial landmarks in a live webcam feed.

## Setup

Use a local terminal (not Colab or Jupyter). From this directory, run:

```powershell
python -m pip uninstall -y opencv-python opencv-python-headless
python -m pip install -r requirements.txt
```

Only one OpenCV wheel should be installed. `opencv-contrib-python` is required because the standard `opencv-python` package does not contain `cv2.face`.
The project pins OpenCV 4.12 because the initial OpenCV 5 Windows wheel may not expose the MSMF camera backend on every machine.

## Run

```powershell
python starter.py
```

Controls:

- `I`: switch CLAHE local contrast enhancement on/off
- `S`: switch EMA landmark smoothing on/off
- `Q` or `Esc`: quit

CLAHE and smoothing start enabled. Their live switches make it easy to capture before/after evidence for the robustness experiment. If the default webcam is unavailable, try:

```powershell
python starter.py --camera 1
```

Useful options can be listed with `python starter.py --help`.
