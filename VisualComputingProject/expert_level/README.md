# Expert Level: ROI Ensemble

This folder contains the final inference-only facial-expression system. Training
experiments and weaker model branches have been removed.

## Final model

The retained model is a weighted probability ensemble:

```text
0.45 * base ROI-CNN + 0.55 * robust ROI-CNN
```

Both networks use the `face_eyes_mouth` variant and predict these seven classes:
`angry`, `disgust`, `fear`, `happy`, `neutral`, `sad`, and `surprise`.

| Test condition | Accuracy | Macro F1 |
| --- | ---: | ---: |
| clean | 0.6809 | 0.6630 |
| eye occlusion | 0.6107 | 0.5829 |
| mouth occlusion | 0.5978 | 0.5730 |
| random occlusion | 0.6581 | 0.6419 |

Full retained metrics and plots are in `results/roi_ensemble/`.

## Directory layout

```text
expert_level/
|-- data/input/raw/facial_expression_dataset/  # retained FER images
|-- models/roi_ensemble/
|   |-- base/                                  # original ROI-CNN checkpoint
|   |-- robust/                                # occlusion-robust checkpoint
|   `-- ensemble_config.json
|-- outputs/runtime/                           # runtime output location
|-- results/roi_ensemble/                      # final evaluation artifacts
|-- src/                                       # inference-only implementation
|-- main.py
`-- README.md
```

Face detection and landmark files remain external dependencies in
`VisualComputingProject/resources/face_models/`.

## Run

```powershell
D:\miniconda3\envs\vc_sws3026\python.exe VisualComputingProject\expert_level\main.py inspect
D:\miniconda3\envs\vc_sws3026\python.exe VisualComputingProject\expert_level\main.py demo
```

Useful demo options:

```powershell
D:\miniconda3\envs\vc_sws3026\python.exe VisualComputingProject\expert_level\main.py demo --camera 0 --multi-face --debug-hud --show-pose --preprocess clahe
```

Press `q` to close the realtime window.
