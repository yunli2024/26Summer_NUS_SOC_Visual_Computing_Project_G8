# Final ROI Ensemble Notes

The parent `README.md` is the primary usage guide.

The retained implementation is inference-only. Two `face_eyes_mouth` ROI-CNN
checkpoints are combined by weighted probability averaging with base weight
`0.45` and robust weight `0.55`.

Historical training, ablation, calibration, and alternative-model documentation
was removed together with the corresponding code and artifacts.
