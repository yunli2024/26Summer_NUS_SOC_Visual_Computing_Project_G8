# Third-Party Notices

This repository combines original Group 8 source code and project assets with
third-party libraries, pretrained models, course materials, and datasets. The
root `LICENSE` does not replace the licences or copyright notices that apply to
those third-party materials.

## Ultralytics YOLO

- Runtime dependency: [`ultralytics`](https://github.com/ultralytics/ultralytics)
- Included model: `resources/pose_models/yolov8n-pose.pt`
- Upstream licence: [GNU AGPL-3.0](https://github.com/ultralytics/ultralytics/blob/main/LICENSE)
- Official licensing guidance: [Ultralytics License](https://www.ultralytics.com/license)

Ultralytics states that its open-source software and trained YOLO models are
licensed under AGPL-3.0. The project therefore uses AGPL-3.0 at the repository
level. Proprietary or commercial deployments that cannot meet AGPL-3.0
obligations may require a separate Ultralytics licence.

## OpenCV Models and Runtime

- `opencv-contrib-python` is distributed under the
  [Apache-2.0 licence](https://github.com/opencv/opencv/blob/4.x/LICENSE).
- `resources/face_models/haarcascade_frontalface_default.xml` retains the Intel
  Open Source Computer Vision Library licence notice embedded in the XML file.
- `resources/face_models/face_detection_yunet_2023mar.onnx` comes from the
  [OpenCV Zoo YuNet directory](https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet),
  whose files are identified upstream as MIT-licensed.

## Course Materials, Datasets, and Media

- `docs/course_materials/Visual_Computing_Project.pdf` remains course material
  of its respective owner and is not relicensed by this repository.
- FER-style facial-expression data, the TikTok exploration dataset, and the LBF
  model are intentionally excluded from Git. Users must obtain them from an
  authorised source and comply with their original terms.
- Reference videos and any other course-provided media retain the rights of
  their respective creators unless a file-specific notice states otherwise.

## Other Dependencies

The packages listed in `environment_setup/requirements.txt` remain subject to
their own licences. Refer to the corresponding upstream project before
redistributing or using this repository outside an academic or open-source
context.
