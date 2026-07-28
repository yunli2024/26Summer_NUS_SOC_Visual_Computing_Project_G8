# Local runtime resources

The repository keeps the source code and the smaller runtime assets required by
the merged demos. Large course datasets, duplicated source materials, generated
outputs, and heavyweight checkpoints are intentionally excluded from Git.

Before running the relevant pipelines, restore these course-provided files
locally at the exact paths below:

- `face_models/lbfmodel.yaml` (about 54 MB): required by the OpenCV LBF
  68-landmark pipeline.
- `expression_data/facial_expression_dataset.zip` (about 78 MB): required for
  full Expert-level training and final evaluation.

The TikTok body-movement dataset remains local under each bonus-level `data/`
directory and is also excluded. It is used for pipeline exploration, not as a
required machine-learning training set.

The following smaller canonical assets remain versioned:

- Haar and YuNet face detectors;
- YOLOv8 nano pose weights;
- the two reference dance videos;
- the selected keypoint expression classifiers.

Git will ignore the excluded resources after they are restored, so they will
not be uploaded accidentally.

For upstream licences and redistribution notes covering the versioned models,
course materials, datasets, and media, see
[Third-Party Notices](../THIRD_PARTY_NOTICES.md).
