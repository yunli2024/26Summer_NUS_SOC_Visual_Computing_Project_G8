# Part Two Task 1 - Results and Discussion

## Method

The supplied FER-2013 split contains 28,709 training images and 7,178 testing images across seven classes. Every image is a 48x48 grayscale, centered face crop.

Each image was enlarged to 192x192 before fitting the supplied LBF model. Haar detection found only about one third of a 70-image stratified sample because the source images are very small. The final extraction therefore used the known centered FER crop as the face region, while retaining `haar` and `haar-fallback` modes for diagnostic comparison. This is a dataset-specific strategy; live video must still use a face detector.

The 68 two-dimensional landmarks were normalized by:

1. Computing the mean point of each eye.
2. Translating the midpoint between the eyes to the origin.
3. Rotating the eye line to horizontal.
4. Dividing all coordinates by the inter-eye distance.
5. Flattening the aligned points into 136 coordinate features.

The improved model appends 38 explicit descriptors derived only from those
normalized landmarks. They cover eyebrow-to-eye gaps and slopes; eye width,
height, EAR, area, and asymmetry; nose-to-mouth distances; mouth opening,
aspect ratios, corner angle and curvature; lip thickness; mouth-to-chin
distance; and left-right facial asymmetry. The final input to the scaler is
therefore 174-dimensional. No raw pixel or texture feature is used.

A class-balanced RBF-SVM was selected using a stratified 80/20 split of the
official training set. A 10,000-sample coarse search compared 12 combinations
of `C` and `gamma`; the best three were then fitted on all 22,967 internal
training samples. Validation Macro F1 selected `C=10` and
`gamma=0.0114943` (twice the baseline of `1/174`). Only after selection was
the model refitted on all 28,709 official training samples and evaluated on
the official test set.

HGB and PCA-SVM were retained as comparison experiments. PCA preserving 95%
variance compressed 136 coordinates to 12 components, improving efficiency but
discarding discriminative low-variance expression information.

## Quantitative results

| Metric | Result |
|---|---:|
| Test accuracy | 47.31% |
| Macro F1 | 46.33% |
| Weighted F1 | 47.35% |
| Training samples | 28,709 |
| Testing samples | 7,178 |
| Successful landmark extractions | 35,887 / 35,887 |
| Final classifier training time | 1,535.67 s |
| Repeated single-image prediction | 18.08 ms |
| Meets the 30 ms requirement | Yes |

| Experiment | Accuracy | Macro F1 | Single-image prediction |
|---|---:|---:|---:|
| HGB | 43.13% | 39.14% | 23.41 ms |
| RBF-SVM, untuned | 46.31% | 43.47% | 5.17 ms |
| PCA 95% + RBF-SVM | 40.32% | 36.13% | 2.68 ms |
| Tuned coordinate-only RBF-SVM | 46.84% | 45.89% | **5.31 ms** |
| Tuned coordinates + geometry RBF-SVM | **47.31%** | **46.33%** | 18.08 ms |

Compared with the tuned coordinate-only SVM, explicit geometry increased test
accuracy by 0.47 percentage points and Macro F1 by 0.45 points. The largest
per-class F1 gains were `fear` (+2.18 points) and `sad` (+1.88 points);
`surprise` gained 0.54 points. `Happy` fell by 1.03 points, while `angry` and
`disgust` fell by about 0.24 points each. The feature engineering therefore
helps subtle expression geometry overall, but does not improve every class.

Per-class recall was highest for `happy` (64.94%) and `surprise` (60.89%).
`fear` remained difficult at 34.86% recall. `Disgust` retained 41.44% recall
and 56.10% precision; however, it contains only 436 training images and 111
test images.

## Confusion and failure analysis

The confusion matrix shows several important patterns:

- `fear` is frequently classified as `angry` (179 images), `sad` (175), or `neutral` (141).
- `sad` is frequently classified as `neutral` (219), `angry` (208), or `fear` (206).
- `happy` and `surprise` are easier because mouth opening, mouth-corner position, and eye opening create strong geometric changes.
- Subtle classes such as `fear`, `sad`, `neutral`, and `disgust` often require texture cues such as wrinkles, cheek tension, or shading that are absent from landmark-only features.

The high-confidence failure sheet also reveals possible label ambiguity. Several images labelled `fear` visually resemble `surprise`, and some images labelled `neutral` contain visible smiles. Low resolution, head rotation, occlusion, unusual crops, and FER label noise all contribute to errors.

## Challenges and solutions

| Challenge | Strategy |
|---|---|
| Haar misses many 48x48 faces | Enlarge to 192x192 and use the known centered FER crop for LBF fitting |
| Different face position, size, and tilt | Eye-based translation, rotation, and scale normalization |
| Strong class imbalance | Use balanced class weights and report Macro F1 as well as accuracy |
| Coordinates do not explicitly encode expression ratios | Append 38 landmark-only distances, angles, ratios, and symmetry features |
| SVM performance depends on `C` and `gamma` | Use a stratified validation search optimizing Macro F1 |
| Real-time requirement | Measure repeated single-image inference rather than batch-only timing |
| Some expressions are not geometrically distinctive | Document confusion and failure cases rather than hiding them |

## Limitations and possible future improvements

- Refine or select the 38 geometry descriptors to remove redundant or noisy ratios.
- Use temporal voting in the live application to stabilize predictions.
- Compare against an MLP trained on the same normalized landmarks.
- Detect and remove obviously incorrect landmark fits instead of accepting every centered crop.
- If project rules permit, combine landmark geometry with a small number of appearance features; this would no longer be a purely keypoint-only comparison and should be reported separately.
- Cross-validation was not used because it is optional and the supplied FER train/test split was preserved.

The final geometry-augmented output is stored in
`artifacts_svm_geometry/metrics.json`.
The candidate table, heatmap, confusion matrix, and failure cases are stored in
the same directory. Cross-validation was not used; the official training set
was split once for parameter selection and then restored for final fitting.
