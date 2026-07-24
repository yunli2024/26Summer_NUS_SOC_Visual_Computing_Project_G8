# Part Two Task 1 - Results and Discussion

## Method

The supplied FER-2013 split contains 28,709 training images and 7,178 testing images across seven classes. Every image is a 48x48 grayscale, centered face crop.

Each image was enlarged to 192x192 before fitting the supplied LBF model. Haar detection found only about one third of a 70-image stratified sample because the source images are very small. The final extraction therefore used the known centered FER crop as the face region, while retaining `haar` and `haar-fallback` modes for diagnostic comparison. This is a dataset-specific strategy; live video must still use a face detector.

The 68 two-dimensional landmarks were normalized by:

1. Computing the mean point of each eye.
2. Translating the midpoint between the eyes to the origin.
3. Rotating the eye line to horizontal.
4. Dividing all coordinates by the inter-eye distance.
5. Flattening the aligned points into 136 features.

A class-balanced histogram gradient boosting classifier was trained on these features. It was selected because a full RBF SVM on 28,709 samples had impractically long training time, while gradient boosting still provides nonlinear decision boundaries and fast inference.

## Quantitative results

| Metric | Result |
|---|---:|
| Test accuracy | 43.13% |
| Macro F1 | 39.14% |
| Weighted F1 | 42.59% |
| Training samples | 28,709 |
| Testing samples | 7,178 |
| Successful landmark extractions | 35,887 / 35,887 |
| Classifier training time | 358.80 s |
| Repeated single-image prediction | 23.41 ms |
| Meets the 30 ms requirement | Yes |

Per-class recall was highest for `happy` (63.30%) and `surprise` (60.41%). It was lowest for `fear` (20.61%). Although `disgust` recall reached 30.63%, its precision was only 24.82%; the class contains only 436 training images and 111 test images.

## Confusion and failure analysis

The confusion matrix shows several important patterns:

- `fear` is frequently classified as `sad` (227 images), `neutral` (182), `happy` (141), or `surprise` (137).
- `sad` is frequently classified as `neutral` (257), `angry` (174), or `fear` (168).
- `happy` and `surprise` are easier because mouth opening, mouth-corner position, and eye opening create strong geometric changes.
- Subtle classes such as `fear`, `sad`, `neutral`, and `disgust` often require texture cues such as wrinkles, cheek tension, or shading that are absent from landmark-only features.

The high-confidence failure sheet also reveals possible label ambiguity. Several images labelled `fear` visually resemble `surprise`, and some images labelled `neutral` contain visible smiles. Low resolution, head rotation, occlusion, unusual crops, and FER label noise all contribute to errors.

## Challenges and solutions

| Challenge | Strategy |
|---|---|
| Haar misses many 48x48 faces | Enlarge to 192x192 and use the known centered FER crop for LBF fitting |
| Different face position, size, and tilt | Eye-based translation, rotation, and scale normalization |
| Strong class imbalance | Use balanced class weights and report Macro F1 as well as accuracy |
| RBF SVM training is too slow on the full set | Use class-balanced histogram gradient boosting |
| Real-time requirement | Measure repeated single-image inference rather than batch-only timing |
| Some expressions are not geometrically distinctive | Document confusion and failure cases rather than hiding them |

## Limitations and possible future improvements

- Add distances and angles focused on the eyebrows, eyelids, and mouth as engineered features.
- Use temporal voting in the live application to stabilize predictions.
- Compare against an MLP trained on the same normalized landmarks.
- Detect and remove obviously incorrect landmark fits instead of accepting every centered crop.
- If project rules permit, combine landmark geometry with a small number of appearance features; this would no longer be a purely keypoint-only comparison and should be reported separately.
- Cross-validation was not used because it is optional and the supplied FER train/test split was preserved.

The complete numeric output is stored in `artifacts/metrics.json`, with visual evidence in `artifacts/confusion_matrix.png` and `artifacts/failure_cases.png`.
