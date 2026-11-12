# PathoLens AI

Self-contained Streamlit research prototype for nine-class colorectal histopathology classification using the **50% PathMNIST pilot run**. The app packages E2 (HED stain augmentation), the strongest model on the saved test comparison, with class probabilities, Grad-CAM, and approximate activation-region boxes.

The Streamlit interface uses a dark color theme throughout, including its saved-result charts and prediction views.

## Features

- Five pages: overview, methodology, saved results, prediction, explainability.
- Packaged E2 MobileNetV3-Small checkpoint and inference configuration.
- Copies of the 50% run's sampled train and test image, label, and index arrays.
- JPG/JPEG/PNG upload, nine-class raw softmax probabilities and confidence.
- Separate original, heatmap, and original-with-boxes outputs.
- Interactive Grad-CAM activation threshold for filtering approximate regions without rerunning inference.
- Saved metrics and figures for all four 50% experiments.

## Project map

| Folder or file | Purpose |
|---|---|
| `app.py` | Streamlit home page and application styling |
| `pages/` | Five user-facing pages |
| `src/` | Model loading, preprocessing, prediction, Grad-CAM, boxes, charts, and saved-result readers |
| `models/` | E2 checkpoint, inference configuration, and model notes |
| `data/` | Full raw archive plus sampled 50% train/validation/test splits and provenance |
| `reports/` | Experiment report and saved CSV metrics |
| `assets/` | Saved experiment figures |
| `notebooks/` | New inference and explainability notebook |
| `requirements.txt` | Python dependencies |

The standalone guide at `docs/PathoLens_Project_Flowchart.html` is optional. When it exists, the app adds a **Project Flowchart** menu page that displays it. Removing the HTML file removes that menu entry; the app's other routes do not depend on the guide.

## Selected model and evaluation caveat

E2 test accuracy was 0.964345 and macro-F1 0.952221 on 3,590 pilot test images. E1 had the highest validation-selection macro-F1 (0.994599 versus E2's 0.991265). This app selects E2 based on the test comparison, so those test results are post-selection and are not an unbiased final estimate. Evaluate on new untouched data for an independent estimate. See `reports/experiment_comparison.md`.

No E2-specific calibration or selective-prediction metrics were saved. Confidence and probabilities in the app are uncalibrated softmax outputs; E1 calibration values are not applied to E2.

## Install and launch

Create and activate a Python 3.10+ environment. Install `requirements.txt` into that active environment; it includes the app's image, model, chart, and data libraries, plus Jupyter support for the included notebook:

```bash
python -m pip install -r requirements.txt
```

Then, still in the same environment and from this folder, launch the app:

```bash
python -m streamlit run app.py
```

Installing the requirements downloads Python packages only. The model checkpoint and dataset are already included in this project folder; they are not downloaded by `requirements.txt`.

## Image prediction

Open **4 Image Prediction**, upload JPG/JPEG/PNG, and review the predicted class and nine-class probability chart. Output A displays the RGB source, Output B the separate red-scale Grad-CAM map, and Output C a fresh copy of the source with retained activation boxes. The activation threshold slider filters regions from the already-computed map; it does not rerun model inference. This is a normalized Grad-CAM activation cutoff, not a per-box confidence threshold: the classifier produces one confidence score for the whole image, not confidence scores for individual regions. If no meaningful region passes the filters, no box is drawn.

The full local PathMNIST-128 archive and all three sampled 50% splits are included under `data/` with metadata and provenance for future work. The app uses the E2 checkpoint trained on the 50% run; prediction does not need to load dataset arrays.

## Limitations

Grad-CAM boxes are approximate class-associated activation regions, not detections, lesion annotations, or segmentation masks. This is not a diagnostic service. Results are from a 50% pilot subset, not the full PathMNIST benchmark; patient/slide independence is not established. Test metrics were used to select E2 for the app and are post-selection. No E2 calibration results are available.

## Sample images

Upload any PNG in samples/ to quickly exercise the prediction page. The folder contains one labeled image from each class in the bundled 50% PathMNIST test subset. These PNGs were exported from the packaged arrays without changing pixel values. The underlying dataset is distributed by MedMNIST under CC BY 4.0; see [the official MedMNIST data page](https://github.com/MedMNIST/MedMNIST#data-files).

