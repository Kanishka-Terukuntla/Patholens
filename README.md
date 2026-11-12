# 🔬 PathoLens AI

### Colorectal histopathology classification with visual explanations

PathoLens AI is an interactive research prototype for classifying colorectal histopathology tiles into nine tissue categories. It brings together a compact image-classification model, class probability charts, and Grad-CAM visualizations in a multi-page Streamlit application.

The project explores how different training recipes affect classification performance and gives users a way to inspect the model's output on an uploaded image. It is intended for education and research exploration.

> ⚠️ **Research use only.** This software is not clinically validated, is not a diagnostic service, and must not be used to make decisions about patient care.

<p align="center">
  <strong>9 tissue classes</strong> &nbsp;·&nbsp; <strong>128 × 128 tiles</strong> &nbsp;·&nbsp; <strong>4 experiment variants</strong> &nbsp;·&nbsp; <strong>Grad-CAM explanations</strong>
</p>

---

## Contents

- [Project at a glance](#-project-at-a-glance)
- [Application tour](#-application-tour)
- [Interactive prediction workflow](#-interactive-prediction-workflow)
- [Dataset and experimental design](#-dataset-and-experimental-design)
- [Results](#-results)
- [Selected model](#-selected-model)
- [Quick start](#-quick-start)
- [Repository map](#-repository-map)
- [Data and model files](#-data-and-model-files)
- [How to interpret the outputs](#-how-to-interpret-the-outputs)
- [Limitations](#-limitations)
- [Credits and dataset citation](#-credits-and-dataset-citation)

## 🧭 Project at a glance

| | Details |
|---|---|
| **Task** | Nine-class colorectal histopathology tile classification |
| **Dataset** | PathMNIST from the MedMNIST collection |
| **Input** | RGB histology images, prepared as 128 × 128 tiles |
| **Pilot subset** | Stratified 50% subset of each official split; seed 42 |
| **Application** | Streamlit multi-page app |
| **Model family** | MobileNetV3-Small |
| **Packaged model** | E2 with HED stain augmentation |
| **Explainability** | Grad-CAM heatmap and approximate high-activation regions |

### The nine tissue categories

Adipose · background · debris · lymphocytes · mucus · smooth muscle · normal colon mucosa · cancer-associated stroma · colorectal adenocarcinoma epithelium

## 🖥️ Application tour

The app uses a dark interface and provides a guided route through the project:

| Page | What it covers |
|---|---|
| **Home** | Project introduction and navigation |
| **Project Overview** | Task, dataset scope, tissue categories, and system workflow |
| **Methodology and Experiments** | Training approach and side-by-side experiment metrics |
| **Test Results and Best Model** | Saved evaluation results and the selected E2 model |
| **Image Prediction** | Upload an image and inspect the predicted class and visual outputs |
| **Explainability** | How Grad-CAM and activation-region visualization work |
| **Project Flowchart** | Optional standalone flowchart page, available when its HTML guide is present |

The flowchart page is loaded from `docs/PathoLens_Project_Flowchart.html`. The main app pages do not depend on that optional guide.

## 🧪 Interactive prediction workflow

On the **Image Prediction** page, upload a JPG, JPEG, or PNG histology tile. The app then:

1. Converts the image to RGB and prepares the model input using the packaged preprocessing configuration.
2. Runs the E2 classifier and reports its top predicted category and image-level softmax confidence.
3. Displays the probability assigned to each of the nine categories.
4. Computes Grad-CAM for the predicted class and displays a red activation heatmap.
5. Finds approximate connected high-activation regions and draws boxes on a fresh copy of the original image.

The threshold slider filters the regions shown from the already-computed activation map; changing it does not rerun model inference. The app requires a minimum image side of 32 pixels for a useful demonstration. Display previews are reduced in size, while region coordinates are mapped to the full-resolution source image.

## 🧬 Dataset and experimental design

The experiments use PathMNIST, a nine-class colorectal histopathology dataset in the MedMNIST collection. The project metadata describes a **stratified 50% pilot subset** of each official split, sampled with seed 42:

| Split | Images in the pilot |
|---|---:|
| Training | 44,998 |
| Validation | 5,002 |
| Test | 3,590 |

The validation split was divided into **3,501 model-selection examples** and **1,501 calibration examples**. The app packages the E2 model trained for this pilot run. These counts and results describe the pilot subset, not the full PathMNIST benchmark.

The four saved experiments share a MobileNetV3-Small backbone family, 128 × 128 inputs, ImageNet normalization, and AdamW training. They vary the image treatment or feature aggregation:

| ID | Experiment | Training variation |
|---|---|---|
| **E1** | RGB baseline | RGB input with random 90° rotations and horizontal/vertical flips; compact classifier head |
| **E2** | HED stain augmentation | Compact head with stochastic H&E optical-density perturbation during training |
| **E3** | Macenko stain normalization | Stain normalization using a reference estimated from training-subset images |
| **E4** | Multi-scale features | Pooled features from MobileNetV3-Small stages 3, 6, and 12 combined in a compact head |

## 📊 Results

The table reports saved results for the 3,590-image pilot test split. Values are rounded to four decimal places; the app's detailed result page uses the saved metric files.

| Experiment | Best epoch | Test accuracy | Macro precision | Macro recall | Test macro-F1 | Validation-selection macro-F1 |
|---|---:|---:|---:|---:|---:|---:|
| **E1 · RGB baseline** | 15 | 0.9465 | — | — | 0.9324 | **0.9946** |
| **E2 · HED augmentation** | 15 | **0.9643** | **0.9513** | **0.9536** | **0.9522** | 0.9913 |
| **E3 · Macenko normalization** | 15 | 0.8521 | — | — | 0.8209 | — |
| **E4 · Multi-scale features** | 14 | 0.9496 | — | — | 0.9334 | 0.9909 |

E2 had the strongest saved test accuracy and macro-F1, so it is the model packaged in the app. E1 had the highest validation-selection macro-F1. Full precision, recall, class-level metrics, training curves, and comparison figures are available in the app and under `reports/` when those files are present.

> **Evaluation caveat:** E2 was selected after comparing test-set results. The test set therefore influenced model selection, making the reported E2 test metrics **post-selection** rather than an unbiased final estimate. A new untouched evaluation set is needed for an independent estimate.

### E2 model notes

- **Architecture:** MobileNetV3-Small with a compact classification head.
- **Training treatment:** Stochastic H&E optical-density perturbation (HED stain augmentation).
- **Trainable parameters:** 1,039,721.
- **Saved checkpoint size:** approximately 4.09 MB.
- **Calibration:** E2-specific calibration metrics were not saved. The application displays raw, uncalibrated softmax probabilities; E1 calibration values are not applied to E2.
- **Inference speed:** depends on the available hardware.

## 🚀 Quick start

### Requirements

- Python 3.10 or newer
- A supported Windows, macOS, or Linux environment
- Internet access during package installation

Create and activate a virtual environment from the project directory.

**Windows PowerShell**

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

**macOS / Linux**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install dependencies and launch the app:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

Open the local URL printed by Streamlit in your browser. Start Streamlit from the repository root so the pages can locate the packaged model, reports, assets, and sample images.

### Try a sample

Open **Image Prediction** and upload a PNG from [`samples/`](samples/). The nine example files cover one image from each tissue category in the pilot test subset. They are convenient examples, not a substitute for an independent evaluation dataset.

## 🗂️ Repository map

```text
PathoLens_Application/
├── app.py                  # Streamlit entry point, theme, and navigation
├── pages/                  # Home, overview, experiments, results, prediction, explainability
├── src/                    # Preprocessing, model loading, prediction, Grad-CAM, metrics, charts
├── models/                 # Packaged E2 checkpoint and inference configuration
├── reports/                # Experiment comparison, metrics, and saved result figures
├── assets/                 # Application and experiment visual assets
├── samples/                # Nine small example images for interactive prediction
├── notebooks/              # Experiment and inference notebooks
├── docs/                   # Optional standalone project flowchart
├── data/                   # Local data directory; large dataset files are Git-ignored
├── requirements.txt        # Python dependencies
└── .gitignore              # Excludes environments, generated files, checkpoints, and data
```

## 📦 Data and model files

Large dataset files under `data/` are intentionally excluded from Git to keep the repository and GitHub uploads manageable. Empty directory placeholders can remain tracked. Ignoring `data/` does not delete local files; it only prevents untracked data files from being added to future commits. If you need the training arrays or full archive, obtain PathMNIST from MedMNIST and follow its dataset instructions.

The packaged inference model and its configuration are stored under `models/`. The app's prediction workflow uses the checkpoint and configuration; it does not need to load the training dataset arrays at runtime. The small example PNGs under `samples/` remain in the project for quick demonstrations.

## 🔎 How to interpret the outputs

- **Predicted class:** the class with the highest model score for the image.
- **Class probabilities:** raw softmax scores across all nine classes. They are not calibrated probabilities of clinical truth.
- **Grad-CAM heatmap:** a relative visualization of image regions that contributed to the selected class score. Red indicates stronger relative activation in the displayed map.
- **Activation boxes:** approximate regions derived from thresholded Grad-CAM activation. They are not object detections and do not carry individual confidence scores.
- **Threshold control:** filters which approximate regions are drawn; it is not a per-region confidence threshold and does not rerun inference.

Grad-CAM is an explanatory visualization of model behavior, not a guarantee that a highlighted region is medically meaningful.

## ⚠️ Limitations

- This is an educational research prototype and is not a clinically validated diagnostic system.
- Results are from a stratified 50% pilot subset, not a complete independent evaluation of the full PathMNIST benchmark.
- E2 was selected using the test comparison, so its test metrics are post-selection.
- Patient- and slide-level independence is not established by the saved project results.
- E2 calibration and selective-prediction metrics are unavailable; displayed softmax outputs are uncalibrated.
- Grad-CAM boxes are coarse, class-associated activation regions, not lesion boundaries, detections, or segmentation masks.
- Performance on images from other sources, scanners, staining protocols, or clinical populations has not been established.

## 📚 Credits and dataset citation

PathMNIST is part of the [MedMNIST](https://medmnist.com/) collection. The dataset files are distributed under [CC BY 4.0](https://github.com/MedMNIST/MedMNIST#data-files). Consult the official MedMNIST documentation and publication for dataset terms and citation details before reuse.

See [`LICENSE`](LICENSE) for the license that applies to this project code. Dataset terms remain those specified by MedMNIST.
