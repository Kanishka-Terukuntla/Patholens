# PathoLens AI

### Explore colorectal histopathology classification with visual explanations

PathoLens AI is a Streamlit research prototype for classifying colorectal histopathology images into nine tissue classes. It pairs an image classifier with class probabilities and Grad-CAM visualizations to make model outputs easier to inspect.

> **Research prototype only.** PathoLens AI is not clinically validated and must not be used to diagnose or guide patient care.

## At a glance

| | |
|---|---|
| **Task** | Nine-class colorectal histopathology image classification |
| **Dataset** | PathMNIST (MedMNIST), 50% pilot run |
| **App** | Streamlit |
| **Selected model** | E2 MobileNetV3-Small with HED stain augmentation |
| **Explanations** | Grad-CAM heatmap and approximate activation regions |

## What you can do

- Explore the project overview, methodology, saved experiment results, predictions, and explanations.
- Upload a JPG, JPEG, or PNG image and view the predicted class with probabilities for all nine classes.
- Compare the source image with its Grad-CAM heatmap and an overlay showing approximate high-activation regions.
- Adjust the activation threshold to filter displayed regions without rerunning inference.
- Review saved metrics and figures for the four pilot experiments.

## Preview with sample images

Use one of the PNGs in [`samples/`](samples/) to try the prediction page. The folder contains one labeled image from each class in the pilot test subset.

## Get started

Use Python 3.10 or newer. From this folder, create and activate a virtual environment, then install the project dependencies:

```bash
python -m venv .venv
```

On Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Launch the app:

```bash
python -m streamlit run app.py
```

Streamlit will print a local URL to open in your browser. Run the command from the project folder so the app can find its model, reports, assets, and samples.

## Repository contents

| Path | Description |
|---|---|
| `app.py` | Streamlit home page and shared styling |
| `pages/` | App pages for methodology, results, prediction, and explainability |
| `src/` | Model loading, preprocessing, inference, Grad-CAM, and result utilities |
| `models/` | Model checkpoint and inference configuration |
| `reports/` | Experiment comparison and saved metrics |
| `assets/` | Saved experiment figures |
| `samples/` | Small example images for trying predictions |
| `notebooks/` | Inference and explainability notebook |
| `data/` | Local dataset files; excluded from Git because of their size |
| `docs/` | Optional project flowchart shown in the app when present |

The PathMNIST dataset is available from [MedMNIST](https://medmnist.com/). Dataset files are intentionally excluded from this repository to keep GitHub uploads small. The application uses the packaged model for prediction and does not need the training arrays at runtime.

## Model and evaluation

The app packages E2, the MobileNetV3-Small experiment with HED stain augmentation. On 3,590 pilot test images, E2 recorded accuracy of **0.964345** and macro-F1 of **0.952221**. E1 had the strongest validation-selection macro-F1 (**0.994599**, compared with E2's **0.991265**).

E2 was selected after comparing test results, so its reported test metrics are post-selection and should not be treated as an unbiased final estimate. Evaluate on a new, untouched dataset for an independent estimate. See [`reports/experiment_comparison.md`](reports/experiment_comparison.md) for the experiment details.

The displayed probabilities are raw, uncalibrated softmax outputs. No E2-specific calibration or selective-prediction metrics were saved; E1 calibration results are not applied to E2.

## Interpreting Grad-CAM

Grad-CAM highlights image regions associated with the model's prediction. The outlined regions are approximate activation areas, not object detections, lesion annotations, or segmentation masks. The threshold slider filters the normalized activation map; it is not a confidence score for individual regions. The classifier produces one confidence score for the full image.

## License and dataset attribution

See [`LICENSE`](LICENSE) for this project's license. PathMNIST is distributed by MedMNIST under [CC BY 4.0](https://github.com/MedMNIST/MedMNIST#data-files); refer to the official dataset page for its terms and citation guidance.
