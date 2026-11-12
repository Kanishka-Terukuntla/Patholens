# Experiment comparison — 50% pilot run

## Scope and evidence

This app packages the 50% PathMNIST pilot run from `PathoLens_Workspace_50pct`: stratified within official splits using seed 42; 44,998 train, 5,002 validation, and 3,590 test examples. Validation contained 3,501 selection and 1,501 calibration examples. Results describe this pilot subset, not the full benchmark.

| ID | Method | Best epoch | Validation macro-F1 | Test accuracy | Test macro precision | Test macro recall | Test macro-F1 | Parameters |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| E1 | RGB baseline | 15 | **0.994599** | 0.946518 | 0.929285 | 0.937222 | 0.932379 | 1,039,721 |
| E2 | HED stain augmentation | 15 | 0.991265 | **0.964345** | **0.951283** | **0.953557** | **0.952221** | 1,039,721 |
| E3 | Macenko stain normalization | 15 | 0.960941 | 0.852089 | 0.825856 | 0.834028 | 0.820918 | 1,039,721 |
| E4 | Multi-scale MobileNetV3-Small | 14 | 0.990867 | 0.949582 | 0.936502 | 0.933827 | 0.933407 | 1,032,457 |

All four experiments use 128×128 RGB-family inputs, MobileNetV3-Small, AdamW training and ImageNet normalization. E1 uses baseline geometric augmentation. E2 adds stochastic H&E optical-density perturbations in training. E3 applies Macenko stain normalization. E4 pools features from backbone stages 3, 6 and 12 into a compact head. Per-class results and confusion counts for all experiments are included in the CSV files. E2-specific calibration and selective-prediction metrics were not saved; the available calibration outputs belong to E1 and are not transferred to E2.

## Why this app packages E2

E2 leads the saved test comparison in accuracy, macro precision, macro recall, and macro-F1. Its architecture has the same parameter count and similar size as E1. This app therefore packages the E2 best checkpoint from `experiments/E2/best_model.pt` and displays E2's per-class report and confusion matrix.

**Selection caveat:** E1 had the highest prespecified validation-selection macro-F1 (0.994599 vs. E2's 0.991265). Choosing E2 after comparing test metrics uses the test set for selection. The displayed E2 test metrics are therefore post-selection, not an unbiased final performance estimate. Evaluate E2 on a new untouched dataset before making claims about generalization.

No E2 calibration temperature, calibration error, acceptance threshold, or selective-prediction metrics were available. The app reports uncalibrated softmax probabilities and confidence. E1 calibration values are not applied.

## E2 per-class test metrics

| Class | Support | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| adipose | 669 | 0.9910 | 0.9880 | 0.9895 |
| background | 424 | 1.0000 | 1.0000 | 1.0000 |
| debris | 170 | 0.9429 | 0.9706 | 0.9565 |
| lymphocytes | 317 | 0.9724 | 1.0000 | 0.9860 |
| mucus | 518 | 0.9941 | 0.9788 | 0.9864 |
| smooth muscle | 296 | 0.8816 | 0.9054 | 0.8933 |
| normal colon mucosa | 370 | 0.9398 | 0.9703 | 0.9548 |
| cancer-associated stroma | 210 | 0.8629 | 0.8095 | 0.8354 |
| colorectal adenocarcinoma epithelium | 616 | 0.9769 | 0.9594 | 0.9681 |

These pilot results do not establish patient/slide independence, external validity, or clinical utility.
