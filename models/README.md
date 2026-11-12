# Selected model

- Experiment: **E2 — HED stain augmentation** from the 50% PathMNIST run.
- Original checkpoint: `PathoLens_Workspace_50pct/experiments/E2/best_model.pt`.
- Copied checkpoint: `models/best_model.pth` (PyTorch checkpoint with model state dictionary).
- Supporting inference config: `models/inference_config.json`, recreated from the saved E1 config for E2's shared architecture/class/preprocessing fields. No E2-specific calibration settings were saved, so those fields are explicitly unavailable.
- Architecture: torchvision MobileNetV3-Small backbone with compact head: Linear(576,192,bias=False), LayerNorm(192), Hardswish, Dropout(0.25), Linear(192,9). E2 adds H&E optical-density perturbation during training; inference uses the original RGB evaluation transform. Grad-CAM targets the final backbone feature stage.
- Input: RGB resized to 128×128, float tensor in [0,1], normalized with mean `[0.485, 0.456, 0.406]` and standard deviation `[0.229, 0.224, 0.225]`.
- Class order: 0 adipose; 1 background; 2 debris; 3 lymphocytes; 4 mucus; 5 smooth muscle; 6 normal colon mucosa; 7 cancer-associated stroma; 8 colorectal adenocarcinoma epithelium.
- Best epoch: 15; validation selection macro-F1 0.991265.
- 3,590-image 50% pilot test results: accuracy 0.964345, macro precision 0.951283, macro recall 0.953557, macro F1 0.952221.

E2 is selected for this app because it leads the saved test comparison on accuracy and macro metrics. Since test results informed this choice, they are post-selection and are not an unbiased final estimate. A new untouched evaluation set is needed for that purpose. Keep `best_model.pth` and `inference_config.json` together; the original workspace is not needed at runtime.
