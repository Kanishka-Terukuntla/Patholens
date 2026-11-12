"""Selected E2 MobileNetV3-Small model and checkpoint loading."""
from pathlib import Path
import json
import pickle
import torch
from torch import nn
from torchvision import models

ROOT = Path(__file__).resolve().parents[1]
CONFIG = json.loads((ROOT / "models" / "inference_config.json").read_text(encoding="utf-8"))

class MobileNetV3SmallClassifier(nn.Module):
    """Matches notebook model_id mobilenet_v3_small_compact_head_v1."""
    def __init__(self):
        super().__init__()
        self.backbone = models.mobilenet_v3_small(weights=None)
        in_features = self.backbone.classifier[0].in_features
        self.backbone.classifier = nn.Sequential(
            nn.Linear(in_features, 192, bias=False), nn.LayerNorm(192),
            nn.Hardswish(), nn.Dropout(p=0.25), nn.Linear(192, 9))

    def forward(self, x):
        return self.backbone(x)

    @property
    def cam_layer(self):
        return self.backbone.features[-1]

def load_model():
    """Load the packaged E2 checkpoint; fail clearly when it is absent."""
    path = ROOT / "models" / "best_model.pth"
    if not path.is_file():
        raise FileNotFoundError(f"Selected E2 checkpoint is missing: {path}")
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        checkpoint = torch.load(path, map_location="cpu")
    except pickle.UnpicklingError:
        # This is the locally bundled training checkpoint, which also stores
        # non-tensor audit metadata. It is copied from the user's own workspace.
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state = checkpoint["model_state_dict"]
    elif isinstance(checkpoint, dict) and all(isinstance(k, str) for k in checkpoint):
        state = checkpoint
    else:
        raise RuntimeError("Checkpoint has an unsupported format; expected the selected E2 model state_dict.")
    model_id = CONFIG.get("model_id")
    if model_id == "mobilenet_v3_small_compact_head_v1":
        model = MobileNetV3SmallClassifier()
    else:
        raise RuntimeError(f"Unsupported model architecture in inference config: {model_id}")
    model.load_state_dict(state, strict=True)
    return model.to(device).eval(), device
