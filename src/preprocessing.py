"""Inference preprocessing matching the selected E1 evaluation transform."""
from pathlib import Path
import json
from PIL import Image, ImageOps
import torch
from torchvision import transforms

ROOT = Path(__file__).resolve().parents[1]
CONFIG = json.loads((ROOT / "models" / "inference_config.json").read_text(encoding="utf-8"))
MEAN, STD = CONFIG["normalization_mean"], CONFIG["normalization_std"]
TRANSFORM = transforms.Compose([
    transforms.Resize((CONFIG["input_size"], CONFIG["input_size"])),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])

def prepare_image(image: Image.Image):
    """Return an EXIF-corrected RGB original copy and normalized model tensor."""
    original = ImageOps.exif_transpose(image).convert("RGB")
    tensor = TRANSFORM(original).unsqueeze(0)
    return original.copy(), tensor
