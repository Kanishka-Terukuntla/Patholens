"""Prediction utilities."""
from dataclasses import dataclass
import torch

@dataclass
class Prediction:
    logits: torch.Tensor
    probabilities: torch.Tensor
    class_index: int
    confidence: float

def predict(model, tensor, device):
    """Return probabilities and confidence from the selected model logits."""
    tensor = tensor.to(device)
    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1)[0]
    index = int(probs.argmax().item())
    return Prediction(logits[0].detach(), probs.detach(), index,
        float(probs[index].item()))
