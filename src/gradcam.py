"""Grad-CAM for the selected class using the final MobileNet feature stage."""
import torch
import torch.nn.functional as F

def generate_gradcam(model, tensor, device, target_class):
    """Return normalized 128x128 CAM; hooks are removed even on errors."""
    activations, gradients = {}, {}
    def capture_activation(_module, _inputs, output):
        activations["value"] = output
        output.register_hook(lambda grad: gradients.__setitem__("value", grad))
    handle = model.cam_layer.register_forward_hook(capture_activation)
    try:
        model.zero_grad(set_to_none=True)
        logits = model(tensor.to(device))
        logits[0, int(target_class)].backward()
        feature = activations["value"].detach()
        grad = gradients["value"].detach()
        weights = grad.mean(dim=(2, 3), keepdim=True)
        cam = torch.relu((weights * feature).sum(dim=1, keepdim=True))
        cam = F.interpolate(cam, size=tensor.shape[-2:], mode="bilinear", align_corners=False)[0, 0]
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam.detach().cpu().numpy()
    finally:
        handle.remove()
        model.zero_grad(set_to_none=True)
