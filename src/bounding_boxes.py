"""Convert high Grad-CAM regions to boxes on an untouched original copy."""
import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage

THRESHOLD = 0.60  # fraction of the normalized heatmap maximum
MIN_AREA_FRACTION = 0.004
MAX_REGIONS = 8

def activation_boxes(heatmap, original_size, threshold=THRESHOLD,
                     min_area_fraction=MIN_AREA_FRACTION, max_regions=MAX_REGIONS):
    """Map filtered connected component boxes (x0,y0,x1,y1) to original pixels."""
    width, height = original_size
    mask = np.asarray(heatmap) >= float(threshold)
    labels, count = ndimage.label(mask, structure=np.ones((3, 3), dtype=np.uint8))
    min_area = max(4, int(mask.size * min_area_fraction))
    regions = []
    for component in range(1, count + 1):
        ys, xs = np.where(labels == component)
        if len(xs) < min_area:
            continue
        x0 = int(np.floor(xs.min() * width / mask.shape[1]))
        y0 = int(np.floor(ys.min() * height / mask.shape[0]))
        x1 = min(width, int(np.ceil((xs.max() + 1) * width / mask.shape[1])))
        y1 = min(height, int(np.ceil((ys.max() + 1) * height / mask.shape[0])))
        regions.append((len(xs), (x0, y0, x1, y1)))
    regions.sort(key=lambda item: item[0], reverse=True)
    return [box for _, box in regions[:max_regions]]

def draw_boxes(original, boxes):
    """Draw boxes on a fresh RGB copy of the uploaded source image."""
    result = original.convert("RGB").copy()
    draw = ImageDraw.Draw(result)
    for box in boxes:
        draw.rectangle(box, outline=(255, 55, 55), width=max(2, result.width // 256))
    return result
