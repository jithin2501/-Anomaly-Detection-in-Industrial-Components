from pathlib import Path
import cv2
import numpy as np

IMG_EXT = {".png", ".jpg", ".jpeg", ".bmp"}


def list_test_images(test_dir: Path):
    """Yield (path, label) for an MVTec-style test dir. label 0 = good, 1 = defective."""
    for sub in sorted(Path(test_dir).iterdir()):
        if not sub.is_dir():
            continue
        label = 0 if sub.name == "good" else 1
        for p in sorted(sub.iterdir()):
            if p.suffix.lower() in IMG_EXT:
                yield p, label


def make_overlay(img_bgr, amap, threshold):
    """Blend a JET heatmap onto the image. Colour scale is centred on the threshold."""
    h, w = img_bgr.shape[:2]
    amap = cv2.resize(np.asarray(amap, dtype=np.float32), (w, h))
    norm = np.clip(amap / (2 * threshold + 1e-8), 0, 1)
    heat = cv2.applyColorMap((norm * 255).astype(np.uint8), cv2.COLORMAP_JET)
    return cv2.addWeighted(img_bgr, 0.6, heat, 0.4, 0)
