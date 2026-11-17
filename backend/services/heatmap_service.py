import cv2
import numpy as np


def make_overlay(img_bgr, amap, threshold):
    """Colorise the anomaly map and blend onto the image with clean transparency on normal regions."""
    h, w = img_bgr.shape[:2]
    amap = cv2.resize(np.asarray(amap, dtype=np.float32), (w, h))
    norm = np.clip(amap / (2 * threshold + 1e-8), 0.0, 1.0)
    heat = cv2.applyColorMap((norm * 255).astype(np.uint8), cv2.COLORMAP_JET)

    # Adaptive alpha: normal regions (norm < 0.10) remain 100% transparent
    alpha = np.clip((norm - 0.10) / 0.35, 0.0, 0.60)[:, :, np.newaxis]
    blended = (img_bgr.astype(np.float32) * (1.0 - alpha) + heat.astype(np.float32) * alpha).astype(np.uint8)
    return blended
