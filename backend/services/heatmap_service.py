import cv2
import numpy as np


def make_overlay(img_bgr, amap, threshold):
    """Colorise the anomaly map (scale centred on the threshold) and blend onto the image."""
    h, w = img_bgr.shape[:2]
    amap = cv2.resize(np.asarray(amap, dtype=np.float32), (w, h))
    norm = np.clip(amap / (2 * threshold + 1e-8), 0, 1)
    heat = cv2.applyColorMap((norm * 255).astype(np.uint8), cv2.COLORMAP_JET)
    return cv2.addWeighted(img_bgr, 0.6, heat, 0.4, 0)
