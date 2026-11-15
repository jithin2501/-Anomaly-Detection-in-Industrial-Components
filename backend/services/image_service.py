import base64

import cv2
import numpy as np


def decode(data: bytes):
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("invalid image")
    return img


def to_base64(img_bgr) -> str:
    ok, buf = cv2.imencode(".png", img_bgr)
    if not ok:
        raise ValueError("encode failed")
    return base64.b64encode(buf).decode()
