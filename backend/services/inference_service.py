import json

from ai_engine.predict import load_model, run_model
from config import MODEL_PATH, THRESHOLD_PATH

_model = None
_threshold = None


def load():
    global _model, _threshold
    _model = load_model(MODEL_PATH)
    if THRESHOLD_PATH.exists():
        _threshold = float(json.loads(THRESHOLD_PATH.read_text())["threshold"])
    else:
        _threshold = 0.25


def get_threshold() -> float:
    return _threshold


def run(img, category: str = "bottle"):
    return run_model(_model, img, category=category)          # (score, anomaly_map)


def decide(score: float, threshold: float = None) -> str:
    th = threshold if threshold is not None else _threshold
    return "DEFECTIVE" if score > th else "NORMAL"
