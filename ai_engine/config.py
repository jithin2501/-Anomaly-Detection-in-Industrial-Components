import os
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_ROOT = ROOT_DIR / "data" / "mvtec_ad"
CATEGORY = os.getenv("CATEGORY", "bottle")

WEIGHTS_DIR = Path(__file__).resolve().parent / "weights" / CATEGORY
MODEL_PATH = WEIGHTS_DIR / "model.ckpt"
THRESHOLD_PATH = WEIGHTS_DIR / "threshold.json"
RESULTS_DIR = Path(__file__).resolve().parent / "results"

IMAGE_SIZE = 256
BACKBONE = "wide_resnet50_2"
LAYERS = ("layer2", "layer3")
CORESET_RATIO = 0.1
