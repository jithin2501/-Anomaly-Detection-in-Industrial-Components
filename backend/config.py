import os
import sys
from pathlib import Path

from dotenv import load_dotenv

BACKEND_DIR = Path(__file__).resolve().parent
ROOT_DIR = BACKEND_DIR.parent
if str(ROOT_DIR) not in sys.path:          # lets us `import ai_engine`
    sys.path.insert(0, str(ROOT_DIR))

load_dotenv(BACKEND_DIR / ".env")


def _path(value: str) -> Path:
    p = Path(value)
    return p if p.is_absolute() else (BACKEND_DIR / p).resolve()


MODEL_PATH = _path(os.getenv("MODEL_PATH", "../ai_engine/weights/bottle/model.ckpt"))
THRESHOLD_PATH = _path(os.getenv("THRESHOLD_PATH", "../ai_engine/weights/bottle/threshold.json"))
DB_URL = os.getenv("DB_URL", "sqlite:///./inspections.db")
CORS_ORIGINS = os.getenv("CORS_ORIGINS", "http://localhost:5173").split(",")
ALLOWED_TYPES = {"image/png", "image/jpeg"}
MAX_UPLOAD_MB = 10
