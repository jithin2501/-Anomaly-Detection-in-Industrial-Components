import os
import io
import json
import zipfile
import tarfile
import re
from pathlib import Path
from fastapi import APIRouter, File, Form, HTTPException, UploadFile
import cv2
import numpy as np

router = APIRouter(prefix="/models", tags=["models"])

REGISTRY_PATH = Path(__file__).resolve().parent.parent / "models_registry.json"
DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"
FRONTEND_PUBLIC = Path(__file__).resolve().parent.parent.parent / "frontend" / "public"


def load_registry():
    if REGISTRY_PATH.is_file():
        try:
            return json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
        except Exception:
            pass
    return [
        {
            "id": "bottle",
            "name": "BOTTLE",
            "subtitle": "MVTec Industrial Glass Bottle",
            "image": "/bottle.png",
            "recommendedThreshold": 0.25,
            "numImages": 209,
        },
        {
            "id": "pill",
            "name": "PILL",
            "subtitle": "MVTec Pharmaceutical Pill & Tablet",
            "image": "/pill.png",
            "recommendedThreshold": 0.25,
            "numImages": 267,
        },
    ]


def save_registry(items):
    REGISTRY_PATH.write_text(json.dumps(items, indent=2), encoding="utf-8")


def extract_normal_images_from_archive(content: bytes, filename: str):
    """
    Extracts images from .zip, .tar, .tar.gz, .tar.xz, .tgz archives.
    Automatically applies Smart MVTec Filtering:
    If the archive contains both normal and defect folders (train/good, test, ground_truth),
    it automatically isolates ONLY the defect-free normal training images.
    """
    valid_exts = (".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff")
    extracted = {}  # filepath -> bytes

    # 1. Try ZIP
    is_zip = False
    try:
        zf = zipfile.ZipFile(io.BytesIO(content))
        for name in zf.namelist():
            if name.lower().endswith(valid_exts) and not name.startswith("__MACOSX"):
                extracted[name] = zf.read(name)
        is_zip = True
    except Exception:
        pass

    # 2. Try TAR (.tar, .tar.gz, .tar.xz, .tgz, .tar.bz2)
    if not is_zip:
        try:
            with tarfile.open(fileobj=io.BytesIO(content), mode="r:*") as tf:
                for member in tf.getmembers():
                    if member.isfile() and member.name.lower().endswith(valid_exts) and not member.name.startswith("__MACOSX"):
                        f = tf.extractfile(member)
                        if f is not None:
                            extracted[member.name] = f.read()
        except Exception as e:
            raise HTTPException(
                status_code=400,
                detail=f"Could not open archive. Supported formats: .zip, .tar, .tar.xz, .tar.gz. Error: {str(e)}"
            )

    if not extracted:
        raise HTTPException(
            status_code=400,
            detail="No valid images found in the uploaded archive (.zip or .tar)."
        )

    # 3. Smart MVTec Filter:
    # If standard MVTec layout exists with train/good, use ONLY train/good!
    train_good_files = {
        k: v for k, v in extracted.items()
        if "train/good" in k.lower().replace("\\", "/") or "train_good" in k.lower()
    }
    if not train_good_files:
        # Also check for 'good' directory while excluding 'test' and 'ground_truth'
        train_good_files = {
            k: v for k, v in extracted.items()
            if "/good/" in k.lower().replace("\\", "/") and "ground_truth" not in k.lower() and "test" not in k.lower()
        }

    if train_good_files:
        # Detected full MVTec archive! Extract ONLY normal training images.
        selected_files = train_good_files
    else:
        # Custom archive or flat folder: Exclude ground_truth masks or defect directories if present
        filtered_files = {
            k: v for k, v in extracted.items()
            if not any(bad in k.lower().replace("\\", "/") for bad in [
                "ground_truth", "mask", "defect", "broken", "contamination", "scratch", "crack"
            ])
        }
        selected_files = filtered_files if filtered_files else extracted

    return selected_files


@router.get("")
def list_models():
    """List all available models in the registry."""
    return load_registry()


@router.post("/add")
async def add_model(
    name: str = Form(...),
    file: UploadFile = File(...)
):
    """
    Accepts a product name and a ZIP or TAR archive containing normal reference images.
    Extracts images, resizes to 256x256, compiles into memory bank, and registers the model.
    """
    clean_name = re.sub(r"[^a-zA-Z0-9_\-]", "", name.strip().lower().replace(" ", "_"))
    if not clean_name:
        raise HTTPException(status_code=400, detail="Invalid product name")

    supported_suffixes = (".zip", ".tar", ".gz", ".xz", ".tgz", ".bz2")
    if not any(file.filename.lower().endswith(s) for s in supported_suffixes):
        raise HTTPException(
            status_code=400,
            detail="Please upload a supported archive format: .zip, .tar, .tar.xz, or .tar.gz"
        )

    content = await file.read()
    image_dict = extract_normal_images_from_archive(content, file.filename)

    refs = []
    first_thumb = None

    for img_name in sorted(image_dict.keys()):
        try:
            raw = image_dict[img_name]
            img = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
            if img is not None:
                if first_thumb is None:
                    first_thumb = img
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                resized = cv2.resize(gray, (256, 256)).astype(np.float32)
                refs.append(resized)
        except Exception:
            continue

    if not refs:
        raise HTTPException(status_code=400, detail="Could not decode any valid images from the archive.")

    # Stack into memory bank (N, 256, 256)
    bank = np.stack(refs, axis=0)

    # Save to data/mvtec_<clean_name>/<clean_name>_memory_bank.npy
    target_dir = DATA_DIR / f"mvtec_{clean_name}"
    target_dir.mkdir(parents=True, exist_ok=True)
    bank_path = target_dir / f"{clean_name}_memory_bank.npy"
    np.save(str(bank_path), bank)

    # Save thumbnail to frontend public directory
    thumb_filename = f"{clean_name}.png"
    thumb_rel_url = f"/{thumb_filename}"
    if first_thumb is not None and FRONTEND_PUBLIC.is_dir():
        try:
            thumb_path = FRONTEND_PUBLIC / thumb_filename
            cv2.imwrite(str(thumb_path), cv2.resize(first_thumb, (400, 400)))
        except Exception:
            thumb_rel_url = "/placeholder.svg"
    else:
        thumb_rel_url = "/placeholder.svg"

    # Register into models registry
    registry = load_registry()
    existing = next((m for m in registry if m["id"] == clean_name), None)
    model_entry = {
        "id": clean_name,
        "name": name.strip().upper(),
        "subtitle": f"Custom MVTec / Factory {name.strip().title()}",
        "image": thumb_rel_url,
        "recommendedThreshold": 0.25,
        "numImages": len(refs),
    }

    if existing:
        existing.update(model_entry)
    else:
        registry.append(model_entry)

    save_registry(registry)

    # Register into live memory bank in inference service
    try:
        from services import inference_service
        if hasattr(inference_service, "_model") and hasattr(inference_service._model, "banks"):
            inference_service._model.banks[clean_name] = bank
    except Exception:
        pass

    return {
        "success": True,
        "message": f"Successfully compiled {len(refs)} normal images for '{clean_name.upper()}'.",
        "model": model_entry,
    }
