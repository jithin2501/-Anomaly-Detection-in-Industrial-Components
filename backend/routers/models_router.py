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
    rgb_imgs = []
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
                rgb_imgs.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
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

    # Automatically compile deep ResNet feature coreset and calibrate baseline
    coreset = None
    feature_baseline = 0.110
    try:
        import torch
        from ai_engine.predict import ResNetPatchCore, _MEAN, _STD
        import torchvision.models as tv_models

        fe = ResNetPatchCore(tv_models.resnet18(weights=tv_models.ResNet18_Weights.DEFAULT).eval())
        step = max(1, len(rgb_imgs) // 60)
        selected_imgs = rgb_imgs[::step]

        batch_tensors = []
        for im in selected_imgs:
            im_res = cv2.resize(im, (256, 256))
            t = torch.from_numpy(im_res.astype(np.float32) / 255.0).permute(2, 0, 1)
            batch_tensors.append(t)

        feats = []
        with torch.no_grad():
            for i in range(0, len(batch_tensors), 16):
                batch = torch.stack(batch_tensors[i:i + 16])
                batch = (batch - _MEAN) / _STD
                f = fe(batch)
                B, C, H, W = f.shape
                f_flat = f.permute(0, 2, 3, 1).reshape(-1, C)
                feats.append(f_flat)

        all_patches = torch.cat(feats, dim=0)
        all_patches = torch.nn.functional.normalize(all_patches, dim=-1)

        N_p = all_patches.shape[0]
        sub_patches = all_patches[torch.randperm(N_p)[:6000]] if N_p > 6000 else all_patches
        num_points = min(2000, sub_patches.shape[0])
        selected = [0]
        min_dists = 1.0 - torch.mm(sub_patches, sub_patches[0:1].t()).squeeze(1)
        for _ in range(num_points - 1):
            idx = torch.argmax(min_dists).item()
            selected.append(idx)
            new_dist = 1.0 - torch.mm(sub_patches, sub_patches[idx:idx + 1].t()).squeeze(1)
            min_dists = torch.minimum(min_dists, new_dist)

        coreset = sub_patches[selected]
        coreset_path = target_dir / f"{clean_name}_coreset.pt"
        torch.save(coreset, str(coreset_path))

        # Measure baseline distance on training images
        train_dists = []
        with torch.no_grad():
            for t_img in batch_tensors[:20]:
                t_in = ((t_img.unsqueeze(0) - _MEAN) / _STD)
                f_test = torch.nn.functional.normalize(fe(t_in).permute(0, 2, 3, 1).reshape(-1, 384), dim=-1)
                dist_t = (1.0 - torch.mm(f_test, coreset.t()).max(dim=1)[0]).numpy()
                train_dists.append(float(np.mean(np.sort(dist_t)[-4:])))
        if train_dists:
            feature_baseline = round(float(np.percentile(train_dists, 95)) * 1.15, 4)
    except Exception:
        coreset = None

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
        "featureBaseline": feature_baseline,
        "numImages": len(refs),
    }

    if existing:
        existing.update(model_entry)
    else:
        registry.append(model_entry)

    save_registry(registry)

    # Register into live memory bank & coreset in inference service
    try:
        from services import inference_service
        if hasattr(inference_service, "_model"):
            if hasattr(inference_service._model, "coresets") and coreset is not None:
                inference_service._model.coresets[clean_name] = coreset
            if hasattr(inference_service._model, "baselines"):
                inference_service._model.baselines[clean_name] = feature_baseline
    except Exception:
        pass

    return {
        "success": True,
        "message": f"Successfully compiled {len(refs)} normal images for '{clean_name.upper()}'.",
        "model": model_entry,
    }
