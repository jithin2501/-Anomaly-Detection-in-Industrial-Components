import config
from fastapi import HTTPException
from services import heatmap_service, history_service, image_service, inference_service


async def handle_predict(file, db, custom_threshold: float = None, category: str = "bottle"):
    if file.content_type not in config.ALLOWED_TYPES:
        raise HTTPException(400, "Only PNG or JPEG images are allowed")

    data = await file.read()
    if len(data) > config.MAX_UPLOAD_MB * 1024 * 1024:
        raise HTTPException(413, f"File larger than {config.MAX_UPLOAD_MB} MB")

    try:
        img = image_service.decode(data)
    except ValueError:
        raise HTTPException(400, "Could not read the image file")

    score, amap = inference_service.run(img, category=category)
    threshold = float(custom_threshold) if custom_threshold is not None else inference_service.get_threshold()
    label = "DEFECTIVE" if score > threshold else "NORMAL"
    overlay = heatmap_service.make_overlay(img, amap, threshold)

    history_service.save(db, file.filename or "upload", label, score, threshold)

    return {
        "label": label,
        "score": float(score),
        "threshold": float(threshold),
        "heatmap": image_service.to_base64(overlay),
    }
