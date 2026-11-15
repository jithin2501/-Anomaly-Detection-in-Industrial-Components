"""Compute AUROC and an F1-optimal threshold on the test set.
Run from project root:  python -m ai_engine.evaluate
"""
import json

import cv2
import numpy as np
from sklearn.metrics import precision_recall_curve, roc_auc_score

from .config import CATEGORY, DATA_ROOT, MODEL_PATH, RESULTS_DIR, THRESHOLD_PATH
from .predict import load_model, run_model
from .utils import list_test_images, make_overlay


def main():
    model = load_model(MODEL_PATH)
    test_dir = DATA_ROOT / CATEGORY / "test"

    scores, labels, items = [], [], []
    for path, label in list_test_images(test_dir):
        img = cv2.imread(str(path))
        score, amap = run_model(model, img)
        scores.append(score)
        labels.append(label)
        items.append((path, img, amap))

    scores, labels = np.array(scores), np.array(labels)
    auroc = float(roc_auc_score(labels, scores))

    prec, rec, thr = precision_recall_curve(labels, scores)
    f1 = 2 * prec[:-1] * rec[:-1] / (prec[:-1] + rec[:-1] + 1e-12)
    best = int(np.argmax(f1))
    threshold = float(thr[best])

    THRESHOLD_PATH.parent.mkdir(parents=True, exist_ok=True)
    THRESHOLD_PATH.write_text(json.dumps({"threshold": threshold}, indent=2))

    RESULTS_DIR.mkdir(exist_ok=True)
    (RESULTS_DIR / "metrics.json").write_text(json.dumps({
        "category": CATEGORY, "image_auroc": auroc,
        "threshold": threshold, "best_f1": float(f1[best]),
        "n_images": int(len(labels)),
    }, indent=2))

    out_dir = RESULTS_DIR / "sample_heatmaps"
    out_dir.mkdir(exist_ok=True)
    for i, (path, img, amap) in enumerate(items[:: max(1, len(items) // 10)]):
        cv2.imwrite(str(out_dir / f"{i:02d}_{path.parent.name}_{path.stem}.png"),
                    make_overlay(img, amap, threshold))

    print(f"Image AUROC: {auroc:.4f} | threshold: {threshold:.4f} | F1: {f1[best]:.4f}")
    print(f"Saved {THRESHOLD_PATH}")


if __name__ == "__main__":
    main()
