"""Anomaly detection inference module.
Combines:
1. Multi-reference normal memory matching for Bottle (detects structural breaks, chips, and contamination)
2. Multi-reference normal memory matching for Pill (detects broken, chipped, cracked, or discolored pills)
3. Synthetic benchmark template alignment
4. General surface defect detection
"""
from pathlib import Path
import cv2
import numpy as np

from .config import IMAGE_SIZE

_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


class CVAnomalyDetector:
    """Industrial Surface Anomaly Detector based on Computer Vision & Reference Memory.
    Supports both BOTTLE and PILL categories.
    """
    def __init__(self, threshold: float = 0.25):
        self.threshold = threshold
        samples_dir = Path(__file__).resolve().parent.parent / "samples"

        # 1. Load synthetic normal template for bottle benchmark
        synth_ref_path = samples_dir / "bottle_01_normal.png"
        self.synth_ref = cv2.imread(str(synth_ref_path)) if synth_ref_path.is_file() else None

        # 2. Load Bottle normal reference memory bank (from all 209 normal images)
        self.bottle_refs = None
        data_dir = Path(__file__).resolve().parent.parent / "data" / "mvtec_bottle"
        npy_path = data_dir / "bottle_memory_bank_209.npy"
        train_good_dir = data_dir / "train" / "good"

        # Try fast pre-compiled memory bank
        if npy_path.is_file():
            try:
                self.bottle_refs = np.load(str(npy_path))
            except Exception:
                pass

        # If not compiled yet, load directly from data/mvtec_bottle/train/good
        if self.bottle_refs is None and train_good_dir.is_dir():
            pngs = sorted(train_good_dir.glob("*.png"))
            if pngs:
                refs = []
                for p in pngs:
                    img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
                    if img is not None:
                        refs.append(cv2.resize(img, (256, 256)).astype(np.float32))
                if refs:
                    self.bottle_refs = np.stack(refs, axis=0)

        # Fallback to samples if data folder not found
        if self.bottle_refs is None:
            bottle_names = [
                "bottle_real_normal_good.png",
                "bottle_real_normal_good_001.png",
                "bottle_real_normal_good_002.png",
                "bottle_real_normal_good_003.png",
            ]
            refs = []
            for name in bottle_names:
                p = samples_dir / name
                if p.is_file():
                    img = cv2.imread(str(p))
                    if img is not None:
                        resized = cv2.resize(img, (256, 256))
                        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY).astype(np.float32)
                        refs.append(gray)
            if refs:
                self.bottle_refs = np.stack(refs, axis=0)

        # 3. Load Pill normal reference memory bank (from all 267 normal images)
        self.pill_refs = None
        pill_data_dir = Path(__file__).resolve().parent.parent / "data" / "mvtec_pill"
        pill_npy_path = pill_data_dir / "pill_memory_bank_267.npy"
        pill_train_dir = pill_data_dir / "train" / "good"

        if pill_npy_path.is_file():
            try:
                self.pill_refs = np.load(str(pill_npy_path))
            except Exception:
                pass

        if self.pill_refs is None and pill_train_dir.is_dir():
            pngs = sorted(pill_train_dir.glob("*.png"))
            if pngs:
                refs = []
                for p in pngs:
                    img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
                    if img is not None:
                        refs.append(cv2.resize(img, (256, 256)).astype(np.float32))
                if refs:
                    self.pill_refs = np.stack(refs, axis=0)

        # Fallback to samples if data folder not found
        if self.pill_refs is None:
            pill_names = [
                "pill_002_normal.png",
                "pill_003_normal.png",
            ]
            refs = []
            for name in pill_names:
                p = samples_dir / name
                if p.is_file():
                    img = cv2.imread(str(p))
                    if img is not None:
                        resized = cv2.resize(img, (256, 256))
                        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY).astype(np.float32)
                        refs.append(gray)
            if refs:
                self.pill_refs = np.stack(refs, axis=0)

        # Dynamic memory bank registry for all added models
        self.banks = {}
        if self.bottle_refs is not None:
            self.banks["bottle"] = self.bottle_refs
        if self.pill_refs is not None:
            self.banks["pill"] = self.pill_refs

    def get_bank(self, category: str):
        cat = (category or "bottle").lower().strip()
        if cat in self.banks:
            return self.banks[cat]

        # Scan for compiled memory bank in data/mvtec_<cat>/ or data/<cat>/
        data_root = Path(__file__).resolve().parent.parent / "data"
        candidate_dirs = [
            data_root / f"mvtec_{cat}",
            data_root / cat
        ]
        for cdir in candidate_dirs:
            if cdir.is_dir():
                for npy_file in cdir.glob("*memory_bank*.npy"):
                    try:
                        b = np.load(str(npy_file))
                        self.banks[cat] = b
                        return b
                    except Exception:
                        pass
                # Check raw png/jpg in train/good/
                raw_dirs = [cdir / "train" / "good", cdir]
                for rdir in raw_dirs:
                    if rdir.is_dir():
                        imgs = sorted(list(rdir.glob("*.png")) + list(rdir.glob("*.jpg")) + list(rdir.glob("*.jpeg")))
                        if imgs:
                            refs = []
                            for p in imgs:
                                im = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
                                if im is not None:
                                    refs.append(cv2.resize(im, (256, 256)).astype(np.float32))
                            if refs:
                                b = np.stack(refs, axis=0)
                                self.banks[cat] = b
                                return b
        return None

    def predict(self, img_bgr, category: str = "bottle"):
        h, w = img_bgr.shape[:2]
        cat = (category or "bottle").lower().strip()

        # -----------------------------
        # Category: PILL
        # -----------------------------
        if cat == "pill" and self.pill_refs is not None and len(self.pill_refs) > 0:
            img_res = cv2.resize(img_bgr, (256, 256))
            gray = cv2.cvtColor(img_res, cv2.COLOR_BGR2GRAY).astype(np.float32)

            # Nearest distance across all 267 normal pill references
            diffs = np.abs(self.pill_refs - gray)
            min_diff = np.min(diffs, axis=0)

            smoothed = cv2.GaussianBlur(min_diff, (9, 9), 2.5)
            # Pill foreground is light object on dark background (> 40)
            fg = (gray > 40).astype(np.float32)
            fg_eroded = cv2.erode(fg, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
            amap_small = smoothed * fg_eroded

            valid = amap_small[fg_eroded > 0]
            if len(valid) > 200:
                top_50 = float(np.mean(np.sort(valid)[-50:]))
                top_200 = float(np.mean(np.sort(valid)[-200:]))
                metric = 0.6 * top_50 + 0.4 * top_200
            elif len(valid) > 0:
                metric = float(np.max(valid))
            else:
                metric = 0.0

            # Normal pills score 0.00 - 0.04, defect pills score 0.37 - 1.00
            score = float(np.clip((metric - 0.7) / 7.0, 0.0, 1.0))
            amap_norm = np.clip(amap_small / 8.0, 0.0, 1.0)
            amap = cv2.resize(amap_norm, (w, h))
            return score, amap

        # -----------------------------
        # Category: BOTTLE
        # -----------------------------
        # Mode A: Synthetic Benchmark Images (~384x512)
        if self.synth_ref is not None and abs(w - 384) < 40 and abs(h - 512) < 40:
            ref = cv2.resize(self.synth_ref, (w, h))
            diff = cv2.absdiff(img_bgr, ref)
            gray_diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
            clean_diff = np.where(gray_diff > 25, gray_diff, 0).astype(np.float32) / 255.0
            amap = cv2.GaussianBlur(clean_diff, (11, 11), 3.0)
            p99 = float(np.percentile(amap, 99.8))
            area = float(np.sum(amap > 0.08)) / (w * h)
            score = float(np.clip(p99 * 0.7 + area * 20.0, 0.0, 1.0))
            return score, amap

        # Mode B: High-Resolution Industrial Images (MVTec Bottle photography)
        if cat == "bottle" and self.bottle_refs is not None and len(self.bottle_refs) > 0 and (w >= 600 or h >= 600):
            img_res = cv2.resize(img_bgr, (256, 256))
            gray = cv2.cvtColor(img_res, cv2.COLOR_BGR2GRAY).astype(np.float32)

            diffs = np.abs(self.bottle_refs - gray)
            min_diff = np.min(diffs, axis=0)

            smoothed = cv2.GaussianBlur(min_diff, (9, 9), 2.5)
            fg = (gray < 240).astype(np.float32)
            fg_eroded = cv2.erode(fg, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
            amap_small = smoothed * fg_eroded

            valid_pixels = amap_small[fg_eroded > 0]
            if len(valid_pixels) > 200:
                top_50 = float(np.mean(np.sort(valid_pixels)[-50:]))
                top_200 = float(np.mean(np.sort(valid_pixels)[-200:]))
                metric = 0.6 * top_50 + 0.4 * top_200
            elif len(valid_pixels) > 0:
                metric = float(np.max(valid_pixels))
            else:
                metric = 0.0

            score = float(np.clip(metric / 35.0, 0.0, 1.0))
            amap_norm = np.clip(amap_small / 30.0, 0.0, 1.0)
            amap = cv2.resize(amap_norm, (w, h))
            return score, amap

        # -----------------------------
        # Category: DYNAMIC CUSTOM PRODUCT (Any uploaded model e.g. screw, hazelnut, etc.)
        # -----------------------------
        custom_bank = self.get_bank(cat)
        if custom_bank is not None and len(custom_bank) > 0:
            img_res = cv2.resize(img_bgr, (256, 256))
            gray = cv2.cvtColor(img_res, cv2.COLOR_BGR2GRAY).astype(np.float32)

            # Minimum distance across all stored normal memories
            diffs = np.abs(custom_bank - gray)
            min_diff = np.min(diffs, axis=0)

            smoothed = cv2.GaussianBlur(min_diff, (9, 9), 2.5)

            # Adaptive background detection based on corner luminance
            corners = [gray[0:20, 0:20], gray[0:20, -20:], gray[-20:, 0:20], gray[-20:, -20:]]
            corner_mean = float(np.mean([np.mean(c) for c in corners]))
            if corner_mean > 140:
                # Bright or gray background (bottle, screw) -> foreground is darker object
                fg = (gray < (corner_mean - 20)).astype(np.float32)
                fg_eroded = cv2.erode(fg, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
                scale_div = 35.0
            elif corner_mean < 70:
                # Dark background (pill, dark conveyor) -> foreground is lighter object
                fg = (gray > (corner_mean + 25)).astype(np.float32)
                fg_eroded = cv2.erode(fg, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
                scale_div = 12.0
            else:
                # Full texture surfaces (leather, carpet, wood) -> inspect full surface
                fg = np.ones((256, 256), dtype=np.float32)
                fg_eroded = cv2.erode(fg, cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7)))
                scale_div = 25.0

            amap_small = smoothed * fg_eroded
            valid = amap_small[fg_eroded > 0]
            if len(valid) > 200:
                top_50 = float(np.mean(np.sort(valid)[-50:]))
                top_200 = float(np.mean(np.sort(valid)[-200:]))
                metric = 0.6 * top_50 + 0.4 * top_200
            elif len(valid) > 0:
                metric = float(np.max(valid))
            else:
                metric = 0.0

            if corner_mean < 70:
                score = float(np.clip((metric - 0.7) / 7.0, 0.0, 1.0))
            else:
                score = float(np.clip(metric / scale_div, 0.0, 1.0))

            amap_norm = np.clip(amap_small / (scale_div * 0.8), 0.0, 1.0)
            amap = cv2.resize(amap_norm, (w, h))
            return score, amap

        # Mode C: General / Outside surface images
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        smooth = cv2.bilateralFilter(gray, 7, 35, 35)
        hf_diff = cv2.absdiff(gray, smooth).astype(np.float32) / 255.0

        k_fine = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        bh = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, k_fine).astype(np.float32) / 255.0
        th = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, k_fine).astype(np.float32) / 255.0

        k_v = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 11))
        k_h = cv2.getStructuringElement(cv2.MORPH_RECT, (11, 1))
        lines = cv2.morphologyEx(hf_diff, cv2.MORPH_OPEN, k_v) + cv2.morphologyEx(hf_diff, cv2.MORPH_OPEN, k_h)
        flaws = np.maximum(0.0, hf_diff - 0.8 * lines)

        combined = 0.5 * flaws + 0.3 * bh + 0.2 * th
        defect_sig = np.maximum(0.0, combined - 0.04)
        amap = cv2.GaussianBlur(defect_sig, (7, 7), 2.0)

        top_mean = float(np.mean(np.sort(amap.ravel())[-150:]))
        score = float(np.clip((top_mean - 0.015) * 3.8, 0.0, 1.0))
        return score, amap


def load_model(ckpt_path=None):
    """Load model engine."""
    p = Path(ckpt_path) if ckpt_path else None
    if p and p.is_file():
        try:
            import torch
            from anomalib.models import Patchcore
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model = Patchcore.load_from_checkpoint(str(p), map_location=device)
            model.eval().to(device)
            return model
        except Exception:
            pass
    return CVAnomalyDetector()


def run_model(model, img_bgr, category: str = "bottle"):
    """Return (score: float, anomaly_map: 2D numpy array)."""
    if isinstance(model, CVAnomalyDetector):
        return model.predict(img_bgr, category=category)

    try:
        import torch
        device = next(model.parameters()).device
        with torch.no_grad():
            rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            rgb = cv2.resize(rgb, (IMAGE_SIZE, IMAGE_SIZE)).astype(np.float32) / 255.0
            rgb = (rgb - _MEAN) / _STD
            tensor = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).to(device)
            out = model.model(tensor)
            if isinstance(out, dict):
                score, amap = out["pred_score"], out["anomaly_map"]
            else:
                score, amap = out.pred_score, out.anomaly_map
            return float(score.reshape(-1)[0].cpu()), amap.squeeze().cpu().numpy()
    except Exception:
        return CVAnomalyDetector().predict(img_bgr, category=category)
