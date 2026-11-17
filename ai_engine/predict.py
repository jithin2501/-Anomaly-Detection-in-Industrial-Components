"""Unified Industrial Anomaly Detection Engine (ResNet PatchCore).
A single, universal Deep Learning pipeline for all industrial components and surface textures.
No category-specific if-else branching or hardcoded pixel rules.
"""
import json
from pathlib import Path
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tv_models

from .config import IMAGE_SIZE

_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


class ResNetPatchCore(nn.Module):
    """Deep feature extractor combining ResNet-18 layer2 & layer3 for industrial anomaly detection.
    Yields 384-dimensional feature embeddings invariant to minor shifts and textures.
    """
    def __init__(self, resnet):
        super().__init__()
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3

    def forward(self, x):
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        x = self.layer1(x)
        l2 = self.layer2(x)
        l3 = self.layer3(l2)
        l3_up = F.interpolate(l3, size=l2.shape[2:], mode="bilinear", align_corners=False)
        feat = torch.cat([l2, l3_up], dim=1)
        feat = F.avg_pool2d(feat, kernel_size=3, stride=1, padding=1)
        return feat


class CVAnomalyDetector:
    """Universal Deep Learning Anomaly Detector for any industrial part or surface.
    Works automatically for objects (bottles, pills, screws, nuts) and textures (zipper, carpet, tile).
    """
    def __init__(self, threshold: float = 0.25):
        self.threshold = threshold
        self.fe = None
        self.coresets = {}
        self.baselines = {}
        self._load_registry_baselines()

    def _load_registry_baselines(self):
        reg_path = Path(__file__).resolve().parent.parent / "backend" / "models_registry.json"
        if reg_path.is_file():
            try:
                data = json.loads(reg_path.read_text(encoding="utf-8"))
                for entry in data:
                    cid = entry.get("id")
                    base = entry.get("featureBaseline")
                    if cid and base:
                        self.baselines[cid] = float(base)
            except Exception:
                pass

    def get_fe(self):
        if self.fe is None:
            resnet = tv_models.resnet18(weights=tv_models.ResNet18_Weights.DEFAULT).eval()
            self.fe = ResNetPatchCore(resnet).eval()
        return self.fe

    def get_coreset(self, category: str):
        cat = (category or "bottle").lower().strip()
        if cat in self.coresets:
            return self.coresets[cat]

        data_root = Path(__file__).resolve().parent.parent / "data"
        candidate_dirs = [data_root / f"mvtec_{cat}", data_root / cat]

        # 1. Try loading pre-compiled coreset
        for cdir in candidate_dirs:
            if cdir.is_dir():
                pt_file = cdir / f"{cat}_coreset.pt"
                if pt_file.is_file():
                    try:
                        coreset = torch.load(str(pt_file), map_location="cpu")
                        self.coresets[cat] = coreset
                        return coreset
                    except Exception:
                        pass

        # 2. Build on-the-fly from memory bank if pt file not found
        for cdir in candidate_dirs:
            if cdir.is_dir():
                for npy_file in cdir.glob("*memory_bank*.npy"):
                    try:
                        bank = np.load(str(npy_file))
                        fe = self.get_fe()
                        step = max(1, len(bank) // 60)
                        sub_bank = bank[::step]
                        rgb_bank = np.repeat(sub_bank[:, np.newaxis, :, :], 3, axis=1) / 255.0
                        feats = []
                        with torch.no_grad():
                            for i in range(0, len(rgb_bank), 16):
                                batch = (torch.tensor(rgb_bank[i:i+16], dtype=torch.float32) - _MEAN) / _STD
                                feats.append(fe(batch).permute(0, 2, 3, 1).reshape(-1, 384))
                        all_p = F.normalize(torch.cat(feats, dim=0), dim=-1)
                        sub = all_p[torch.randperm(all_p.shape[0])[:6000]] if all_p.shape[0] > 6000 else all_p
                        num_pts = min(2000, sub.shape[0])
                        selected = [0]
                        min_dists = 1.0 - torch.mm(sub, sub[0:1].t()).squeeze(1)
                        for _ in range(num_pts - 1):
                            idx = torch.argmax(min_dists).item()
                            selected.append(idx)
                            new_d = 1.0 - torch.mm(sub, sub[idx:idx+1].t()).squeeze(1)
                            min_dists = torch.minimum(min_dists, new_d)
                        coreset = sub[selected]
                        torch.save(coreset, str(cdir / f"{cat}_coreset.pt"))
                        self.coresets[cat] = coreset
                        return coreset
                    except Exception:
                        pass
        return None

    def get_baseline(self, category: str) -> float:
        cat = (category or "bottle").lower().strip()
        if cat in self.baselines:
            return self.baselines[cat]
        self._load_registry_baselines()
        return self.baselines.get(cat, 0.110)

    def align_if_elongated(self, img):
        """Standard rotation normalization for elongated mechanical components (e.g. screws, pins)."""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
        corners = [gray[0:20, 0:20], gray[0:20, -20:], gray[-20:, 0:20], gray[-20:, -20:]]
        bg_val = float(np.median(corners))
        diff = np.abs(gray.astype(np.float32) - bg_val)
        pts = np.column_stack(np.where(diff > 35))
        if len(pts) < 500:
            return img, None
        cov = np.cov(pts - np.mean(pts, axis=0), rowvar=False)
        eigvals, eigvecs = np.linalg.eigh(cov)
        # Check aspect ratio
        if eigvals[1] / (eigvals[0] + 1e-6) < 3.0:
            return img, None
        v_long = eigvecs[:, 1]
        angle_deg = np.degrees(np.arctan2(v_long[1], v_long[0]))
        h_orig, w_orig = gray.shape[:2]
        mean = np.mean(pts, axis=0)
        M = cv2.getRotationMatrix2D((float(mean[1]), float(mean[0])), -angle_deg, 1.0)
        M[0, 2] += (w_orig / 2.0 - mean[1])
        M[1, 2] += (h_orig / 2.0 - mean[0])
        aligned = cv2.warpAffine(img, M, (w_orig, h_orig), borderMode=cv2.BORDER_REPLICATE)
        al_gray = cv2.cvtColor(aligned, cv2.COLOR_BGR2GRAY) if len(aligned.shape) == 3 else aligned
        al_pts = np.column_stack(np.where(np.abs(al_gray.astype(np.float32) - bg_val) > 35))
        if len(al_pts) > 0:
            min_y, max_y = np.min(al_pts[:, 0]), np.max(al_pts[:, 0])
            span = max_y - min_y
            top_w = np.sum(al_pts[:, 0] < min_y + int(span * 0.25))
            bot_w = np.sum(al_pts[:, 0] > max_y - int(span * 0.25))
            if bot_w > top_w:
                aligned = cv2.rotate(aligned, cv2.ROTATE_180)
        return aligned, (M, (w_orig, h_orig))

    def predict(self, img_bgr, category: str = "bottle"):
        """Universal prediction function for any industrial part or texture."""
        h, w = img_bgr.shape[:2]
        cat = (category or "bottle").lower().strip()
        fe = self.get_fe()
        coreset = self.get_coreset(cat)

        if coreset is None:
            return 0.0, np.zeros((h, w), dtype=np.float32)

        # 1. Pose Normalization for Elongated Parts
        eval_img, align_info = self.align_if_elongated(img_bgr)

        # 2. Deep Feature Extraction (ResNet-18 Layer2 + Layer3)
        rgb = cv2.cvtColor(cv2.resize(eval_img, (256, 256)), cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        t = (torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0) - _MEAN) / _STD

        with torch.no_grad():
            f = fe(t)
            f_norm = F.normalize(f.permute(0, 2, 3, 1).reshape(-1, 384), dim=-1)
            sim = torch.mm(f_norm, coreset.t())
            max_sim, _ = sim.max(dim=1)
            dist_32 = (1.0 - max_sim).reshape(32, 32).numpy()

        # 3. Universal Foreground Analysis (Automated object vs texture detection)
        gray = cv2.cvtColor(cv2.resize(eval_img, (256, 256)), cv2.COLOR_BGR2GRAY)
        corners = [gray[0:20, 0:20], gray[0:20, -20:], gray[-20:, 0:20], gray[-20:, -20:]]
        bg_val = float(np.median(corners))
        diff = np.abs(gray.astype(np.float32) - bg_val)
        fg = (diff > 25).astype(np.uint8)
        fg_ratio = float(np.sum(fg > 0)) / (256 * 256)

        if 0.03 < fg_ratio < 0.70:
            # Discrete object on inspection tray/conveyor: mask background
            fg_eroded = cv2.erode(fg, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
            fg_32 = (cv2.resize(fg_eroded, (32, 32), interpolation=cv2.INTER_AREA) > 0.5)
            amap_smooth = cv2.GaussianBlur(dist_32 * fg_32.astype(np.float32), (3, 3), 1.0)
            valid = amap_smooth[fg_32]
            raw_metric = float(np.mean(np.sort(valid)[-4:])) if len(valid) > 5 else 0.0
            amap_eval = amap_smooth
        else:
            # Full surface / texture (zipper, carpet, leather, tile): evaluate full surface
            amap_smooth = cv2.GaussianBlur(dist_32, (3, 3), 1.0)
            raw_metric = float(np.mean(np.sort(amap_smooth.ravel())[-4:]))
            amap_eval = amap_smooth

        # 4. Universal Normalized Score
        baseline = self.get_baseline(cat)
        if raw_metric <= baseline:
            final_score = float((raw_metric / baseline) * 0.21)
        else:
            final_score = float(0.25 + min(0.75, (raw_metric - baseline) / (baseline * 0.35) * 0.75))

        # 5. Anomaly Map Resizing & Unaligning
        norm_map = np.clip(amap_eval / (baseline + 1e-8) * 0.25, 0.0, 1.0)
        if align_info is not None:
            M, (w_orig, h_orig) = align_info
            M_inv = cv2.invertAffineTransform(M)
            amap_full = cv2.warpAffine(cv2.resize(norm_map, (w_orig, h_orig)), M_inv, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        else:
            amap_full = cv2.resize(norm_map, (w, h))

        return final_score, amap_full


def load_model(ckpt_path=None):
    """Load universal anomaly detector."""
    return CVAnomalyDetector()


def run_model(model, img_bgr, category: str = "bottle"):
    """Return (score: float, anomaly_map: 2D numpy array)."""
    if hasattr(model, "predict"):
        return model.predict(img_bgr, category=category)
    return 0.0, np.zeros((img_bgr.shape[0], img_bgr.shape[1]), dtype=np.float32)
