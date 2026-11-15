# Industrial Anomaly Detection Platform

An Unsupervised Industrial Surface Anomaly Detection platform inspired by the **PatchCore** Memory-Bank paradigm. Built with **Python, NumPy, OpenCV, FastAPI**, and **React 18 + Vite**.

---

## 🌟 Key Features

- **Unsupervised Learning:** Learns solely from defect-free normal images. No defect labels or manual annotations required.
- **Ultra-Fast CPU Inference:** Vectorized matrix matching in **~35 milliseconds** per image without GPU dependencies.
- **Multi-Product Support:** Built-in support for **Bottle** (209 references) and **Pill** (267 references).
- **Dynamic Model Upload:** Click **"Add Model"** in the web interface to upload any MVTec category archive (`.zip`, `.tar`, `.tar.xz`) to compile a new product memory bank in under 2 seconds.
- **Smart MVTec Archive Filter:** Automatically isolates `train/good` defect-free images from full archives, discarding test defects and masks.
- **Dual-Scale Extreme-Value Energy Pooling:** Accurately catches micro-defects (cracks, tiny contamination pinholes) and macro-defects (broken structures, chips).
- **Localization Heatmaps:** High-resolution OpenCV JET heatmap overlays showing the defect location.
- **Interactive Web Interface:** Real-time adjustable threshold slider ($0.05$ – $0.95$), single image analysis, and multi-image batch inspection.

---

## 🏗️ Architecture & Pipeline

```
[ Input Inspection Image ]
            │
            ▼
[ Step 1: Normalization ] ────► Resized to 256×256 & converted to Grayscale
            │
            ▼
[ Step 2: 1-NN Projection ] ──► Vectorized absolute difference against all normal memories:
                               min_diff = min(|Memory_Bank - Input|)
            │
            ▼
[ Step 3: Noise Suppression ] ─► 9×9 2D Gaussian Kernel smoothing
            │
            ▼
[ Step 4: Product Masking ] ──► Adaptive foreground thresholding & morphological erosion
            │
            ▼
[ Step 5: Dual-Scale Pooling ] ─► Metric = 0.6 * Top50_Mean + 0.4 * Top200_Mean
            │
            ▼
[ Step 6: Normalization ] ────► Calibrated Anomaly Score ∈ [0.00, 1.00]
            │
            ├──► If Score >= Threshold: ❌ DEFECTIVE
            └──► If Score <  Threshold: ✅ NORMAL
            │
            ▼
[ Step 7: Heatmap Blending ] ─► Upscaled to original resolution + OpenCV JET Colormap
```

---

## 🚀 Quickstart Guide

### 1. Start the Backend (FastAPI + AI Engine)

In PowerShell:
```powershell
cd "anomaly-app\backend"
..\..\.venv\Scripts\Activate.ps1
uvicorn main:app --reload --port 8000
```

Or in Git Bash:
```bash
cd /c/Users/jithi/Downloads/anomaly-app
source .venv/Scripts/activate
cd anomaly-app/backend
uvicorn main:app --reload --port 8000
```
> Backend API: **`http://127.0.0.1:8000`** (Swagger docs at `/docs`)

---

### 2. Start the Frontend (React + Vite)

In a second terminal:
```bash
cd anomaly-app/frontend
npm install
npm run dev
```
> Frontend Web App: **`http://localhost:5173`**

---

## 📦 Dataset Download Scripts

To download the full standard MVTec benchmark datasets:

```powershell
# Download all 209 Bottle normal images & defects
python scripts/download_mvtec_bottle.py

# Download all 267 Pill normal images & defects
python scripts/download_mvtec_pill.py
```

---

## 🛠️ Tech Stack

- **Computer Vision & AI:** Python 3.11, NumPy, OpenCV
- **Backend API:** FastAPI, Uvicorn, SQLAlchemy, SQLite, Pydantic
- **Frontend UI:** React 18, Vite, Vanilla CSS (Dark Industrial Glassmorphism)
- **Deployment:** Docker Compose ready
