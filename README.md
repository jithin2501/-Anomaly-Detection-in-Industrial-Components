# Industrial Anomaly Detection App

Upload an image -> get DEFECTIVE / NORMAL (from a validated threshold) + an anomaly heatmap.
Model: PatchCore (Anomalib). Backend: FastAPI. Frontend: React + Tailwind (Vite).

## 1. Setup
```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r ai_engine/requirements.txt -r backend/requirements.txt
```

## 2. Create the model (run from the project root)
```bash
python -m ai_engine.train       # downloads MVTec AD (category in config.py), builds memory bank
python -m ai_engine.evaluate    # computes AUROC + F1-optimal threshold -> weights/<category>/threshold.json
```
Outputs: `ai_engine/weights/bottle/model.ckpt` and `threshold.json`.

Change category: `CATEGORY=screw python -m ai_engine.train` (and evaluate), then set
`MODEL_PATH` / `THRESHOLD_PATH` in `backend/.env`.

If auto-download fails, download MVTec AD manually and place a category at
`data/mvtec_ad/bottle/{train/good,test,ground_truth}`.
(Do not create an empty `bottle/` folder beforehand, or auto-download is skipped.)

## 3. Run backend
```bash
cd backend
uvicorn main:app --reload --port 8000
```
API docs: http://localhost:8000/docs  (POST /predict, GET /history)

## 4. Run frontend
```bash
cd frontend
npm install
npm run dev          # http://localhost:5173
```

## Notes
- Anomalib's API changes between versions; this code targets anomalib >= 2.0 (falls back to `MVTec` for 1.x).
  If `predict.py` fails to load the checkpoint, check the docs for your installed version.
- Image conditions (camera angle, lighting, background) must match the training data.
- One model per product category.
- MVTec AD is licensed CC BY-NC-SA (non-commercial).


1. Activate Virtual Environment & Run Backend
bash
cd /c/Users/jithi/Downloads/anomaly-app
source .venv/Scripts/activate
cd anomaly-app/backend
uvicorn main:app --reload --port 8000
2. Run Frontend
bash
cd /c/Users/jithi/Downloads/anomaly-app/anomaly-app/frontend
npm run dev



Step 1: Model Architecture

"We developed a custom unsupervised anomaly detection model based on a Memory-Bank and Nearest-Neighbor approach, inspired by PatchCore. Our memory bank contains 209 normal reference images."

Step 2: Image Preprocessing

"First, we use OpenCV to resize the uploaded image to 256 × 256 pixels. Then we use cv2.cvtColor() with cv2.COLOR_BGR2GRAY to convert the three-channel BGR image into a single-channel grayscale image."

Step 3: Nearest-Neighbor Comparison

"Next, we compare the uploaded grayscale image against all 209 normal reference images stored in our memory bank.

We calculate the absolute pixel-wise differences and select the minimum difference at each pixel location.

If a particular region has a large difference even from its closest normal reference, that region is considered potentially anomalous."

Step 4: Residual Distance Map

"Using these minimum pixel-wise differences, we generate a Residual Distance Map, which represents how much each pixel location differs from the stored normal references."

Step 5: Noise Reduction and Foreground Masking

"Next, we apply Gaussian Smoothing to reduce unwanted pixel noise.

Then we perform Foreground Masking to isolate the product region from the background.

We also apply Morphological Erosion to suppress unwanted boundary reflections."

Step 6: Dual-Scale Pooling

"After that, we calculate the anomaly score using two pooling methods.

Micro-Scale Pooling selects the top 50 highest anomaly-valued pixels and calculates their average.

Macro-Scale Pooling selects the top 200 highest anomaly-valued pixels and calculates their average."

Step 7: Weighted Aggregation and Normalization

"We combine the Micro and Macro averages using weighted aggregation, giving 60% weight to Micro and 40% to Macro.

Then we divide the combined metric by 35 and clip the result to obtain a normalized anomaly score between 0 and 1."

Step 8: Binary Classification

"We compare the final anomaly score against a predefined threshold, which is 0.25 by default.

If the score is greater than or equal to the threshold, the image is classified as DEFECTIVE. Otherwise, it is classified as NORMAL."

Step 9: Heatmap Generation

"Finally, we use OpenCV's JET Colormap to convert anomaly intensity values into different colors.

High anomaly values appear in red, while lower values appear in cooler colors.

We then use Alpha Blending to overlay the heatmap onto the original image, making the suspicious regions easier to identify."