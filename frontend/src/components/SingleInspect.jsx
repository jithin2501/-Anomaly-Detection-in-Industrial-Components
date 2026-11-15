import { useState, useRef } from "react";
import { predict } from "../api/client.js";
import ScoreBar from "./ScoreBar.jsx";
import HeatmapViewer from "./HeatmapViewer.jsx";

export default function SingleInspect({ material, threshold }) {
  const fileInputRef = useRef(null);
  const [selectedFile, setSelectedFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  function handleFileChange(e) {
    const file = e.target.files?.[0];
    if (!file) return;
    setSelectedFile(file);
    setPreview(URL.createObjectURL(file));
    setResult(null);
    setError(null);
  }

  async function handleAnalyze() {
    if (!selectedFile) return;
    setLoading(true);
    setError(null);
    try {
      const res = await predict(selectedFile, threshold, material?.id || "bottle");
      setResult(res);
    } catch (err) {
      setError(err.message || "Failed to analyze image");
    } finally {
      setLoading(false);
    }
  }

  function handleReset() {
    setSelectedFile(null);
    setPreview(null);
    setResult(null);
    setError(null);
  }

  const isAnomaly = result ? result.score > threshold : false;

  return (
    <div>
      <div className="page-header-row">
        <h2 className="page-section-title">Single Image Analysis • {material.name}</h2>
      </div>

      {error && (
        <div className="error-box" style={{ maxWidth: 540, margin: "0 auto 20px" }}>
          <span>⚠️ {error}</span>
        </div>
      )}

      <div className="single-inspect-card">
        {/* Image Preview / Drop Area */}
        <div
          className="single-preview-frame"
          style={{ cursor: "pointer" }}
          onClick={() => fileInputRef.current?.click()}
        >
          {preview ? (
            <img src={preview} alt="Selected sample" />
          ) : (
            <div style={{ textAlign: "center", color: "#6b7280", padding: 20 }}>
              <svg
                width="48"
                height="48"
                fill="none"
                stroke="currentColor"
                viewBox="0 0 24 24"
                style={{ margin: "0 auto 10px", display: "block", color: "#38bdf8" }}
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth="1.5"
                  d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z"
                />
              </svg>
              <p style={{ fontWeight: 600, color: "#e5e5e5", fontSize: "0.95rem" }}>
                Click to select a single image
              </p>
              <p style={{ fontSize: "0.78rem", color: "#9ca3af" }}>
                Supports PNG, JPG, JPEG
              </p>
            </div>
          )}
        </div>

        <input
          ref={fileInputRef}
          type="file"
          accept="image/png,image/jpeg,image/jpg"
          style={{ display: "none" }}
          onChange={handleFileChange}
        />

        {/* Action Button Centered */}
        {!selectedFile ? (
          <button
            className="single-analyze-btn"
            onClick={() => fileInputRef.current?.click()}
          >
            Select File
          </button>
        ) : (
          <button
            className="single-analyze-btn"
            disabled={loading}
            onClick={handleAnalyze}
          >
            {loading ? "Analyzing..." : "Analyze"}
          </button>
        )}

        {/* Result Box */}
        {result && (
          <div className="single-result-summary">
            <p className={`single-result-status ${isAnomaly ? "anomaly" : "normal"}`}>
              Result: {isAnomaly ? "ANOMALY" : "NORMAL"}
            </p>
            <p className="single-result-score">
              Score: {result.score.toFixed(3)}
            </p>

            <button
              onClick={handleReset}
              className="btn-back"
              style={{ marginTop: 12 }}
            >
              Select Another Image
            </button>
          </div>
        )}
      </div>

      {/* Heatmap localization view if result exists */}
      {result && (
        <div style={{ marginTop: 28 }}>
          <div className="panel-card">
            <ScoreBar score={result.score} threshold={threshold} />
          </div>

          <HeatmapViewer
            original={preview}
            heatmap={result.heatmap}
            filename={selectedFile?.name || "inspected_sample.png"}
          />
        </div>
      )}
    </div>
  );
}
