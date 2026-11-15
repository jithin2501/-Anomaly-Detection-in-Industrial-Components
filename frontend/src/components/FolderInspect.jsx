import { useState, useRef } from "react";
import { predict } from "../api/client.js";
import HeatmapViewer from "./HeatmapViewer.jsx";

function formatHeatmapSrc(src) {
  if (!src) return "";
  if (src.startsWith("data:")) return src;
  return `data:image/png;base64,${src}`;
}

export default function FolderInspect({ material, threshold }) {
  const filesInputRef = useRef(null);

  const [items, setItems] = useState([]);
  const [loading, setLoading] = useState(false);
  const [progress, setProgress] = useState({ current: 0, total: 0 });
  const [selectedItem, setSelectedItem] = useState(null);
  const [error, setError] = useState(null);

  async function processFiles(fileList) {
    if (!fileList || fileList.length === 0) return;
    const validFiles = Array.from(fileList).filter((f) =>
      ["image/png", "image/jpeg", "image/jpg"].includes(f.type)
    );

    if (validFiles.length === 0) {
      setError("No valid PNG or JPEG images found.");
      return;
    }

    setError(null);
    setItems([]);
    setSelectedItem(null);
    setLoading(true);
    setProgress({ current: 0, total: validFiles.length });

    const results = [];
    for (let i = 0; i < validFiles.length; i++) {
      const file = validFiles[i];
      setProgress({ current: i + 1, total: validFiles.length });
      try {
        const preview = URL.createObjectURL(file);
        const res = await predict(file, threshold, material?.id || "bottle");
        results.push({
          filename: file.name,
          preview,
          score: res.score,
          heatmap: formatHeatmapSrc(res.heatmap),
        });
      } catch (err) {
        console.error(`Error inspecting ${file.name}:`, err);
      }
    }

    setItems(results);
    setLoading(false);
  }

  function handleFilesChange(e) {
    const files = e.target.files;
    if (!files || files.length === 0) return;
    processFiles(files);
    e.target.value = "";
  }

  function handleDrop(e) {
    e.preventDefault();
    if (loading) return;
    processFiles(e.dataTransfer.files);
  }

  const processed = items.map((item) => {
    const isAnomaly = item.score > threshold;
    return {
      ...item,
      status: isAnomaly ? "ANOMALY" : "NORMAL",
      isAnomaly,
    };
  });

  const total = processed.length;
  const anomalyCount = processed.filter((i) => i.isAnomaly).length;
  const normalCount = total - anomalyCount;

  return (
    <div>
      {/* Top Header */}
      <div className="page-header-row">
        <h2 className="page-section-title">Multiple Image Analysis • {material.name}</h2>
      </div>

      {error && (
        <div className="error-box" style={{ maxWidth: 540, margin: "0 auto 20px" }}>
          <span>⚠️ {error}</span>
        </div>
      )}

      {/* Centered Upload Card - Identical UI structure to Single Image */}
      <div className="single-inspect-card" style={{ maxWidth: 540, margin: "0 auto 24px" }}>
        <div
          className="single-preview-frame"
          style={{ cursor: "pointer", height: 200 }}
          onClick={() => filesInputRef.current?.click()}
          onDragOver={(e) => e.preventDefault()}
          onDrop={handleDrop}
        >
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
                d="M19 11H5m14 0a2 2 0 012 2v6a2 2 0 01-2 2H5a2 2 0 01-2-2v-6a2 2 0 012-2m14 0V9a2 2 0 00-2-2M5 11V9a2 2 0 012-2m0 0V5a2 2 0 012-2h6a2 2 0 012 2v2M7 7h10"
              />
            </svg>
            <p style={{ fontWeight: 600, color: "#e5e5e5", fontSize: "0.95rem" }}>
              Click or drag multiple images here
            </p>
            <p style={{ fontSize: "0.78rem", color: "#9ca3af" }}>
              Supports batch inspection for multiple PNG, JPG, JPEG files
            </p>
          </div>
        </div>

        <input
          ref={filesInputRef}
          type="file"
          multiple
          accept="image/png,image/jpeg,image/jpg"
          style={{ display: "none" }}
          onChange={handleFilesChange}
        />

        {/* Single Centered Button: Select Files */}
        <button
          className="single-analyze-btn"
          disabled={loading}
          onClick={() => filesInputRef.current?.click()}
        >
          {loading ? "Analyzing Multiple Images..." : "Select Files"}
        </button>
      </div>

      {/* Progress Box */}
      {loading && (
        <div className="loading-box" style={{ maxWidth: 540, margin: "0 auto 24px" }}>
          <div className="loading-content">
            <div className="spinner" />
            <span>
              Analyzing images: {progress.current} of {progress.total}...
            </span>
          </div>
          <div className="progress-bar-track">
            <div
              className="progress-bar-fill"
              style={{
                width: `${(progress.current / progress.total) * 100}%`,
              }}
            />
          </div>
        </div>
      )}

      {/* Batch Summary Counters */}
      {total > 0 && (
        <div
          style={{
            display: "flex",
            alignItems: "center",
            justifyContent: "space-between",
            background: "#0d0d0f",
            border: "1px solid #222226",
            borderRadius: 12,
            padding: "14px 20px",
            marginBottom: 20,
          }}
        >
          <span style={{ fontSize: "0.95rem", fontWeight: 700, color: "#ffffff" }}>
            Batch Summary
          </span>
          <div style={{ display: "flex", gap: 20, fontSize: "0.88rem" }}>
            <span style={{ color: "#9ca3af" }}>
              Total: <strong style={{ color: "#ffffff" }}>{total}</strong>
            </span>
            <span style={{ color: "#10b981" }}>
              Normal: <strong>{normalCount}</strong>
            </span>
            <span style={{ color: "#ef4444" }}>
              Anomaly: <strong>{anomalyCount}</strong>
            </span>
          </div>
        </div>
      )}

      {/* Results Table */}
      {total > 0 && (
        <div className="batch-table-wrap">
          <table className="batch-table">
            <thead>
              <tr>
                <th>IMAGE</th>
                <th>STATUS</th>
                <th>SCORE</th>
              </tr>
            </thead>
            <tbody>
              {processed.map((item, idx) => (
                <tr
                  key={idx}
                  onClick={() => setSelectedItem(item)}
                  style={{ cursor: "pointer" }}
                >
                  <td style={{ fontWeight: 600 }}>{item.filename}</td>
                  <td className={`status-cell ${item.isAnomaly ? "anomaly" : "normal"}`}>
                    {item.status}
                  </td>
                  <td className="score-cell">{item.score.toFixed(3)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      {/* Inspected Samples - Stacked: One item per row, the next below that */}
      {total > 0 && (
        <div>
          <h3 style={{ fontSize: "0.95rem", color: "#9ca3af", marginBottom: 14, fontWeight: 700 }}>
            INSPECTED SAMPLES ({total})
          </h3>
          <div className="batch-gallery-grid">
            {processed.map((item, idx) => (
              <div
                key={idx}
                className={`batch-item-card ${item.isAnomaly ? "is-anomaly" : "is-normal"}`}
                onClick={() => setSelectedItem(item)}
              >
                {/* Row Card Header */}
                <div className="batch-card-header">
                  <span
                    className="batch-item-filename"
                    style={{ fontSize: "0.9rem", color: "#ffffff", fontWeight: 600 }}
                  >
                    {item.filename}
                  </span>
                  <div style={{ display: "flex", alignItems: "center", gap: 16 }}>
                    <span
                      className={`batch-item-status ${item.isAnomaly ? "anomaly" : "normal"}`}
                      style={{ fontSize: "0.88rem" }}
                    >
                      {item.status}
                    </span>
                    <span
                      className="batch-item-score"
                      style={{ fontSize: "0.85rem", color: "#e5e5e5" }}
                    >
                      Score: {item.score.toFixed(3)}
                    </span>
                  </div>
                </div>

                {/* Dual View in this row: Original Image + Defect Heatmap */}
                <div className="batch-item-thumbs-dual">
                  <div className="batch-thumb-col">
                    <span className="batch-thumb-tag">Original</span>
                    <img src={item.preview} alt={item.filename} />
                  </div>
                  <div className="batch-thumb-col">
                    <span className="batch-thumb-tag tag-heatmap">Heatmap</span>
                    <img src={formatHeatmapSrc(item.heatmap)} alt="Defect Heatmap" />
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Heatmap detail modal on card click */}
      {selectedItem && (
        <div
          style={{
            position: "fixed",
            inset: 0,
            backgroundColor: "rgba(0, 0, 0, 0.85)",
            backdropFilter: "blur(6px)",
            zIndex: 100,
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            padding: 20,
          }}
          onClick={() => setSelectedItem(null)}
        >
          <div
            style={{
              backgroundColor: "#0d0d0f",
              border: "1px solid #2e2e32",
              borderRadius: 18,
              padding: 24,
              maxWidth: 800,
              width: "100%",
              maxHeight: "90vh",
              overflowY: "auto",
            }}
            onClick={(e) => e.stopPropagation()}
          >
            <div
              style={{
                display: "flex",
                alignItems: "center",
                justifyContent: "space-between",
                marginBottom: 16,
              }}
            >
              <div>
                <h3 style={{ fontSize: "1.1rem", color: "#ffffff", fontWeight: 700 }}>
                  {selectedItem.filename}
                </h3>
                <span
                  style={{
                    color: selectedItem.isAnomaly ? "#ef4444" : "#10b981",
                    fontWeight: 700,
                    fontSize: "0.85rem",
                  }}
                >
                  {selectedItem.status} • Score: {selectedItem.score.toFixed(3)} (Threshold: {threshold.toFixed(2)})
                </span>
              </div>
              <button onClick={() => setSelectedItem(null)} className="btn-back">
                ✕ Close
              </button>
            </div>

            <HeatmapViewer
              original={selectedItem.preview}
              heatmap={selectedItem.heatmap}
              filename={selectedItem.filename}
            />
          </div>
        </div>
      )}
    </div>
  );
}
