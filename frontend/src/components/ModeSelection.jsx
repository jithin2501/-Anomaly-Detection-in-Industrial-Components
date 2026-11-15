export default function ModeSelection({
  material,
  threshold,
  onThresholdChange,
  onChooseMode,
}) {
  return (
    <div>
      <div className="page-header-row">
        <h2 className="page-section-title">Detection Mode Selection</h2>
      </div>

      <div className="mode-selection-layout">
        {/* Left Side: Material & Threshold */}
        <div className="mode-left-card">
          <div className="mode-material-preview">
            <img
              src={material.image}
              alt={material.name}
              onError={(e) => {
                e.target.src = "/placeholder.svg";
              }}
            />
          </div>

          <h3 className="mode-material-title">{material.name}</h3>

          <div className="mode-threshold-wrap">
            <div className="mode-threshold-label-row">
              <span className="mode-threshold-label">
                Threshold: <strong className="mode-threshold-val">{threshold.toFixed(2)}</strong>
              </span>
            </div>
            <input
              type="range"
              min="0.05"
              max="0.95"
              step="0.01"
              value={threshold}
              onChange={(e) => onThresholdChange(parseFloat(e.target.value))}
              className="mode-range-slider"
            />
            <p className="mode-threshold-hint">
              (Recommended: {material.recommendedThreshold ? material.recommendedThreshold.toFixed(2) : "0.25"})
            </p>
          </div>
        </div>

        {/* Right Side: Two Buttons (Single Image & Multiple Image Detection) */}
        <div className="mode-right-buttons">
          <button
            className="mode-action-btn"
            onClick={() => onChooseMode("single")}
          >
            <svg width="24" height="24" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z" />
            </svg>
            <span>Single Image</span>
          </button>

          <button
            className="mode-action-btn"
            onClick={() => onChooseMode("multiple")}
          >
            <svg width="24" height="24" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M19 11H5m14 0a2 2 0 012 2v6a2 2 0 01-2 2H5a2 2 0 01-2-2v-6a2 2 0 012-2m14 0V9a2 2 0 00-2-2M5 11V9a2 2 0 012-2m0 0V5a2 2 0 012-2h6a2 2 0 012 2v2M7 7h10" />
            </svg>
            <span>Multiple Image Detection</span>
          </button>
        </div>
      </div>
    </div>
  );
}
