export default function HeatmapViewer({ original, heatmap, filename }) {
  const heatmapSrc = !heatmap
    ? ""
    : heatmap.startsWith("data:")
    ? heatmap
    : `data:image/png;base64,${heatmap}`;

  return (
    <div className="heatmap-card">
      {filename && (
        <div className="heatmap-card-header">
          <span className="file-title">{filename}</span>
          <span style={{ fontSize: "0.72rem", color: "var(--text-dim)", textTransform: "uppercase" }}>
            Visual Inspection
          </span>
        </div>
      )}

      <div className="heatmap-grid">
        <figure className="image-panel">
          <div className="image-frame">
            <img src={original} alt="Original Product" />
          </div>
          <figcaption className="image-caption original">
            <span className="caption-dot original" />
            Original Image
          </figcaption>
        </figure>

        <figure className="image-panel">
          <div className="image-frame">
            <img src={heatmapSrc} alt="Anomaly Heatmap" />
          </div>
          <figcaption className="image-caption heatmap">
            <span className="caption-dot heatmap" />
            Anomaly Heatmap (Red/Yellow = Defect)
          </figcaption>
        </figure>
      </div>
    </div>
  );
}
