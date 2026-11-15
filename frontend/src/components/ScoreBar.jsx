export default function ScoreBar({ score, threshold }) {
  const max = Math.max(score, threshold, 0.6) * 1.2 || 1;
  const pct = (v) => `${Math.min(100, Math.max(0, (v / max) * 100))}%`;
  const bad = score > threshold;

  return (
    <div className="scorebar-container">
      <div className="scorebar-header">
        <div>
          <span>Anomaly Score:</span>
          <span className={`score-value ${bad ? "defective" : "normal"}`}>
            {score.toFixed(3)}
          </span>
        </div>
        <div>
          <span>Threshold: </span>
          <span className="threshold-val">{threshold.toFixed(2)}</span>
        </div>
      </div>

      <div className="score-track">
        {/* Fill bar */}
        <div
          className={`score-fill ${bad ? "defective" : "normal"}`}
          style={{ width: pct(score) }}
        />
        {/* Threshold marker pin */}
        <div
          className="threshold-pin"
          style={{ left: pct(threshold) }}
          title={`Threshold: ${threshold.toFixed(2)}`}
        />
      </div>

      <div className="scorebar-footer">
        <span>0.00 (Clean)</span>
        <span className="threshold-indicator-text">▲ Threshold ({threshold.toFixed(2)})</span>
        <span>1.00 (Severe)</span>
      </div>
    </div>
  );
}
