import { useState } from "react";
import InspectPage from "./pages/InspectPage.jsx";

export default function App() {
  const [stage, setStage] = useState("materials");

  function handleBack() {
    if (stage === "single" || stage === "multiple" || stage === "folder") {
      setStage("mode_select");
    } else if (stage === "mode_select") {
      setStage("materials");
    }
  }

  return (
    <div className="app-wrapper">
      {/* Top Header Bar */}
      <header className="header">
        <div className="header-container">
          <div className="brand-section">
            <h1 className="brand-title">Industrial Anomaly Detection</h1>
          </div>

          {/* Right Corner: Back Button */}
          {stage !== "materials" && (
            <button onClick={handleBack} className="btn-back">
              ← Back
            </button>
          )}
        </div>
      </header>

      {/* Main Content Area */}
      <main className="main-content">
        <InspectPage stage={stage} setStage={setStage} />
      </main>

      {/* Footer */}
      <footer className="footer">
        Industrial Anomaly Detection System • Real-Time Visual Inspection
      </footer>
    </div>
  );
}
