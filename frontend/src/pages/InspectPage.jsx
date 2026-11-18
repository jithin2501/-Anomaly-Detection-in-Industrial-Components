import { useState } from "react";
import MaterialSelector from "../components/MaterialSelector.jsx";
import ModeSelection from "../components/ModeSelection.jsx";
import SingleInspect from "../components/SingleInspect.jsx";
import FolderInspect from "../components/FolderInspect.jsx";

export default function InspectPage({ stage, setStage }) {
  const [selectedMaterial, setSelectedMaterial] = useState(null);
  const [threshold, setThreshold] = useState(0.25);

  function handleSelectMaterial(mat) {
    setSelectedMaterial(mat);
    setThreshold(mat.recommendedThreshold || 0.25);
    setStage("mode_select");
  }

  return (
    <div>
      {/* 1. Initial Material Selection */}
      {stage === "materials" && (
        <MaterialSelector onSelectMaterial={handleSelectMaterial} />
      )}

      {/* 2. Detection Mode Selection (Material & Threshold on Left, Buttons on Right) */}
      {stage === "mode_select" && selectedMaterial && (
        <ModeSelection
          material={selectedMaterial}
          threshold={threshold}
          onThresholdChange={setThreshold}
          onChooseMode={(mode) => setStage(mode)}
        />
      )}

      {/* 3A. Single Image Analysis */}
      {stage === "single" && selectedMaterial && (
        <SingleInspect
          material={selectedMaterial}
          threshold={threshold}
        />
      )}

      {/* 3B. Multiple Image Analysis */}
      {(stage === "multiple" || stage === "folder") && selectedMaterial && (
        <FolderInspect
          material={selectedMaterial}
          threshold={threshold}
        />
      )}
    </div>
  );
}
