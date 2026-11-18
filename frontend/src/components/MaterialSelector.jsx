import { useState, useEffect, useRef } from "react";
import { getModels, addModel, deleteModel } from "../api/client.js";

export const MATERIALS = [];

export default function MaterialSelector({ onSelectMaterial }) {
  const [materials, setMaterials] = useState([]);
  const [showModal, setShowModal] = useState(false);
  const [productName, setProductName] = useState("");
  const [zipFile, setZipFile] = useState(null);
  const [isUploading, setIsUploading] = useState(false);
  const [errorMessage, setErrorMessage] = useState("");
  const [successMessage, setSuccessMessage] = useState("");
  const fileInputRef = useRef(null);

  useEffect(() => {
    fetchModelsList();
  }, []);

  async function fetchModelsList() {
    try {
      const data = await getModels();
      if (Array.isArray(data)) {
        setMaterials(data);
      }
    } catch {
      setMaterials([]);
    }
  }

  async function handleAddSubmit(e) {
    e.preventDefault();
    setErrorMessage("");
    setSuccessMessage("");

    if (!productName.trim()) {
      setErrorMessage("Please enter a product name.");
      return;
    }
    if (!zipFile) {
      setErrorMessage("Please select a .zip file containing normal images.");
      return;
    }

    try {
      setIsUploading(true);
      const res = await addModel(productName.trim(), zipFile);
      setIsUploading(false);
      setSuccessMessage(res.message || "Model added successfully!");

      // Refresh list and select the new material after 1 second
      await fetchModelsList();
      setTimeout(() => {
        setShowModal(false);
        setProductName("");
        setZipFile(null);
        if (res.model) {
          onSelectMaterial(res.model);
        }
      }, 1200);
    } catch (err) {
      setIsUploading(false);
      setErrorMessage(err.message || "Failed to compile model from ZIP file.");
    }
  }

  async function handleDeleteModel(e, mat) {
    e.stopPropagation();
    const confirmed = window.confirm(
      `Are you sure you want to delete "${mat.name}"?\nThis will remove its memory bank, trained features, and dataset from disk.`
    );
    if (!confirmed) return;

    try {
      await deleteModel(mat.id);
      await fetchModelsList();
    } catch (err) {
      alert(`Failed to delete model: ${err.message}`);
    }
  }

  return (
    <div>
      <div className="page-header-row" style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
        <h2 className="page-section-title">Select Material For Anomaly Detection</h2>
        <button
          className="btn-add-model"
          onClick={() => {
            setErrorMessage("");
            setSuccessMessage("");
            setShowModal(true);
          }}
        >
          <span style={{ fontSize: "1.1rem", marginRight: "6px" }}>+</span> Add Model / Product
        </button>
      </div>

      <div className="materials-grid-duo">
        {materials.map((mat) => (
          <div
            key={mat.id}
            className="material-card"
            onClick={() => onSelectMaterial(mat)}
          >
            <div className="material-thumb-wrap">
              <button
                type="button"
                className="btn-delete-model"
                title={`Delete ${mat.name} model`}
                onClick={(e) => handleDeleteModel(e, mat)}
              >
                <svg
                  width="14"
                  height="14"
                  viewBox="0 0 24 24"
                  fill="none"
                  stroke="currentColor"
                  strokeWidth="2.2"
                  strokeLinecap="round"
                  strokeLinejoin="round"
                >
                  <polyline points="3 6 5 6 21 6" />
                  <path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2" />
                  <line x1="10" y1="11" x2="10" y2="17" />
                  <line x1="14" y1="11" x2="14" y2="17" />
                </svg>
              </button>

              <img
                src={mat.image}
                alt={mat.name}
                className="material-thumb-img"
                onError={(e) => {
                  e.target.src = "/placeholder.svg";
                }}
              />
            </div>
            <div className="material-card-body">
              <h3 className="material-card-title">{mat.name}</h3>
              {mat.numImages && (
                <div style={{ fontSize: "0.78rem", color: "#9ca3af", marginTop: "4px" }}>
                  Memory Bank: <strong style={{ color: "#60a5fa" }}>{mat.numImages} Normal Images</strong>
                </div>
              )}
            </div>
          </div>
        ))}

        {/* Add Model Interactive Card */}
        <div
          className="material-card material-card-add"
          onClick={() => {
            setErrorMessage("");
            setSuccessMessage("");
            setShowModal(true);
          }}
          title="Add a new custom material or MVTec category"
        >
          <div className="material-thumb-wrap add-thumb-wrap">
            <div className="add-card-visual">
              <div className="add-card-glow"></div>
              <svg
                width="50"
                height="50"
                viewBox="0 0 24 24"
                fill="none"
                stroke="currentColor"
                strokeWidth="1.6"
                strokeLinecap="round"
                strokeLinejoin="round"
                className="add-card-svg"
              >
                <circle cx="12" cy="12" r="10" />
                <line x1="12" y1="8" x2="12" y2="16" />
                <line x1="8" y1="12" x2="16" y2="12" />
              </svg>
            </div>
          </div>
          <div className="material-card-body">
            <h3 className="material-card-title add-card-title">Add New Product</h3>
            <div className="add-card-sub">Custom Dataset / MVTec Model</div>
          </div>
        </div>
      </div>

      {/* Add Model Modal */}
      {showModal && (
        <div className="modal-overlay" onClick={() => !isUploading && setShowModal(false)}>
          <div className="modal-dialog" onClick={(e) => e.stopPropagation()}>
            <div className="modal-header">
              <h3 className="modal-title">Add New Product / Model</h3>
              <button
                className="modal-close-btn"
                disabled={isUploading}
                onClick={() => setShowModal(false)}
              >
                &times;
              </button>
            </div>

            <form onSubmit={handleAddSubmit}>
              <div className="modal-body">
                {errorMessage && (
                  <div className="modal-alert modal-alert-error">
                    {errorMessage}
                  </div>
                )}

                {successMessage && (
                  <div className="modal-alert modal-alert-success">
                    {successMessage}
                  </div>
                )}

                <div className="modal-field">
                  <label className="modal-label">Product / Category Name</label>
                  <input
                    type="text"
                    className="modal-input"
                    placeholder="e.g. Screw, Hazelnut, Capsule, Leather"
                    value={productName}
                    disabled={isUploading}
                    onChange={(e) => setProductName(e.target.value)}
                    required
                  />
                </div>

                <div className="modal-field">
                  <label className="modal-label">Normal Images Archive (.ZIP, .TAR, .TAR.XZ)</label>
                  <div
                    className="modal-file-dropzone"
                    onClick={() => !isUploading && fileInputRef.current?.click()}
                  >
                    <input
                      ref={fileInputRef}
                      type="file"
                      accept=".zip,.tar,.tar.gz,.tgz,.tar.xz,.tar.bz2"
                      style={{ display: "none" }}
                      onChange={(e) => {
                        if (e.target.files && e.target.files[0]) {
                          setZipFile(e.target.files[0]);
                          if (!productName) {
                            const inferred = e.target.files[0].name
                              .replace(/\.(zip|tar\.xz|tar\.gz|tar|tgz|tar\.bz2)$/i, "")
                              .replace(/[^a-zA-Z0-9_\- ]/g, "");
                            setProductName(inferred);
                          }
                        }
                      }}
                    />
                    <svg
                      width="38"
                      height="38"
                      viewBox="0 0 24 24"
                      fill="none"
                      stroke="#60a5fa"
                      strokeWidth="1.8"
                      strokeLinecap="round"
                      strokeLinejoin="round"
                      style={{ marginBottom: "8px", display: "inline-block" }}
                    >
                      <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z" />
                      <polyline points="14 2 14 8 20 8" />
                      <line x1="16" y1="13" x2="8" y2="13" />
                      <line x1="16" y1="17" x2="8" y2="17" />
                    </svg>
                    {zipFile ? (
                      <div>
                        <strong style={{ color: "#60a5fa" }}>{zipFile.name}</strong>
                        <div style={{ fontSize: "0.78rem", color: "#9ca3af", marginTop: "4px" }}>
                          {(zipFile.size / 1024 / 1024).toFixed(2)} MB &bull; Click to change
                        </div>
                      </div>
                    ) : (
                      <div>
                        <span style={{ color: "#60a5fa", fontWeight: "600" }}>Click to select ZIP or TAR archive</span>
                        <div style={{ fontSize: "0.78rem", color: "#6b7280", marginTop: "4px" }}>
                          Supports .zip, .tar, .tar.xz (MVTec official archives)
                        </div>
                      </div>
                    )}
                  </div>
                </div>
              </div>

              <div className="modal-footer">
                <button
                  type="button"
                  className="modal-btn-cancel"
                  disabled={isUploading}
                  onClick={() => setShowModal(false)}
                >
                  Cancel
                </button>
                <button
                  type="submit"
                  className="modal-btn-submit"
                  disabled={isUploading}
                >
                  {isUploading ? (
                    <span>Compiling Memory Bank...</span>
                  ) : (
                    <span>Add Model</span>
                  )}
                </button>
              </div>
            </form>
          </div>
        </div>
      )}
    </div>
  );
}
