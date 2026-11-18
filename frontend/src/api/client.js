const API = import.meta.env.VITE_API_URL || "http://localhost:8000";

export async function predict(file, threshold = null, category = "bottle") {
  const body = new FormData();
  body.append("file", file);
  const params = new URLSearchParams();
  if (threshold !== null && threshold !== undefined) {
    params.set("threshold", threshold);
  }
  if (category) {
    params.set("category", category);
  }
  const queryString = params.toString();
  const url = queryString ? `${API}/predict?${queryString}` : `${API}/predict`;

  const res = await fetch(url, { method: "POST", body });
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.detail || `Request failed (${res.status})`);
  }
  return res.json();
}

export async function getHistory() {
  const res = await fetch(`${API}/history`);
  if (!res.ok) throw new Error(`Request failed (${res.status})`);
  return res.json();
}

export async function getModels() {
  const res = await fetch(`${API}/models`);
  if (!res.ok) throw new Error(`Failed to fetch models (${res.status})`);
  return res.json();
}

export async function addModel(productName, zipFile) {
  const body = new FormData();
  body.append("name", productName);
  body.append("file", zipFile);
  const res = await fetch(`${API}/models/add`, {
    method: "POST",
    body,
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.detail || `Failed to add model (${res.status})`);
  }
  return res.json();
}

export async function deleteModel(modelId) {
  const res = await fetch(`${API}/models/${modelId}`, {
    method: "DELETE",
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.detail || `Failed to delete model (${res.status})`);
  }
  return res.json();
}
