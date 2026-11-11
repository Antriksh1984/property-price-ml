// Thin wrapper around the FastAPI backend.
export const API_URL = (import.meta.env.VITE_API_URL || "http://localhost:8000").replace(/\/$/, "");

/** Error carrying a user-friendly message plus optional per-field messages. */
export class ApiError extends Error {
  constructor(message, fieldErrors = {}) {
    super(message);
    this.fieldErrors = fieldErrors;
  }
}

/** Turn FastAPI's 422 `detail` array into readable messages keyed by field. */
function parseValidationDetail(detail) {
  const fieldErrors = {};
  const messages = [];
  for (const item of Array.isArray(detail) ? detail : []) {
    const field = item.loc?.[item.loc.length - 1];
    // Model-level validator errors come back as "Value error, ..."
    const msg = String(item.msg || "Invalid value").replace(/^Value error,\s*/i, "");
    if (field && field !== "body") fieldErrors[field] = msg;
    messages.push(field && field !== "body" ? `${field}: ${msg}` : msg);
  }
  return { fieldErrors, message: messages.join(" · ") || "The request was invalid." };
}

async function request(path, options = {}) {
  let res;
  try {
    res = await fetch(`${API_URL}${path}`, {
      headers: { "Content-Type": "application/json" },
      ...options,
    });
  } catch {
    throw new ApiError(`Can't reach the API at ${API_URL}. Is the backend running?`);
  }
  const body = await res.json().catch(() => null);
  if (!res.ok) {
    if (res.status === 422 && body?.detail) {
      const { fieldErrors, message } = parseValidationDetail(body.detail);
      throw new ApiError(message, fieldErrors);
    }
    throw new ApiError(body?.detail || `Request failed (HTTP ${res.status}).`);
  }
  return body;
}

export const getLocations = () => request("/locations");
export const getMetrics = () => request("/metrics");
export const predict = (payload) => request("/predict", { method: "POST", body: JSON.stringify(payload) });
export const staticUrl = (file) => `${API_URL}/static/${file}`;
