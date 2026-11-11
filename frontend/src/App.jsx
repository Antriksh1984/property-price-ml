import { useEffect, useState } from "react";
import { getLocations, getMetrics, predict } from "./api.js";
import { LIMITS, validate } from "./format.js";
import LocationSelect from "./components/LocationSelect.jsx";
import OptionGroup from "./components/OptionGroup.jsx";
import ResultCard from "./components/ResultCard.jsx";
import AboutModel from "./components/AboutModel.jsx";

const BHK_OPTIONS = [1, 2, 3, 4, 5, 6];
const BATH_OPTIONS = [1, 2, 3, 4, 5, 6, 7, 8];

export default function App() {
  const [locations, setLocations] = useState([]);
  const [locationsError, setLocationsError] = useState("");
  const [metrics, setMetrics] = useState(null);
  const [metricsError, setMetricsError] = useState("");

  const [form, setForm] = useState({ location: "", totalSqft: "1200", bhk: 2, bath: 2 });
  const [errors, setErrors] = useState({});
  const [apiError, setApiError] = useState("");
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [lastRequest, setLastRequest] = useState(null);

  // Load reference data once.
  useEffect(() => {
    getLocations()
      .then(setLocations)
      .catch((e) => setLocationsError(e.message));
    getMetrics()
      .then(setMetrics)
      .catch((e) => setMetricsError(e.message));
  }, []);

  const update = (field, value) => {
    setForm((f) => {
      const next = { ...f, [field]: value };
      // Keep bathrooms within the allowed bhk + 2.
      if (field === "bhk" && next.bath > value + LIMITS.maxExtraBaths) next.bath = value + LIMITS.maxExtraBaths;
      return next;
    });
    setErrors((e) => ({ ...e, [field === "totalSqft" ? "total_sqft" : field]: undefined }));
  };

  const onSubmit = async (e) => {
    e.preventDefault();
    setApiError("");
    const found = validate(form, locations);
    setErrors(found);
    if (Object.keys(found).length) return;

    const payload = {
      location: form.location,
      total_sqft: Number(form.totalSqft),
      bhk: form.bhk,
      bath: form.bath,
    };
    setLoading(true);
    try {
      const res = await predict(payload);
      setResult(res);
      setLastRequest(payload);
    } catch (err) {
      setApiError(err.message);
      if (err.fieldErrors) setErrors(err.fieldErrors);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="page">
      <header className="site-header">
        <div className="brand">
          <span className="logo" aria-hidden="true">
            <svg viewBox="0 0 32 32">
              <path d="M8 16l8-7 8 7v8h-5v-5h-6v5H8z" />
            </svg>
          </span>
          <div>
            <h1>Bengaluru Price Estimator</h1>
            <p>Machine-learning price estimates for apartments and houses across 240+ localities.</p>
          </div>
        </div>
        <a className="header-link" href="#about">
          About the model
        </a>
      </header>

      <main>
        <div className="workspace">
          <form className="card form-card" onSubmit={onSubmit} noValidate>
            <h2>Property details</h2>

            <div className="field">
              <label htmlFor="location">Locality</label>
              <LocationSelect
                locations={locations}
                value={form.location}
                onChange={(v) => update("location", v)}
                error={errors.location}
                disabled={!locations.length}
              />
              {locationsError && <p className="field-error">Couldn't load localities: {locationsError}</p>}
              {errors.location && <p className="field-error">{errors.location}</p>}
            </div>

            <div className="field">
              <label htmlFor="sqft">Built-up area</label>
              <div className={`input-suffix ${errors.total_sqft ? "has-error" : ""}`}>
                <input
                  id="sqft"
                  type="number"
                  inputMode="numeric"
                  min={LIMITS.sqftMin}
                  max={LIMITS.sqftMax}
                  step="10"
                  value={form.totalSqft}
                  aria-invalid={Boolean(errors.total_sqft)}
                  onChange={(e) => update("totalSqft", e.target.value)}
                />
                <span>sqft</span>
              </div>
              {errors.total_sqft ? (
                <p className="field-error">{errors.total_sqft}</p>
              ) : (
                <p className="hint">Between 300 and 30,000 sqft.</p>
              )}
            </div>

            <div className="field">
              <span className="label">Bedrooms (BHK)</span>
              <OptionGroup label="Bedrooms" name="bhk" options={BHK_OPTIONS} value={form.bhk} onChange={(v) => update("bhk", v)} />
            </div>

            <div className="field">
              <span className="label">Bathrooms</span>
              <OptionGroup
                label="Bathrooms"
                name="bath"
                options={BATH_OPTIONS}
                value={form.bath}
                onChange={(v) => update("bath", v)}
                isDisabled={(n) => n > form.bhk + LIMITS.maxExtraBaths}
              />
              {errors.bath ? (
                <p className="field-error">{errors.bath}</p>
              ) : (
                <p className="hint">Up to BHK + 2.</p>
              )}
            </div>

            {apiError && (
              <p className="notice notice-error" role="alert">
                {apiError}
              </p>
            )}

            <button className="btn-primary" type="submit" disabled={loading}>
              {loading ? (
                <>
                  <span className="spinner" aria-hidden="true" /> Estimating…
                </>
              ) : (
                "Predict price"
              )}
            </button>
          </form>

          <ResultCard result={result} loading={loading} request={lastRequest} />
        </div>

        <AboutModel metrics={metrics} error={metricsError} />
      </main>

      <footer className="site-footer">
        Data: Kaggle “Bengaluru House Price Data”. Estimates are for illustration only.
      </footer>
    </div>
  );
}
