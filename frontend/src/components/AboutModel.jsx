import { useState } from "react";
import { staticUrl } from "../api.js";

/** "About the model": headline test metrics, CV comparison table and chart. */
export default function AboutModel({ metrics, error }) {
  const [imgFailed, setImgFailed] = useState(false);

  if (error) {
    return (
      <section className="card about" id="about">
        <h2>About the model</h2>
        <p className="notice notice-error">Couldn't load model details: {error}</p>
      </section>
    );
  }
  if (!metrics) {
    return (
      <section className="card about" id="about" aria-busy="true">
        <h2>About the model</h2>
        <div className="skeleton skeleton-line" />
        <div className="skeleton skeleton-line short" />
      </section>
    );
  }

  const t = metrics.test_metrics;
  const ds = metrics.dataset;
  const rows = [...metrics.cv_results].sort((a, b) => a.rmse - b.rmse);
  const trainedOn = new Date(metrics.trained_at).toLocaleDateString("en-IN", { dateStyle: "medium" });

  return (
    <section className="card about" id="about">
      <div className="about-head">
        <div>
          <p className="eyebrow">About the model</p>
          <h2>{metrics.chosen_model}</h2>
          <p className="muted">
            Trained {trainedOn} on {ds.train_rows.toLocaleString("en-IN")} cleaned listings from the Kaggle Bengaluru
            House Price dataset, evaluated once on {ds.test_rows.toLocaleString("en-IN")} held-out listings.
          </p>
        </div>
      </div>

      <div className="stats">
        <div className="stat">
          <span className="stat-value">{t.r2.toFixed(3)}</span>
          <span className="stat-label">Test R²</span>
        </div>
        <div className="stat">
          <span className="stat-value">₹ {t.mae.toFixed(1)} L</span>
          <span className="stat-label">Test mean abs. error</span>
        </div>
        <div className="stat">
          <span className="stat-value">{t.median_abs_pct_error.toFixed(1)}%</span>
          <span className="stat-label">Median abs. % error</span>
        </div>
        <div className="stat">
          <span className="stat-value">{Math.round(t.interval_coverage * 100)}%</span>
          <span className="stat-label">Test prices inside range</span>
        </div>
      </div>

      <div className="about-grid">
        <div className="table-wrap">
          <p className="table-caption" id="cv-caption">
            5-fold cross-validation on the training split (lower RMSE is better)
          </p>
          <table aria-describedby="cv-caption">
            <thead>
              <tr>
                <th scope="col">Model</th>
                <th scope="col" className="num">R²</th>
                <th scope="col" className="num">MAE (L)</th>
                <th scope="col" className="num">RMSE (L)</th>
              </tr>
            </thead>
            <tbody>
              {rows.map((r) => (
                <tr key={r.model} className={r.model === metrics.chosen_model ? "is-chosen" : ""}>
                  <th scope="row">
                    {r.model}
                    {r.model === metrics.chosen_model && <span className="badge">chosen</span>}
                  </th>
                  <td className="num">{r.r2.toFixed(3)}</td>
                  <td className="num">{r.mae.toFixed(2)}</td>
                  <td className="num">{r.rmse.toFixed(2)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <figure className="chart">
          {imgFailed ? (
            <p className="muted">Comparison chart unavailable.</p>
          ) : (
            <img
              src={staticUrl("model_comparison.png")}
              alt="Bar chart of 5-fold CV RMSE per model; the chosen model has the lowest error."
              loading="lazy"
              onError={() => setImgFailed(true)}
            />
          )}
        </figure>
      </div>
    </section>
  );
}
