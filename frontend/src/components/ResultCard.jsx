import { formatPrice, formatRupees } from "../format.js";

/** Shows the latest prediction, an empty state, or a loading shimmer. */
export default function ResultCard({ result, loading, request }) {
  if (loading && !result) {
    return (
      <section className="card result-card" aria-busy="true" aria-live="polite">
        <p className="eyebrow">Estimated price</p>
        <div className="skeleton skeleton-price" />
        <div className="skeleton skeleton-line" />
        <div className="skeleton skeleton-line short" />
      </section>
    );
  }

  if (!result) {
    return (
      <section className="card result-card is-empty" aria-live="polite">
        <div className="empty-illustration" aria-hidden="true">
          <svg viewBox="0 0 64 64">
            <path d="M10 30L32 12l22 18v22H40V38H24v14H10z" />
          </svg>
        </div>
        <h2>Your estimate will appear here</h2>
        <p className="muted">Pick a locality, enter the area and room counts, then press Predict.</p>
      </section>
    );
  }

  const { predicted_price_lakhs: price, low_lakhs: low, high_lakhs: high, price_per_sqft: pps } = result;
  // Position of the point estimate within the low–high band (for the range bar marker).
  const markerPct = high > low ? ((price - low) / (high - low)) * 100 : 50;

  return (
    <section className={`card result-card ${loading ? "is-refreshing" : ""}`} aria-live="polite">
      <p className="eyebrow">Estimated price</p>
      <p className="price">{formatPrice(price)}</p>
      <p className="muted summary">
        {request.bhk} BHK · {request.bath} bath · {Number(request.total_sqft).toLocaleString("en-IN")} sqft in{" "}
        <strong>{result.location_used === "other" ? "Other locality" : result.location_used}</strong>
      </p>

      <div className="range">
        <div className="range-labels">
          <span>
            <small>Low</small>
            {formatPrice(low)}
          </span>
          <span className="right">
            <small>High</small>
            {formatPrice(high)}
          </span>
        </div>
        <div className="range-bar" role="img" aria-label={`Likely range ${formatPrice(low)} to ${formatPrice(high)}`}>
          <span className="range-marker" style={{ left: `${markerPct}%` }} />
        </div>
        <p className="range-caption muted">About 8 in 10 comparable listings fall within this range.</p>
      </div>

      <dl className="facts">
        <div>
          <dt>Price per sqft</dt>
          <dd>{formatRupees(pps)}</dd>
        </div>
        <div>
          <dt>Model</dt>
          <dd>{result.model_name}</dd>
        </div>
      </dl>

      {result.warning && (
        <p className="notice notice-warn" role="status">
          <span aria-hidden="true">⚠</span> {result.warning}
        </p>
      )}
      <p className="disclaimer">
        This is a statistical estimate from historical listing prices, not a valuation. Actual prices depend on
        condition, floor, amenities and negotiation.
      </p>
    </section>
  );
}
