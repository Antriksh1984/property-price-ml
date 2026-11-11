// Formatting + validation helpers shared across components.

/** Format a price given in lakhs: "₹ 75.55 L" below 100 L, "₹ 1.25 Cr" from 100 L up. */
export function formatPrice(lakhs) {
  if (lakhs == null || Number.isNaN(lakhs)) return "—";
  if (lakhs >= 100) return `₹ ${(lakhs / 100).toFixed(2)} Cr`;
  return `₹ ${lakhs.toFixed(2)} L`;
}

/** Indian digit grouping, e.g. 6296.18 -> "₹ 6,296". */
export function formatRupees(value) {
  return `₹ ${Math.round(value).toLocaleString("en-IN")}`;
}

// Bounds mirror backend/schemas.py (and ml/config.py).
export const LIMITS = {
  sqftMin: 300,
  sqftMax: 30000,
  maxExtraBaths: 2,
};

/** Client-side validation; returns { field: message } (empty when valid). */
export function validate({ location, totalSqft, bhk, bath }, locations) {
  const errors = {};
  if (!location) errors.location = "Pick a locality from the list.";
  else if (locations.length && !locations.includes(location)) errors.location = "Choose a locality from the list.";

  const sqft = Number(totalSqft);
  if (totalSqft === "" || Number.isNaN(sqft)) errors.total_sqft = "Enter the built-up area in sqft.";
  else if (sqft < LIMITS.sqftMin || sqft > LIMITS.sqftMax)
    errors.total_sqft = `Area must be between ${LIMITS.sqftMin.toLocaleString("en-IN")} and ${LIMITS.sqftMax.toLocaleString("en-IN")} sqft.`;

  if (bath > bhk + LIMITS.maxExtraBaths) errors.bath = `At most ${bhk + LIMITS.maxExtraBaths} bathrooms for a ${bhk} BHK.`;
  return errors;
}
