"""Test-set metrics and diagnostic plots.

Used by ``ml.train`` after model selection, and runnable on its own to
re-evaluate a saved model::

    python -m ml.evaluate
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")  # headless backend: works without a display
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score  # noqa: E402

from ml import config  # noqa: E402

# ---------------------------------------------------------------------------
# Plot styling (validated reference palette; recessive axes, thin marks)
# ---------------------------------------------------------------------------
SURFACE = "#fcfcfb"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID = "#e4e3df"
SERIES_BLUE = "#2a78d6"
SERIES_ORANGE = "#eb6834"

plt.rcParams.update(
    {
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "axes.edgecolor": GRID,
        "axes.labelcolor": TEXT_SECONDARY,
        "axes.titlecolor": TEXT_PRIMARY,
        "axes.titlesize": 13,
        "axes.titleweight": "bold",
        "axes.titlelocation": "left",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.8,
        "xtick.color": TEXT_SECONDARY,
        "ytick.color": TEXT_SECONDARY,
        "font.size": 10,
        "savefig.dpi": 130,
        "savefig.bbox": "tight",
    }
)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """R², MAE and RMSE (MAE/RMSE in lakhs), rounded for reporting."""
    return {
        "r2": round(float(r2_score(y_true, y_pred)), 4),
        "mae": round(float(mean_absolute_error(y_true, y_pred)), 3),
        "rmse": round(float(np.sqrt(mean_squared_error(y_true, y_pred))), 3),
    }


def interval_coverage(y_true: np.ndarray, y_pred: np.ndarray, low_ratio: float, high_ratio: float) -> float:
    """Share of actual prices that fall inside [pred*low_ratio, pred*high_ratio]."""
    inside = (y_true >= y_pred * low_ratio) & (y_true <= y_pred * high_ratio)
    return round(float(np.mean(inside)), 4)


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------
def plot_model_comparison(cv_results: list[dict[str, Any]], chosen: str, path: Path) -> None:
    """Horizontal bar chart of CV RMSE per model; the chosen model is highlighted."""
    rows = sorted(cv_results, key=lambda r: r["rmse"], reverse=True)
    names = [r["model"] for r in rows]
    rmse = [r["rmse"] for r in rows]
    colors = [SERIES_ORANGE if n == chosen else SERIES_BLUE for n in names]

    fig, ax = plt.subplots(figsize=(8, 0.55 * len(rows) + 1.4))
    bars = ax.barh(names, rmse, color=colors, height=0.55)
    ax.grid(axis="y", visible=False)
    for bar, value in zip(bars, rmse):
        ax.text(bar.get_width() + max(rmse) * 0.01, bar.get_y() + bar.get_height() / 2,
                f"{value:.1f}", va="center", color=TEXT_PRIMARY, fontsize=9)
    ax.set_xlabel("5-fold CV RMSE (lakhs ₹, lower is better)")
    ax.set_title(f"Model comparison — chosen: {chosen}")
    ax.set_xlim(0, max(rmse) * 1.12)
    fig.savefig(path)
    plt.close(fig)


def plot_pred_vs_actual(y_true: np.ndarray, y_pred: np.ndarray, path: Path) -> None:
    """Log-log scatter of predicted vs actual with the identity line."""
    fig, ax = plt.subplots(figsize=(6.5, 6))
    ax.scatter(y_true, y_pred, s=10, alpha=0.45, color=SERIES_BLUE, edgecolors="none")
    lo = min(y_true.min(), y_pred.min()) * 0.9
    hi = max(y_true.max(), y_pred.max()) * 1.1
    ax.plot([lo, hi], [lo, hi], color=TEXT_SECONDARY, linewidth=1.2, linestyle="--", label="perfect prediction")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel("Actual price (lakhs ₹, log scale)")
    ax.set_ylabel("Predicted price (lakhs ₹, log scale)")
    ax.set_title("Predicted vs actual (test set)")
    ax.legend(frameon=False, labelcolor=TEXT_SECONDARY)
    fig.savefig(path)
    plt.close(fig)


def plot_residuals(y_true: np.ndarray, y_pred: np.ndarray, path: Path) -> None:
    """Histogram of percentage errors ((pred - actual) / actual)."""
    pct_err = (y_pred - y_true) / y_true * 100
    clipped = np.clip(pct_err, -100, 100)
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.hist(clipped, bins=50, color=SERIES_BLUE, edgecolor=SURFACE, linewidth=1)
    ax.axvline(0, color=TEXT_SECONDARY, linewidth=1.2, linestyle="--")
    ax.grid(axis="x", visible=False)
    ax.set_xlabel("Prediction error (% of actual price, clipped to ±100%)")
    ax.set_ylabel("Listings")
    ax.set_title(f"Residuals (test set) — median abs. error {np.median(np.abs(pct_err)):.1f}%")
    fig.savefig(path)
    plt.close(fig)


def feature_importances(model: Any, top_n: int = 20) -> pd.Series:
    """Return the top-N feature importances (trees) or |coefficients| (linear models)."""
    pipe = model.regressor_  # fitted Pipeline inside TransformedTargetRegressor
    names = pipe.named_steps["prep"].get_feature_names_out()
    est = pipe.named_steps["model"]
    if hasattr(est, "feature_importances_"):
        values = np.asarray(est.feature_importances_)
    elif hasattr(est, "coef_"):
        values = np.abs(np.ravel(est.coef_))
    else:
        return pd.Series(dtype=float)
    clean_names = [n.split("__", 1)[-1].replace("location_", "loc: ") for n in names]
    return pd.Series(values, index=clean_names).sort_values(ascending=False).head(top_n)


def plot_feature_importance(model: Any, path: Path, top_n: int = 20) -> None:
    """Horizontal bar chart of the top-N most important features."""
    imp = feature_importances(model, top_n)
    if imp.empty:
        return
    imp = imp.iloc[::-1]  # largest at the top
    fig, ax = plt.subplots(figsize=(8, 0.32 * len(imp) + 1.4))
    ax.barh(imp.index, imp.values, color=SERIES_BLUE, height=0.6)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Importance")
    ax.set_title(f"Top {len(imp)} feature importances")
    fig.savefig(path)
    plt.close(fig)


def evaluate_on_test(model: Any, X_test: pd.DataFrame, y_test: pd.Series,
                     interval: dict[str, float] | None = None) -> dict[str, Any]:
    """Compute test metrics (plus interval coverage) and write the test-set plots."""
    config.PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    y_true = y_test.to_numpy(dtype=float)
    y_pred = model.predict(X_test)
    metrics: dict[str, Any] = regression_metrics(y_true, y_pred)
    metrics["median_abs_pct_error"] = round(float(np.median(np.abs(y_pred - y_true) / y_true)) * 100, 2)
    if interval:
        metrics["interval_coverage"] = interval_coverage(
            y_true, y_pred, interval["low_ratio"], interval["high_ratio"]
        )
    plot_pred_vs_actual(y_true, y_pred, config.PLOTS_DIR / "pred_vs_actual.png")
    plot_residuals(y_true, y_pred, config.PLOTS_DIR / "residuals.png")
    plot_feature_importance(model, config.PLOTS_DIR / "feature_importance.png")
    return metrics


def main() -> None:
    """Re-evaluate the saved model on the same held-out split used in training."""
    import joblib
    from sklearn.model_selection import train_test_split

    from ml.preprocess import load_and_clean

    if not config.MODEL_PATH.exists():
        raise SystemExit("No model found. Run `python -m ml.train` first.")
    model = joblib.load(config.MODEL_PATH)
    metrics_file = json.loads(config.METRICS_PATH.read_text(encoding="utf-8"))
    df, _ = load_and_clean(verbose=False, save=False)
    _, X_test, _, y_test = train_test_split(
        df[config.FEATURES], df[config.TARGET], test_size=config.TEST_SIZE, random_state=config.RANDOM_SEED
    )
    metrics = evaluate_on_test(model, X_test, y_test, metrics_file.get("interval"))
    print(f"Model: {metrics_file.get('chosen_model')}")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
