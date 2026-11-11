"""Train, compare, tune and save the property price model.

    python -m ml.train

Steps
-----
1. Clean the raw data (ml.preprocess) and split 80/20 (seed 42).
2. Compare candidate models with 5-fold CV on the training split.
3. Tune the top 2 with RandomizedSearchCV (~20 iterations each).
4. Pick the best by CV RMSE, compute a prediction interval from out-of-fold
   residual ratios, evaluate once on the held-out test set.
5. Save model.joblib, locations.json, metrics.json and plots to artifacts/.

All feature transforms live inside the sklearn Pipeline, so they are fitted
only on the training folds (no leakage).
"""
from __future__ import annotations

import json
import logging
import time
import warnings
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import joblib
import numpy as np
import pandas as pd
from scipy.stats import loguniform, randint, uniform
from sklearn.base import RegressorMixin
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.model_selection import KFold, RandomizedSearchCV, cross_val_predict, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from ml import config
from ml.evaluate import evaluate_on_test, plot_model_comparison
from ml.preprocess import load_and_clean

logger = logging.getLogger(__name__)

SCORING = {"r2": "r2", "mae": "neg_mean_absolute_error", "rmse": "neg_root_mean_squared_error"}


# ---------------------------------------------------------------------------
# Model definitions
# ---------------------------------------------------------------------------
@dataclass
class Candidate:
    """A model to compare, plus the search space used if it makes the top 2."""

    name: str
    estimator: RegressorMixin
    scale_numeric: bool  # StandardScaler on numerics (linear models only)
    search_space: dict[str, Any] = field(default_factory=dict)


def _boosting_candidate() -> Candidate:
    """XGBoost if installed, else LightGBM, else sklearn's HistGradientBoosting."""
    try:
        from xgboost import XGBRegressor

        return Candidate(
            "XGBoost",
            XGBRegressor(n_estimators=400, learning_rate=0.05, max_depth=6, subsample=0.9,
                         colsample_bytree=0.8, random_state=config.RANDOM_SEED, n_jobs=1, verbosity=0),
            scale_numeric=False,
            search_space={
                "n_estimators": randint(200, 900),
                "learning_rate": loguniform(0.02, 0.2),
                "max_depth": randint(3, 10),
                "min_child_weight": randint(1, 8),
                "subsample": uniform(0.6, 0.4),
                "colsample_bytree": uniform(0.4, 0.6),
                "reg_lambda": loguniform(0.1, 10),
            },
        )
    except ImportError:
        pass
    try:
        from lightgbm import LGBMRegressor

        return Candidate(
            "LightGBM",
            LGBMRegressor(n_estimators=400, learning_rate=0.05, random_state=config.RANDOM_SEED, verbose=-1),
            scale_numeric=False,
            search_space={
                "n_estimators": randint(200, 900),
                "learning_rate": loguniform(0.02, 0.2),
                "num_leaves": randint(15, 127),
                "min_child_samples": randint(5, 50),
                "subsample": uniform(0.6, 0.4),
                "colsample_bytree": uniform(0.4, 0.6),
            },
        )
    except ImportError:
        pass
    from sklearn.ensemble import HistGradientBoostingRegressor

    return Candidate(
        "HistGradientBoosting",
        HistGradientBoostingRegressor(random_state=config.RANDOM_SEED),
        scale_numeric=False,
        search_space={
            "learning_rate": loguniform(0.02, 0.2),
            "max_iter": randint(200, 800),
            "max_leaf_nodes": randint(15, 63),
            "min_samples_leaf": randint(5, 40),
            "l2_regularization": loguniform(1e-3, 10),
        },
    )


def get_candidates() -> list[Candidate]:
    """All models to compare in cross-validation."""
    return [
        Candidate("LinearRegression", LinearRegression(), scale_numeric=True),
        Candidate("Ridge", Ridge(alpha=1.0), scale_numeric=True,
                  search_space={"alpha": loguniform(1e-3, 1e2)}),
        # Default alpha=1.0 zeroes every coefficient on a log target; 1e-3 is a sane start.
        Candidate("Lasso", Lasso(alpha=1e-3, max_iter=20000), scale_numeric=True,
                  search_space={"alpha": loguniform(1e-5, 1e-1)}),
        Candidate(
            "RandomForest",
            RandomForestRegressor(n_estimators=200, random_state=config.RANDOM_SEED, n_jobs=1),
            scale_numeric=False,
            search_space={
                "n_estimators": randint(100, 350),
                "max_depth": [None, 10, 20, 30],
                "min_samples_leaf": randint(1, 6),
                "max_features": [0.3, 0.5, 0.7, 1.0, "sqrt"],
            },
        ),
        _boosting_candidate(),
    ]


def build_pipeline(estimator: RegressorMixin, scale_numeric: bool) -> TransformedTargetRegressor:
    """Preprocessing + model, trained on log1p(price) and inverted with expm1.

    Linear models get log1p(total_sqft) then StandardScaler on all numerics, so
    they fit a log-log relationship (price grows multiplicatively with area).
    Tree models get the raw numerics: they are invariant to monotonic scaling.
    """
    location = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    if scale_numeric:
        transformers = [
            ("location", location, config.CATEGORICAL_FEATURES),
            ("sqft", Pipeline([("log", FunctionTransformer(np.log1p, feature_names_out="one-to-one")),
                               ("scale", StandardScaler())]), ["total_sqft"]),
            ("numeric", StandardScaler(), ["bhk", "bath"]),
        ]
    else:
        transformers = [
            ("location", location, config.CATEGORICAL_FEATURES),
            ("numeric", "passthrough", config.NUMERIC_FEATURES),
        ]
    prep = ColumnTransformer(transformers)
    pipe = Pipeline([("prep", prep), ("model", estimator)])
    return TransformedTargetRegressor(regressor=pipe, func=np.log1p, inverse_func=np.expm1)


def _prefixed(space: dict[str, Any]) -> dict[str, Any]:
    """Map bare model params to the nested TransformedTargetRegressor path."""
    return {f"regressor__model__{k}": v for k, v in space.items()}


def _cv_summary(name: str, scores: dict[str, np.ndarray], stage: str) -> dict[str, Any]:
    return {
        "model": name,
        "stage": stage,
        "r2": round(float(np.mean(scores["test_r2"])), 4),
        "r2_std": round(float(np.std(scores["test_r2"])), 4),
        "mae": round(float(-np.mean(scores["test_mae"])), 3),
        "rmse": round(float(-np.mean(scores["test_rmse"])), 3),
        "rmse_std": round(float(np.std(scores["test_rmse"])), 3),
    }


def print_table(rows: list[dict[str, Any]], title: str) -> None:
    """Print a fixed-width comparison table to the console."""
    print(f"\n{title}")
    header = f"{'Model':<28}{'R2':>9}{'MAE (L)':>11}{'RMSE (L)':>11}{'RMSE sd':>10}"
    print(header)
    print("-" * len(header))
    for r in sorted(rows, key=lambda r: r["rmse"]):
        print(f"{r['model']:<28}{r['r2']:>9.4f}{r['mae']:>11.2f}{r['rmse']:>11.2f}{r['rmse_std']:>10.2f}")


# ---------------------------------------------------------------------------
# Main training routine
# ---------------------------------------------------------------------------
def train() -> dict[str, Any]:
    """Run the full training workflow and write all artifacts. Returns metrics."""
    start = time.time()
    config.ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    config.PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Data --------------------------------------------------------------
    df, cleaning_log = load_and_clean(verbose=True, save=True)
    X, y = df[config.FEATURES], df[config.TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=config.TEST_SIZE, random_state=config.RANDOM_SEED
    )
    print(f"\nTrain rows: {len(X_train)}  |  Test rows: {len(X_test)}")
    cv = KFold(n_splits=config.CV_FOLDS, shuffle=True, random_state=config.RANDOM_SEED)

    # 2. Baseline comparison ----------------------------------------------
    candidates = {c.name: c for c in get_candidates()}
    cv_results: list[dict[str, Any]] = []
    for cand in candidates.values():
        pipe = build_pipeline(cand.estimator, cand.scale_numeric)
        scores = cross_validate(pipe, X_train, y_train, cv=cv, scoring=SCORING, n_jobs=-1)
        cv_results.append(_cv_summary(cand.name, scores, "baseline"))
        print(f"  CV done: {cand.name:<22} RMSE {cv_results[-1]['rmse']:.2f}")
    print_table(cv_results, "Baseline 5-fold CV (training split)")

    # 3. Tune the top 2 tunable models ------------------------------------
    # Plain LinearRegression has no hyperparameters, so it is skipped here
    # (it still competes in the final selection with its baseline score).
    ranked = [r["model"] for r in sorted(cv_results, key=lambda r: r["rmse"])]
    top2 = [name for name in ranked if candidates[name].search_space][:2]
    print(f"\nTuning top 2 tunable models: {', '.join(top2)} ({config.SEARCH_ITERATIONS} iterations each)")
    tuned: dict[str, dict[str, Any]] = {}
    for name in top2:
        cand = candidates[name]
        search = RandomizedSearchCV(
            build_pipeline(cand.estimator, cand.scale_numeric),
            _prefixed(cand.search_space),
            n_iter=config.SEARCH_ITERATIONS,
            scoring=SCORING,
            refit="rmse",
            cv=cv,
            random_state=config.RANDOM_SEED,
            n_jobs=-1,
        )
        search.fit(X_train, y_train)
        i = search.best_index_
        res = search.cv_results_
        summary = {
            "model": f"{name} (tuned)",
            "stage": "tuned",
            "r2": round(float(res["mean_test_r2"][i]), 4),
            "r2_std": round(float(res["std_test_r2"][i]), 4),
            "mae": round(float(-res["mean_test_mae"][i]), 3),
            "rmse": round(float(-res["mean_test_rmse"][i]), 3),
            "rmse_std": round(float(res["std_test_rmse"][i]), 3),
            "best_params": {k.replace("regressor__model__", ""): _jsonable(v) for k, v in search.best_params_.items()},
        }
        tuned[summary["model"]] = {"summary": summary, "estimator": search.best_estimator_}
        cv_results.append(summary)
        print(f"  Tuned {name}: CV RMSE {summary['rmse']:.2f}  params={summary['best_params']}")
    print_table(cv_results, "All models incl. tuned (5-fold CV RMSE; lower is better)")

    # 4. Select the best by CV RMSE ---------------------------------------
    best = min(cv_results, key=lambda r: r["rmse"])
    chosen_name = best["model"]
    if chosen_name in tuned:
        model = tuned[chosen_name]["estimator"]
    else:
        cand = candidates[chosen_name]
        model = build_pipeline(cand.estimator, cand.scale_numeric)
    print(f"\nChosen model: {chosen_name}")

    # Prediction interval from out-of-fold residual ratios (training split only)
    oof_pred = cross_val_predict(model, X_train, y_train, cv=cv, n_jobs=-1)
    ratios = y_train.to_numpy() / oof_pred
    q_lo, q_hi = config.INTERVAL_QUANTILES
    interval = {
        "low_quantile": q_lo,
        "high_quantile": q_hi,
        "low_ratio": round(float(np.quantile(ratios, q_lo)), 4),
        "high_ratio": round(float(np.quantile(ratios, q_hi)), 4),
    }
    print(f"Interval ratios (actual/predicted): p10={interval['low_ratio']}, p90={interval['high_ratio']}")

    # 5. Fit on full training split, evaluate once on test ---------------
    model.fit(X_train, y_train)
    test_metrics = evaluate_on_test(model, X_test, y_test, interval)
    print(
        f"Test set: R2={test_metrics['r2']:.4f}  MAE={test_metrics['mae']:.2f} L  RMSE={test_metrics['rmse']:.2f} L  "
        f"80% interval coverage={test_metrics['interval_coverage']:.1%}"
    )

    # 6. Save artifacts -----------------------------------------------------
    plot_model_comparison(cv_results, chosen_name, config.PLOTS_DIR / "model_comparison.png")
    joblib.dump(model, config.MODEL_PATH)

    locations = sorted(loc for loc in X_train["location"].unique() if loc != config.OTHER_LOCATION)
    locations.append(config.OTHER_LOCATION)
    config.LOCATIONS_PATH.write_text(json.dumps(locations, indent=2), encoding="utf-8")

    metrics = {
        "chosen_model": chosen_name,
        "chosen_params": best.get("best_params", {}),
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "target": "price (lakhs INR), modelled as log1p(price)",
        "features": config.FEATURES,
        "cv_folds": config.CV_FOLDS,
        "random_seed": config.RANDOM_SEED,
        "cv_results": cv_results,
        "test_metrics": test_metrics,
        "interval": interval,
        "dataset": {
            "raw_rows": cleaning_log[0]["before"],
            "clean_rows": len(df),
            "train_rows": len(X_train),
            "test_rows": len(X_test),
            "n_locations": len(locations),
            "cleaning_steps": cleaning_log,
        },
        "training_seconds": round(time.time() - start, 1),
    }
    config.METRICS_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"\nSaved model, locations ({len(locations)}), metrics and plots to {config.ARTIFACTS_DIR}")
    print(f"Done in {metrics['training_seconds']}s")
    return metrics


def _jsonable(value: Any) -> Any:
    """Convert numpy scalars to plain Python types for JSON."""
    if isinstance(value, np.generic):
        return value.item()
    return value


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    warnings.filterwarnings("ignore", category=UserWarning)  # OHE unknown-category notices in CV folds
    pd.set_option("display.width", 120)
    train()
