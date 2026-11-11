"""Cleaning and feature engineering for the Bengaluru House Price dataset.

Every step is a pure function (DataFrame in -> new DataFrame out) so it can be
unit-tested and reused from the notebook. ``run_pipeline`` chains them together
and logs a before/after row count for each step.

Run directly to produce ``data/processed/bengaluru_clean.csv``::

    python -m ml.preprocess
"""
from __future__ import annotations

import logging
import math
import re
from typing import Callable

import pandas as pd

from ml import config

logger = logging.getLogger(__name__)

_RANGE_RE = re.compile(r"^\s*([\d.]+)\s*-\s*([\d.]+)\s*$")
_UNIT_RE = re.compile(r"^\s*([\d.]+)\s*([A-Za-z. ]+?)\s*$")


# ---------------------------------------------------------------------------
# Parsers (scalar helpers)
# ---------------------------------------------------------------------------
def parse_bhk(size: object) -> int | None:
    """Convert a size string like '2 BHK', '4 Bedroom' or '1 RK' into an int.

    An RK (room + kitchen) is treated as 1 BHK. Returns None if unparseable.
    """
    if size is None or (isinstance(size, float) and math.isnan(size)):
        return None
    match = re.match(r"^\s*(\d+)\s*(BHK|Bedroom|RK)\b", str(size), flags=re.IGNORECASE)
    if not match:
        return None
    return max(int(match.group(1)), 1)


def parse_sqft(value: object) -> float | None:
    """Convert a raw total_sqft value into square feet.

    Handles plain numbers ('1056'), ranges ('1200 - 1500' -> midpoint) and
    numbers with a unit suffix ('34.46Sq. Meter', '2Acres'). Returns None when
    the value cannot be converted.
    """
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    text = str(value).strip()

    # Plain number
    try:
        return float(text)
    except ValueError:
        pass

    # Range -> midpoint
    range_match = _RANGE_RE.match(text)
    if range_match:
        low, high = float(range_match.group(1)), float(range_match.group(2))
        return (low + high) / 2.0

    # Number + unit
    unit_match = _UNIT_RE.match(text)
    if unit_match:
        number, unit = float(unit_match.group(1)), unit_match.group(2).strip()
        factor = config.UNIT_TO_SQFT.get(unit)
        if factor is not None:
            return number * factor
    return None


# ---------------------------------------------------------------------------
# DataFrame steps (each returns a new frame)
# ---------------------------------------------------------------------------
def drop_unused_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Drop columns we deliberately ignore to keep the model simple."""
    return df.drop(columns=[c for c in config.DROP_COLUMNS if c in df.columns])


def drop_missing(df: pd.DataFrame) -> pd.DataFrame:
    """Drop rows missing any of the fields we need."""
    return df.dropna(subset=["location", "size", "bath"]).copy()


def add_bhk(df: pd.DataFrame) -> pd.DataFrame:
    """Parse ``size`` into an integer ``bhk`` column and drop ``size``."""
    out = df.copy()
    out["bhk"] = out["size"].map(parse_bhk)
    out = out.dropna(subset=["bhk"]).drop(columns=["size"])
    out["bhk"] = out["bhk"].astype(int)
    return out


def convert_sqft(df: pd.DataFrame) -> pd.DataFrame:
    """Convert ``total_sqft`` to float square feet, dropping unconvertible rows."""
    out = df.copy()
    raw = out["total_sqft"].astype(str).str.strip()
    is_range = raw.str.contains(r"\d\s*-\s*\d", regex=True)
    is_unit = raw.str.contains(r"[A-Za-z]", regex=True)
    out["total_sqft"] = out["total_sqft"].map(parse_sqft)
    n_bad = int(out["total_sqft"].isna().sum())
    logger.info(
        "    total_sqft: %d range(s) -> midpoint, %d unit value(s) converted, %d unconvertible row(s) dropped",
        int(is_range.sum()), int((is_unit & out["total_sqft"].notna()).sum()), n_bad,
    )
    return out.dropna(subset=["total_sqft"])


def clean_locations(df: pd.DataFrame, min_listings: int = config.MIN_LISTINGS_PER_LOCATION) -> pd.DataFrame:
    """Strip whitespace and group rare locations (<= ``min_listings``) into 'other'."""
    out = df.copy()
    out["location"] = out["location"].astype(str).str.strip()
    counts = out["location"].value_counts()
    rare = counts[counts <= min_listings].index
    out.loc[out["location"].isin(rare), "location"] = config.OTHER_LOCATION
    return out


def add_price_per_sqft(df: pd.DataFrame) -> pd.DataFrame:
    """Add price_per_sqft (INR). Used only for outlier removal, never as a feature."""
    out = df.copy()
    out["price_per_sqft"] = out["price"] * 1e5 / out["total_sqft"]
    return out


def remove_small_units(df: pd.DataFrame, min_sqft_per_bhk: float = config.MIN_SQFT_PER_BHK) -> pd.DataFrame:
    """Drop listings with implausibly small rooms (total_sqft / bhk < threshold)."""
    return df[df["total_sqft"] / df["bhk"] >= min_sqft_per_bhk].copy()


def remove_pps_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """Per location, keep rows whose price_per_sqft lies within mean +/- 1 std."""
    grouped = df.groupby("location")["price_per_sqft"]
    mean = grouped.transform("mean")
    std = grouped.transform("std").fillna(0.0)  # single-row groups: std NaN -> keep row
    mask = (df["price_per_sqft"] >= mean - std) & (df["price_per_sqft"] <= mean + std)
    return df[mask].copy()


def remove_bhk_outliers(df: pd.DataFrame, min_group: int = config.MIN_GROUP_SIZE_FOR_BHK_RULE) -> pd.DataFrame:
    """Per location, drop n-BHK flats cheaper per sqft than the mean (n-1)-BHK flat.

    The rule only applies when the (n-1)-BHK group in that location has more
    than ``min_group`` rows.
    """
    stats = (
        df.groupby(["location", "bhk"])["price_per_sqft"]
        .agg(["mean", "count"])
        .reset_index()
    )
    # Shift stats so each row describes the (n-1)-BHK group for a given n.
    prev = stats.assign(bhk=stats["bhk"] + 1).rename(columns={"mean": "prev_mean", "count": "prev_count"})
    merged = df.merge(prev, on=["location", "bhk"], how="left")
    merged.index = df.index
    is_outlier = (merged["prev_count"] > min_group) & (merged["price_per_sqft"] < merged["prev_mean"])
    return df[~is_outlier].copy()


def remove_bath_outliers(df: pd.DataFrame, max_extra: int = config.MAX_EXTRA_BATHS) -> pd.DataFrame:
    """Drop listings with more than bhk + ``max_extra`` bathrooms."""
    return df[df["bath"] <= df["bhk"] + max_extra].copy()


def restrict_to_serving_range(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only rows inside the input range the API accepts.

    The model should be trained on the same domain it serves. This also removes
    obvious data-entry errors such as a 653,400 sqft (15 acre) "3 BHK".
    """
    mask = (
        df["total_sqft"].between(config.SQFT_MIN, config.SQFT_MAX)
        & df["bhk"].between(config.BHK_MIN, config.BHK_MAX)
        & df["bath"].between(config.BATH_MIN, config.BATH_MAX)
    )
    return df[mask].copy()


def finalize(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only model features + target, with tidy dtypes."""
    out = df[config.FEATURES + [config.TARGET]].copy()
    out["bhk"] = out["bhk"].astype(int)
    out["bath"] = out["bath"].astype(int)
    return out.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------
STEPS: list[tuple[str, Callable[[pd.DataFrame], pd.DataFrame]]] = [
    ("Drop unused columns", drop_unused_columns),
    ("Drop rows missing location/size/bath", drop_missing),
    ("Parse size -> bhk", add_bhk),
    ("Convert total_sqft to sq ft", convert_sqft),
    ("Clean + group rare locations", clean_locations),
    ("Add price_per_sqft", add_price_per_sqft),
    ("Remove sqft/bhk < 300", remove_small_units),
    ("Remove per-location pps outliers (mean +/- 1 std)", remove_pps_outliers),
    ("Remove BHK price-inversion outliers", remove_bhk_outliers),
    ("Remove bath > bhk + 2", remove_bath_outliers),
    ("Keep API range (sqft 300-30000, bhk/bath <= 10)", restrict_to_serving_range),
    ("Select final features", finalize),
]


def run_pipeline(df: pd.DataFrame, verbose: bool = True) -> tuple[pd.DataFrame, list[dict[str, object]]]:
    """Run every cleaning step and return (clean_df, per-step row-count log)."""
    log: list[dict[str, object]] = []
    current = df
    for name, step in STEPS:
        before = len(current)
        current = step(current)
        after = len(current)
        log.append({"step": name, "before": before, "after": after, "dropped": before - after})
        if verbose:
            print(f"  {name:<50} {before:>6} -> {after:>6}  (dropped {before - after})")
    return current, log


def load_raw(path=config.RAW_DATA_PATH) -> pd.DataFrame:
    """Read the raw CSV, failing with a helpful message if it's missing."""
    if not path.exists():
        raise FileNotFoundError(
            f"Raw dataset not found at {path}.\n"
            "Download 'Bengaluru House Price Data' from Kaggle and save it there."
        )
    return pd.read_csv(path)


def load_and_clean(verbose: bool = True, save: bool = True) -> tuple[pd.DataFrame, list[dict[str, object]]]:
    """Load the raw CSV, clean it, optionally save to data/processed/."""
    raw = load_raw()
    if verbose:
        print(f"Preprocessing {config.RAW_DATA_PATH.name} ({len(raw)} rows)")
    clean, log = run_pipeline(raw, verbose=verbose)
    if save:
        config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
        clean.to_csv(config.PROCESSED_DATA_PATH, index=False)
        if verbose:
            print(f"Saved {len(clean)} clean rows to {config.PROCESSED_DATA_PATH}")
    return clean, log


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    df_clean, _ = load_and_clean()
    print(df_clean.describe(include="all").T)
    print(f"Unique locations: {df_clean['location'].nunique()}")
