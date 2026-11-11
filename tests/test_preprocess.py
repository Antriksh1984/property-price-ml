"""Unit tests for the pure cleaning functions in ml.preprocess."""
from __future__ import annotations

import pandas as pd
import pytest

from ml import config
from ml.preprocess import (
    clean_locations,
    convert_sqft,
    parse_bhk,
    parse_sqft,
    remove_bath_outliers,
    remove_bhk_outliers,
    remove_pps_outliers,
    remove_small_units,
)


# --- total_sqft parsing -----------------------------------------------------
@pytest.mark.parametrize(
    ("raw", "expected"),
    [("1056", 1056.0), ("1200.5", 1200.5), (1440, 1440.0), (" 980 ", 980.0)],
)
def test_parse_sqft_plain_numbers(raw: object, expected: float) -> None:
    assert parse_sqft(raw) == pytest.approx(expected)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [("1200 - 1500", 1350.0), ("2100 - 2850", 2475.0), ("1000-1001", 1000.5)],
)
def test_parse_sqft_range_returns_midpoint(raw: str, expected: float) -> None:
    assert parse_sqft(raw) == pytest.approx(expected)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("100Sq. Meter", 1076.39),
        ("100Sq. Yards", 900.0),
        ("2Acres", 87120.0),
        ("3Cents", 1306.8),
        ("1Grounds", 2400.0),
        ("24Guntha", 26136.0),
        ("1Perch", 272.25),
    ],
)
def test_parse_sqft_unit_conversion(raw: str, expected: float) -> None:
    assert parse_sqft(raw) == pytest.approx(expected, rel=1e-6)


@pytest.mark.parametrize("raw", ["", "abc", "12Hectares", "1200 - ", None, float("nan")])
def test_parse_sqft_unconvertible_returns_none(raw: object) -> None:
    assert parse_sqft(raw) is None


def test_convert_sqft_drops_unconvertible_rows() -> None:
    df = pd.DataFrame({"total_sqft": ["1000", "1200 - 1400", "10Sq. Yards", "junk"]})
    out = convert_sqft(df)
    assert out["total_sqft"].tolist() == pytest.approx([1000.0, 1300.0, 90.0])


# --- BHK parsing --------------------------------------------------------------
@pytest.mark.parametrize(
    ("raw", "expected"),
    [("2 BHK", 2), ("4 Bedroom", 4), ("1 RK", 1), ("11 BHK", 11), ("3 bhk", 3)],
)
def test_parse_bhk(raw: str, expected: int) -> None:
    assert parse_bhk(raw) == expected


@pytest.mark.parametrize("raw", [None, float("nan"), "Studio", ""])
def test_parse_bhk_invalid(raw: object) -> None:
    assert parse_bhk(raw) is None


# --- Location grouping --------------------------------------------------------
def test_clean_locations_strips_and_groups_rare() -> None:
    locs = ["Whitefield"] * 11 + [" Whitefield "] * 1 + ["Tiny Place"] * 10 + ["Lonely"]
    out = clean_locations(pd.DataFrame({"location": locs}))
    counts = out["location"].value_counts()
    # Whitespace is stripped before counting: 12 Whitefield listings stay.
    assert counts["Whitefield"] == 12
    # <= 10 listings -> "other" (boundary case: exactly 10 is grouped).
    assert "Tiny Place" not in counts and "Lonely" not in counts
    assert counts[config.OTHER_LOCATION] == 11


# --- Outlier rules ------------------------------------------------------------
def test_remove_small_units() -> None:
    df = pd.DataFrame({"total_sqft": [600.0, 500.0, 1200.0], "bhk": [2, 2, 4]})
    out = remove_small_units(df)
    assert out["total_sqft"].tolist() == [600.0, 1200.0]


def test_remove_pps_outliers_keeps_within_one_std() -> None:
    df = pd.DataFrame({"location": ["A"] * 5, "price_per_sqft": [5000.0, 5100.0, 4900.0, 5050.0, 20000.0]})
    out = remove_pps_outliers(df)
    assert 20000.0 not in out["price_per_sqft"].tolist()
    assert len(out) == 4


def test_remove_bhk_outliers() -> None:
    # Six 2-BHK flats averaging 6000/sqft; a 3-BHK below that is dropped, one above is kept.
    df = pd.DataFrame(
        {
            "location": ["A"] * 8,
            "bhk": [2] * 6 + [3, 3],
            "price_per_sqft": [6000.0] * 6 + [5000.0, 7000.0],
        }
    )
    out = remove_bhk_outliers(df)
    assert out.loc[out["bhk"] == 3, "price_per_sqft"].tolist() == [7000.0]


def test_remove_bhk_outliers_needs_more_than_five_rows() -> None:
    df = pd.DataFrame({"location": ["A"] * 6, "bhk": [2] * 5 + [3], "price_per_sqft": [6000.0] * 5 + [1000.0]})
    assert len(remove_bhk_outliers(df)) == 6  # only 5 two-BHK rows -> rule does not apply


def test_remove_bath_outliers() -> None:
    df = pd.DataFrame({"bhk": [2, 2, 3], "bath": [4, 5, 3]})
    assert remove_bath_outliers(df)["bath"].tolist() == [4, 3]
