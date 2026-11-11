"""API tests using FastAPI's TestClient against the trained artifacts."""
from __future__ import annotations

from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

from ml import config

pytestmark = pytest.mark.skipif(
    not config.MODEL_PATH.exists(), reason="Model artifacts missing - run `python -m ml.train` first."
)


@pytest.fixture(scope="module")
def client() -> Iterator[TestClient]:
    from backend.main import app

    with TestClient(app) as c:  # context manager runs the lifespan (model load)
        yield c


def test_health(client: TestClient) -> None:
    res = client.get("/health")
    assert res.status_code == 200
    body = res.json()
    assert body["status"] == "ok"
    assert body["model_name"]
    assert body["trained_at"]


def test_locations(client: TestClient) -> None:
    res = client.get("/locations")
    assert res.status_code == 200
    locations = res.json()
    assert isinstance(locations, list) and len(locations) > 50
    assert config.OTHER_LOCATION in locations
    assert "Whitefield" in locations


def test_metrics(client: TestClient) -> None:
    body = client.get("/metrics").json()
    assert {"chosen_model", "test_metrics", "cv_results", "interval"} <= body.keys()


def test_predict_valid(client: TestClient) -> None:
    res = client.post("/predict", json={"location": "Whitefield", "total_sqft": 1200, "bhk": 2, "bath": 2})
    assert res.status_code == 200
    body = res.json()
    price = body["predicted_price_lakhs"]
    assert 10 < price < 500  # a 1200 sqft 2BHK in Whitefield is tens of lakhs, not crores
    assert body["low_lakhs"] < price < body["high_lakhs"]
    assert body["price_per_sqft"] == pytest.approx(price * 1e5 / 1200, rel=1e-3)
    assert body["location_used"] == "Whitefield"
    assert body["warning"] is None
    # Outputs are rounded to 2 decimals.
    assert round(price, 2) == price


def test_predict_location_case_insensitive(client: TestClient) -> None:
    res = client.post("/predict", json={"location": "  whitefield ", "total_sqft": 1200, "bhk": 2, "bath": 2})
    assert res.json()["location_used"] == "Whitefield"


def test_predict_unknown_location_maps_to_other(client: TestClient) -> None:
    res = client.post("/predict", json={"location": "Atlantis", "total_sqft": 1500, "bhk": 3, "bath": 3})
    assert res.status_code == 200
    body = res.json()
    assert body["location_used"] == config.OTHER_LOCATION
    assert "Atlantis" in body["warning"]


def test_bigger_home_costs_more(client: TestClient) -> None:
    def price(sqft: int, bhk: int) -> float:
        payload = {"location": "Whitefield", "total_sqft": sqft, "bhk": bhk, "bath": bhk}
        return client.post("/predict", json=payload).json()["predicted_price_lakhs"]

    assert price(2400, 4) > price(1000, 2)


@pytest.mark.parametrize(
    "payload",
    [
        {"location": "Whitefield", "total_sqft": 100, "bhk": 2, "bath": 2},  # sqft too small
        {"location": "Whitefield", "total_sqft": 50000, "bhk": 2, "bath": 2},  # sqft too large
        {"location": "Whitefield", "total_sqft": 1200, "bhk": 0, "bath": 1},  # bhk too small
        {"location": "Whitefield", "total_sqft": 1200, "bhk": 2, "bath": 5},  # bath > bhk + 2
        {"location": "Whitefield", "total_sqft": 1200, "bhk": 2},  # missing bath
        {"location": "", "total_sqft": 1200, "bhk": 2, "bath": 2},  # empty location
    ],
)
def test_predict_invalid_returns_422(client: TestClient, payload: dict) -> None:
    assert client.post("/predict", json=payload).status_code == 422


def test_static_plot_served(client: TestClient) -> None:
    res = client.get("/static/model_comparison.png")
    assert res.status_code == 200
    assert res.headers["content-type"] == "image/png"
