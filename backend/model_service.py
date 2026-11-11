"""Loads the trained pipeline once and turns requests into predictions."""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from backend.schemas import PredictRequest, PredictResponse
from ml import config

logger = logging.getLogger(__name__)


class ArtifactsMissingError(RuntimeError):
    """Raised when model artifacts have not been produced yet."""


@dataclass
class ModelService:
    """Holds the fitted pipeline plus metadata needed at inference time."""

    model: Any
    locations: list[str]
    metrics: dict[str, Any]

    @classmethod
    def load(
        cls,
        model_path: Path = config.MODEL_PATH,
        locations_path: Path = config.LOCATIONS_PATH,
        metrics_path: Path = config.METRICS_PATH,
    ) -> "ModelService":
        """Load artifacts from disk, failing with an actionable message if absent."""
        missing = [p.name for p in (model_path, locations_path, metrics_path) if not p.exists()]
        if missing:
            raise ArtifactsMissingError(
                f"Model artifacts missing ({', '.join(missing)}) in {model_path.parent}. "
                "Run `python -m ml.train` first."
            )
        model = joblib.load(model_path)
        locations = json.loads(locations_path.read_text(encoding="utf-8"))
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        logger.info("Loaded %s trained at %s (%d locations)", metrics.get("chosen_model"),
                    metrics.get("trained_at"), len(locations))
        return cls(model=model, locations=locations, metrics=metrics)

    def __post_init__(self) -> None:
        # Case-insensitive lookup so "whitefield " still matches "Whitefield".
        self._lookup = {loc.strip().lower(): loc for loc in self.locations}

    @property
    def model_name(self) -> str:
        return str(self.metrics.get("chosen_model", "unknown"))

    @property
    def trained_at(self) -> str:
        return str(self.metrics.get("trained_at", "unknown"))

    def resolve_location(self, location: str) -> tuple[str, str | None]:
        """Map user input to a known location, falling back to 'other' with a warning."""
        match = self._lookup.get(location.strip().lower())
        if match is not None:
            return match, None
        return config.OTHER_LOCATION, (
            f"Location '{location}' is not in the training data; "
            f"using the generic '{config.OTHER_LOCATION}' category, so the estimate is less specific."
        )

    def predict(self, req: PredictRequest) -> PredictResponse:
        """Predict a price and an ~80% range from out-of-fold residual quantiles."""
        location, warning = self.resolve_location(req.location)
        row = pd.DataFrame(
            [{"location": location, "total_sqft": float(req.total_sqft), "bhk": req.bhk, "bath": req.bath}],
            columns=config.FEATURES,
        )
        price = float(self.model.predict(row)[0])
        interval = self.metrics.get("interval", {})
        low = price * float(interval.get("low_ratio", 1.0))
        high = price * float(interval.get("high_ratio", 1.0))
        return PredictResponse(
            predicted_price_lakhs=round(price, 2),
            low_lakhs=round(low, 2),
            high_lakhs=round(high, 2),
            price_per_sqft=round(price * 1e5 / req.total_sqft, 2),
            model_name=self.model_name,
            location_used=location,
            warning=warning,
        )
