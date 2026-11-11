"""Pydantic request/response models for the prediction API."""
from __future__ import annotations

from pydantic import BaseModel, Field, model_validator

from ml import config


class PredictRequest(BaseModel):
    """Inputs for a single price prediction. Bounds mirror the frontend form."""

    location: str = Field(..., min_length=1, max_length=100, examples=["Whitefield"])
    total_sqft: float = Field(..., ge=config.SQFT_MIN, le=config.SQFT_MAX, examples=[1200])
    bhk: int = Field(..., ge=config.BHK_MIN, le=config.BHK_MAX, examples=[2])
    bath: int = Field(..., ge=config.BATH_MIN, le=config.BATH_MAX, examples=[2])

    @model_validator(mode="after")
    def check_bath_vs_bhk(self) -> "PredictRequest":
        """Same sanity rule used to clean the training data."""
        if self.bath > self.bhk + config.MAX_EXTRA_BATHS:
            raise ValueError(f"bath must be at most bhk + {config.MAX_EXTRA_BATHS} (got bath={self.bath}, bhk={self.bhk})")
        return self


class PredictResponse(BaseModel):
    """Predicted price (lakhs INR) with an ~80% range."""

    predicted_price_lakhs: float
    low_lakhs: float
    high_lakhs: float
    price_per_sqft: float = Field(..., description="Predicted price per sqft in INR")
    model_name: str
    location_used: str = Field(..., description="Location the model actually used (may be 'other')")
    warning: str | None = None


class HealthResponse(BaseModel):
    status: str
    model_name: str
    trained_at: str
