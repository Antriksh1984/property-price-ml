"""FastAPI app: health, locations, metrics and price prediction.

    uvicorn backend.main:app --reload --port 8000
"""
from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from backend.model_service import ModelService
from backend.schemas import HealthResponse, PredictRequest, PredictResponse
from ml import config

logging.basicConfig(level=logging.INFO, format="%(levelname)s:     %(name)s - %(message)s")
logger = logging.getLogger("backend")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Load the model exactly once at startup; refuse to start without it."""
    app.state.service = ModelService.load()  # raises ArtifactsMissingError with a clear message
    yield


app = FastAPI(
    title="Bengaluru Property Price API",
    description="Predicts Bengaluru property prices (lakhs INR) from location, area, BHK and bathrooms.",
    version="1.0.0",
    lifespan=lifespan,
)

# Comma-separated extra origins can be supplied via CORS_ORIGINS (e.g. for Docker).
_origins = ["http://localhost:5173", "http://127.0.0.1:5173"]
_origins += [o.strip() for o in os.getenv("CORS_ORIGINS", "").split(",") if o.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_origins,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# Model plots (comparison chart etc.) for the frontend's "About the model" section.
app.mount("/static", StaticFiles(directory=config.PLOTS_DIR, check_dir=False), name="static")


def _service(request: Request) -> ModelService:
    return request.app.state.service


@app.get("/health", response_model=HealthResponse)
def health(request: Request) -> HealthResponse:
    svc = _service(request)
    return HealthResponse(status="ok", model_name=svc.model_name, trained_at=svc.trained_at)


@app.get("/locations", response_model=list[str])
def locations(request: Request) -> list[str]:
    return _service(request).locations


@app.get("/metrics")
def metrics(request: Request) -> dict[str, Any]:
    return _service(request).metrics


@app.post("/predict", response_model=PredictResponse)
def predict(body: PredictRequest, request: Request) -> PredictResponse:
    result = _service(request).predict(body)
    if result.warning:
        logger.warning(result.warning)
    return result
