# Bengaluru Property Price Prediction

An end-to-end, locally runnable ML project. Pick a Bengaluru locality, enter the built-up area,
bedrooms (BHK) and bathrooms, and get a **predicted price in ₹ lakhs with a likely range**, from
a model trained on real listing data.

- **Model:** XGBoost (tuned), trained on `log1p(price)`
- **Test set (1,477 held-out listings):** R² **0.912**, MAE **₹ 14.49 L**, RMSE **₹ 25.70 L**,
  median absolute error **11.3 %**
- **Stack:** pandas · scikit-learn · XGBoost · FastAPI · React + Vite (plain JS, hand-written CSS)

---

## Architecture

```
                 ┌──────────────────────────────── offline ─────────────────────────────────┐
                 │                                                                          │
 data/raw/       │   ml/preprocess.py          ml/train.py                 ml/evaluate.py    │
 bengaluru_  ───────► clean + feature  ───────► 5-fold CV of 5 models ───► test metrics     │
 house_data.csv  │    engineering              RandomizedSearchCV         + plots          │
                 │        │                    pick best by CV RMSE            │            │
                 │        ▼                           │                        │            │
                 │  data/processed/                   ▼                        ▼            │
                 │  bengaluru_clean.csv     artifacts/  model.joblib  locations.json        │
                 │                                      metrics.json  plots/*.png           │
                 └──────────────────────────────────────────┬───────────────────────────────┘
                                                            │ loaded once at startup
                                                            ▼
 ┌────────────────────────────┐   HTTP/JSON   ┌──────────────────────────────────────────┐
 │ frontend/  (React + Vite)  │ ────────────► │ backend/  (FastAPI, uvicorn :8000)       │
 │ http://localhost:5173      │ ◄──────────── │  GET  /health  /locations  /metrics      │
 │ form · result · about      │               │  POST /predict                           │
 └────────────────────────────┘               │  GET  /static/*.png  (artifacts/plots)   │
                                              └──────────────────────────────────────────┘
```

## Dataset

[Bengaluru House Price Data](https://www.kaggle.com/datasets/amitabhajoy/bengaluru-house-price-data)
(Kaggle). 13,320 listings with `area_type, availability, location, size, society, total_sqft, bath,
balcony, price` (price in lakhs, 1 lakh = ₹100,000).

The CSV is **not** committed. Download it and save it as:

```
data/raw/bengaluru_house_data.csv
```

### Cleaning steps (`ml/preprocess.py`)

Each step is a pure, unit-tested function. Row counts come from the actual run (`metrics.json → dataset.cleaning_steps`):

| # | Step | Rows before | Rows after | Dropped |
|---|------|-----------:|-----------:|--------:|
| 1 | Drop `area_type`, `availability`, `society`, `balcony` | 13,320 | 13,320 | 0 |
| 2 | Drop rows missing location / size / bath | 13,320 | 13,246 | 74 |
| 3 | `size` → `bhk` ("2 BHK", "4 Bedroom", "1 RK") | 13,246 | 13,246 | 0 |
| 4 | `total_sqft` → float: ranges → midpoint (144 rows); units converted (46 rows: Sq. Meter, Sq. Yards, Acres, Cents, Guntha, Grounds, Perch) | 13,246 | 13,246 | 0 |
| 5 | Strip location whitespace; locations with ≤ 10 listings → `other` | 13,246 | 13,246 | 0 |
| 6 | Add `price_per_sqft` (outlier removal only, **never a feature**) | 13,246 | 13,246 | 0 |
| 7 | Drop `total_sqft / bhk < 300` | 13,246 | 12,498 | 748 |
| 8 | Per location, keep `price_per_sqft` within mean ± 1 std | 12,498 | 10,315 | 2,183 |
| 9 | Per location, drop n-BHK flats cheaper per sqft than the mean (n-1)-BHK flat (when that group has > 5 rows) | 10,315 | 7,393 | 2,922 |
| 10 | Drop `bath > bhk + 2` | 7,393 | 7,389 | 4 |
| 11 | Keep the range the API accepts (sqft 300–30,000, BHK/bath ≤ 10) | 7,389 | 7,382 | 7 |

Final features: `location` (categorical, 242 values incl. `other`), `total_sqft`, `bhk`, `bath`.
Target: `price` (lakhs).

> **Step 11 was added** beyond the original spec after training exposed a 653,400 sqft (15-acre)
> "3 BHK" listed at ₹217 L. It removes such data-entry errors and makes the model train on the same
> domain the API serves.

## Model comparison

80/20 train/test split (seed 42). Every model is a scikit-learn `Pipeline`
(`OneHotEncoder(handle_unknown="ignore")` on location; for linear models, `log1p` on sqft and
`StandardScaler` on numerics) wrapped in
`TransformedTargetRegressor(func=np.log1p, inverse_func=np.expm1)`, so all preprocessing is fitted
inside each training fold.

5-fold cross-validation on the training split (5,905 rows), sorted by RMSE. MAE and RMSE are in lakhs:

| Model | CV R² | CV MAE (L) | CV RMSE (L) | RMSE std |
|-------|------:|-----------:|------------:|---------:|
| **XGBoost (tuned)** ✅ | **0.8336** | **15.55** | **37.82** | 17.15 |
| XGBoost | 0.8093 | 16.40 | 40.62 | 17.54 |
| LinearRegression | 0.7204 | 15.54 | 40.84 | 20.91 |
| Ridge (tuned) | 0.7204 | 15.54 | 40.84 | 20.91 |
| Ridge | 0.7150 | 15.83 | 41.95 | 20.27 |
| RandomForest | 0.7293 | 17.17 | 47.52 | 16.26 |
| Lasso | 0.5748 | 20.80 | 55.95 | 20.00 |

The two best *tunable* models (XGBoost, Ridge) were tuned with `RandomizedSearchCV` (20 iterations each).
Plain LinearRegression has no hyperparameters, so it was compared but not tuned.

**Held-out test set (1,477 rows, evaluated once):**

| Metric | Value |
|--------|------:|
| R² | **0.9124** |
| MAE | ₹ 14.49 L |
| RMSE | ₹ 25.70 L |
| Median absolute % error | 11.29 % |
| Coverage of the 80 % range | 81.3 % |

The test R² (0.91) is higher than the CV mean (0.83 ± 0.10). RMSE is dominated by a few multi-crore
listings, so it varies a lot between folds, and this particular test split happens to hold fewer extreme
homes. **A realistic expectation is R² of about 0.83–0.91.**

**Prediction range.** Out-of-fold predictions on the training split give the ratio
`actual / predicted`. Its 10th and 90th percentiles (**0.798** and **1.266**) are saved in `metrics.json`,
and the API returns `low = pred × 0.798` and `high = pred × 1.266`. On the test set, 81.3 % of actual
prices fell inside that range.

Plots in `artifacts/plots/`: `model_comparison.png`, `pred_vs_actual.png`, `residuals.png`,
`feature_importance.png`.

## Screenshots

> _Placeholder: add screenshots of the prediction form, result card and "About the model" section here._
>
> `docs/screenshot-form.png` · `docs/screenshot-result.png`

## Setup & run (Windows)

Prerequisites: **Python 3.10+** and **Node 18+** on PATH, plus the dataset in `data/raw/`.

### One command

```powershell
.\run.ps1
```

This command:

1. creates `.venv` and installs `requirements.txt` (and re-installs only when the requirements change);
2. trains the model if `artifacts/` is missing (about 2 minutes);
3. runs `npm install` if needed;
4. opens the backend (:8000) and the frontend (:5173) in **separate windows**, then opens your browser.

Options: `.\run.ps1 -Retrain` forces a retrain; `.\run.ps1 -SetupOnly` installs and trains without starting the servers.
If script execution is blocked, run `powershell -ExecutionPolicy Bypass -File .\run.ps1`.

### Manual steps

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

python -m ml.preprocess          # optional: just the cleaning step, writes data/processed/
python -m ml.train               # clean + train + evaluate + write artifacts/
python -m ml.evaluate            # optional: re-evaluate the saved model

uvicorn backend.main:app --reload --port 8000

# in a second terminal
cd frontend
npm install
npm run dev                      # http://localhost:5173
```

Set `VITE_API_URL` (see `frontend/.env.example`) if the API isn't at `http://localhost:8000`.

### Tests

```powershell
python -m pytest
```

This runs 50 tests:

- **Preprocessing:** sqft range parsing, unit conversion, BHK parsing, location grouping and each outlier rule.
- **API (FastAPI `TestClient`):** `/health`, `/locations`, `/metrics`, a valid `/predict`, unknown locality → `other`, monotonicity, six invalid payloads → 422, and static plots.

### EDA notebook

`notebooks/eda.ipynb` is committed with its outputs. To re-execute it:

```powershell
jupyter nbconvert --to notebook --execute --inplace notebooks/eda.ipynb
```

### Docker (optional)

```powershell
python -m ml.train          # the backend image bundles artifacts/, so train first
docker compose up --build   # frontend http://localhost:8080 · API http://localhost:8000
```

- `backend/Dockerfile`: python:3.12-slim with `requirements-api.txt` and the trained artifacts.
- `frontend/Dockerfile`: a multi-stage Vite build served by nginx.
- `docker-compose.yml`: runs both and sets `CORS_ORIGINS` for the :8080 frontend.

> These files were written but **not built or tested**, because Docker isn't installed on the development machine.

## API

Interactive docs are at `http://localhost:8000/docs`.

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | `{status, model_name, trained_at}` |
| GET | `/locations` | Sorted list of known localities, with `other` last |
| GET | `/metrics` | Contents of `metrics.json` |
| POST | `/predict` | Predicted price with range |
| GET | `/static/<file>.png` | Model plots from `artifacts/plots/` |

**Validation:** `total_sqft` 300–30,000, `bhk` 1–10, `bath` 1–10, and `bath ≤ bhk + 2`. Violations
return **422**. Locality matching ignores case. An unknown locality maps to `other` and the response
includes a `warning`.

```bash
curl -X POST http://localhost:8000/predict -H "Content-Type: application/json" -d "{\"location\": \"Whitefield\", \"total_sqft\": 1200, \"bhk\": 2, \"bath\": 2}"
```

```json
{
  "predicted_price_lakhs": 75.55,
  "low_lakhs": 60.29,
  "high_lakhs": 95.64,
  "price_per_sqft": 6296.18,
  "model_name": "XGBoost (tuned)",
  "location_used": "Whitefield",
  "warning": null
}
```

## Project structure

```
.
├── data/
│   ├── raw/                     # input CSV (gitignored)
│   └── processed/               # bengaluru_clean.csv
├── notebooks/eda.ipynb          # EDA (imports from ml/), executed with outputs
├── ml/
│   ├── config.py                # paths, seed 42, constants, unit factors, API bounds
│   ├── preprocess.py            # pure cleaning / feature functions + run_pipeline
│   ├── train.py                 # CV comparison, tuning, selection, interval, save
│   └── evaluate.py              # metrics + plots
├── artifacts/                   # model.joblib, locations.json, metrics.json, plots/
├── backend/
│   ├── main.py                  # FastAPI app (lifespan load, CORS, static plots)
│   ├── schemas.py               # Pydantic request/response models
│   ├── model_service.py         # loads artifacts once, resolves location, predicts
│   └── Dockerfile
├── frontend/                    # React + Vite (src/App.jsx, components/, styles.css), Dockerfile, nginx.conf
├── tests/                       # test_preprocess.py, test_api.py
├── requirements.txt             # full dev env (pinned)
├── requirements-api.txt         # runtime subset for the API image
├── run.ps1                      # one-command Windows startup
├── docker-compose.yml
├── pytest.ini
└── README.md
```

## Limitations

- **Old listing prices, not sale prices.** The data is a snapshot of asking prices from several years ago.
  It doesn't reflect today's market or the final negotiated price.
- **Only four features.** Floor, age, amenities, carpet vs super built-up area (`area_type` was dropped),
  and ready vs under-construction are all ignored. Two homes with identical inputs can differ a lot in price.
- **Cleaning runs before the split.** The outlier rules and the "≤ 10 listings → `other`" grouping
  are applied to the whole dataset (as specified), so the test set is also "clean". Errors on unusual real
  listings will be larger than the test metrics suggest. All *fitted* transforms (encoding, scaling) are
  fitted on training folds only.
- **`other` is a catch-all.** About 16 % of clean rows fall into it, so predictions for unlisted localities
  are much less specific.
- **The range is global.** One pair of ratios applies to every home. It doesn't widen for rare
  configurations such as 6+ BHK or very large plots, where the model has seen little data.
- **No time dimension, no geography.** The model doesn't know which localities are near each other.

## Future work

- **SHAP** explanations per prediction, e.g. "location adds ₹ 12 L vs city average".
- **Docker:** verify the provided compose setup in CI and publish images.
- **Map view:** geocode localities and show predicted ₹/sqft on a map. Distance-based features could also replace `other`.
- Conditional prediction intervals, such as quantile regression or conformal prediction per segment.
- Add `area_type` and availability back as features and measure the gain.
- Retrain on fresher data and track drift.
