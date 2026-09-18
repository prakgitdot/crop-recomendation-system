# Crop Management System

A Flask web app with five tools for a plot of land, all on one page and one
deployment. The original crop recommendation tool is unchanged; the other four
features come from the
[crop-management-system](https://github.com/ab007shetty/crop-management-system)
project, reimplemented in Python/Flask instead of PHP so everything runs in a
single process.

| # | Tool | Inputs | Output |
|---|------|--------|--------|
| 01 | Crop Recommendation | N, P, K, temperature, humidity, pH, rainfall | Best crop, with top-3 confidences |
| 02 | Crop Prediction | State, district, season | Crops actually grown there, ranked by area share |
| 03 | Fertilizer Recommendation | Temperature, humidity, soil moisture, soil type, crop type, N, P, K | Fertilizer to apply |
| 04 | Rainfall Prediction | Subdivision, month or season, year | Expected rainfall with trend, range and extremes |
| 05 | Yield Prediction | State, district, season, crop, area, year | Expected production, next to the historical figure |

## Project structure

```
crop-recommendation-app/
├── app.py                        # Flask entry point — all five features
├── requirements.txt              # Pinned Python dependencies
├── Procfile                      # Cloud start command (gunicorn)
├── runtime.txt                   # Python version for the cloud platform
├── .gitignore
├── data/
│   ├── generate_dataset.py       # Generates the crop recommendation dataset
│   ├── Crop_recommendation.csv
│   ├── fertilizer_recommendation.csv
│   ├── rainfall_in_india_1901-2015.csv
│   └── crop_production_india.csv # District-level production records
├── model/
│   ├── train_model.py            # 01 — RandomForestClassifier
│   ├── crop_model.pkl
│   ├── scaler.pkl
│   ├── label_encoder.pkl
│   ├── build_crop_prediction.py  # 02 — district/season crop table
│   ├── crop_prediction.pkl
│   ├── train_fertilizer.py       # 03 — DecisionTreeClassifier
│   ├── fertilizer_model.pkl
│   ├── fertilizer_meta.pkl
│   ├── build_rainfall.py         # 04 — per-subdivision trend + spread
│   ├── rainfall_model.pkl
│   ├── train_yield.py            # 05 — HistGradientBoostingRegressor
│   ├── yield_model.pkl
│   └── yield_meta.pkl
├── templates/
│   └── index.html                # One page, five tabs
└── static/
    ├── style.css
    └── script.js
```

## Tech stack

- **Backend:** Flask 3, served in production by gunicorn
- **ML:** scikit-learn — random forest, decision tree, gradient boosting
- **Frontend:** plain HTML/CSS/JS, no build step

## Run locally

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
python app.py
```

Visit http://localhost:5000

## Rebuilding the models (optional)

Every artifact is committed in `model/`, so nothing needs rebuilding to run the
app. Rebuild only if you change a dataset or want to try a different algorithm:

```bash
python model/train_model.py             # 01 crop recommendation
python model/build_crop_prediction.py   # 02 crop prediction
python model/train_fertilizer.py        # 03 fertilizer
python model/build_rainfall.py          # 04 rainfall
python model/train_yield.py             # 05 yield  (~1 min, 246k rows)
```

Each script reads from `data/` and writes to `model/`, and each one is
independent — rebuilding one does not disturb the others.

## API

Every tool has a JSON endpoint. All accept either a JSON body or a form POST,
and all return `{"success": true, ...}` or `{"success": false, "error": "..."}`.

```bash
# 01 Crop recommendation  (POST /predict, also /api/crop-recommendation)
curl -X POST localhost:5000/predict -H 'Content-Type: application/json' \
  -d '{"N":90,"P":42,"K":43,"temperature":20.8,"humidity":82,"ph":6.5,"rainfall":202}'
# -> {"recommended_crop":"rice","top_3":[...]}

# 02 Crop prediction
curl -X POST localhost:5000/api/crop-prediction -H 'Content-Type: application/json' \
  -d '{"state":"Karnataka","district":"BAGALKOT","season":"Kharif"}'
# -> {"recommended_crop":"Maize","crops":[{"crop":"Maize","share":24.4,...}]}

# 03 Fertilizer
curl -X POST localhost:5000/api/fertilizer -H 'Content-Type: application/json' \
  -d '{"temperature":26,"humidity":52,"moisture":38,"soil_type":"Sandy",
       "crop_type":"Maize","nitrogen":37,"potassium":0,"phosphorous":0}'
# -> {"fertilizer":"Urea","top_3":[...],"note":"..."}

# 04 Rainfall
curl -X POST localhost:5000/api/rainfall -H 'Content-Type: application/json' \
  -d '{"subdivision":"COASTAL KARNATAKA","period":"JUL","year":2026}'
# -> {"predicted_rainfall_mm":1143.9,"historical_average_mm":1127.0,...}

# 05 Yield
curl -X POST localhost:5000/api/yield -H 'Content-Type: application/json' \
  -d '{"state":"Karnataka","district":"BAGALKOT","season":"Kharif",
       "crop":"Rice","area":197,"year":2026}'
# -> {"predicted_production":567.86,"historical":{...}}
```

Two supporting endpoints:

- `GET /api/options` — every dropdown list (states, districts, crops, seasons,
  subdivisions, soil types), read straight from the trained artifacts. The
  frontend fetches this once on load, so the options can never drift away from
  what the models were trained on.
- `GET /health` — reports each feature as `ok` or `unavailable`, plus the reason
  for anything that failed to load. Returns 200 only when all five are healthy.

## Notes on the implementation

A few places where this differs from the original PHP project, and why:

- **No subprocesses.** The PHP version shelled out to a Python script per
  request, which retrained the model every single time. Here every artifact is
  trained once offline and loaded into memory at startup, so a prediction is a
  function call rather than a process spawn plus a model fit.
- **Crop prediction is a lookup, not a tree.** The original built a decision
  tree over three categorical columns and read class counts off the matching
  leaf. With all-categorical inputs that leaf is exactly the set of rows with
  that state, district and season, so the counts are just the grouped
  frequencies — computed directly here, which is exact and turns a 12 MB CSV
  plus a pickled tree into a 236 KB table. Crops are ranked by share of cropped
  area rather than by row count, because every crop tends to appear once per
  year and a row count leaves them all tied.
- **Yield uses gradient boosting.** One-hot encoding 646 districts and 124 crops
  for a random forest produces a pickle in the hundreds of megabytes.
  `HistGradientBoostingRegressor` handles categories natively, trains on all
  246k rows in about a minute and pickles to 1.4 MB. Test R² is 0.97 on the log
  scale. It also covers all of India rather than Karnataka alone.
- **Rainfall reports a range.** The original returned the mean of a month across
  every year, ignoring the year asked about. This fits a trend line per
  subdivision and period, and returns the standard deviation and recorded
  extremes alongside it, because a single number for monsoon rainfall hides how
  much it varies year to year.
- **Degraded, not dead.** A missing artifact disables only its own tool. The
  other four keep working and the affected tab says so.

## Deployment

Unchanged from before: push to GitHub, point Render (or any similar platform) at
the repo, and it will use `Procfile` and `runtime.txt`. `PORT` is injected by the
platform; no other environment variables are required.

One thing worth checking: `data/crop_production_india.csv` is 14 MB and the
committed models come to about 22 MB, so if your host imposes a repository or
slug size limit, confirm it is above roughly 40 MB.
