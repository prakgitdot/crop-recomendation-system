"""
app.py
------
Crop Management System — Flask entry point.

Five tools live behind this one app, all sharing the same page, the same
stylesheet and the same deployment:

  1. Crop Recommendation  — soil nutrients + weather        -> best crop
  2. Crop Prediction      — state + district + season       -> crops grown there
  3. Fertilizer Recommendation — soil/crop conditions       -> fertilizer
  4. Rainfall Prediction  — subdivision + period + year     -> expected rainfall
  5. Yield Prediction     — state/district/season/crop/area -> expected production

This is the production entry point used both locally (`python app.py`)
and on the cloud (via `gunicorn app:app`, see Procfile).

Design notes for cloud-readiness:
- All file paths are built from BASE_DIR (this file's own folder) using
  os.path.join, so there is NO dependency on local Windows paths like
  C:\\Users\\... The app runs identically on Windows, Linux, macOS, or
  any cloud host's container filesystem.
- The port is read from the PORT environment variable (falls back to
  5000 locally). Cloud platforms like Render inject PORT automatically.
- Every model artifact is loaded once at startup and kept in memory — no
  per-request disk reads, no per-request retraining, no subprocess calls.
- A missing artifact disables only its own feature. The other four keep
  working, and /health reports exactly which ones are unavailable.
- CORS is enabled so a separately hosted frontend can call the API
  endpoints without cross-origin errors.
"""

import os
import math
import joblib
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS

# ---------------------------------------------------------------------------
# Paths — all relative to this file's own directory. No hardcoded local
# paths anywhere, so this works the same on any machine or cloud host.
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "model")


def _p(name):
    return os.path.join(MODEL_DIR, name)


FEATURE_ORDER = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]

app = Flask(__name__)
CORS(app)  # allow cross-origin requests to the API endpoints

# ---------------------------------------------------------------------------
# Load model artifacts ONCE at startup.
#
# Each feature is loaded independently: if the fertilizer model is missing,
# crop recommendation still works. ARTIFACTS holds whatever loaded, and
# LOAD_ERRORS records why anything did not, so both /health and the error
# returned to the user can say something specific instead of "server error".
# ---------------------------------------------------------------------------
ARTIFACTS = {}
LOAD_ERRORS = {}

_SPEC = {
    "crop_recommendation": ["crop_model.pkl", "scaler.pkl", "label_encoder.pkl"],
    "fertilizer": ["fertilizer_model.pkl", "fertilizer_meta.pkl"],
    "yield": ["yield_model.pkl", "yield_meta.pkl"],
    "crop_prediction": ["crop_prediction.pkl"],
    "rainfall": ["rainfall_model.pkl"],
}

_BUILD_HINT = {
    "crop_recommendation": "python model/train_model.py",
    "fertilizer": "python model/train_fertilizer.py",
    "yield": "python model/train_yield.py",
    "crop_prediction": "python model/build_crop_prediction.py",
    "rainfall": "python model/build_rainfall.py",
}

for _feature, _files in _SPEC.items():
    try:
        _loaded = []
        for _fname in _files:
            _path = _p(_fname)
            if not os.path.exists(_path):
                raise FileNotFoundError(
                    f"{_fname} not found in model/. "
                    f"Run '{_BUILD_HINT[_feature]}' to generate it."
                )
            _loaded.append(joblib.load(_path))
        ARTIFACTS[_feature] = _loaded
        print(f"[startup] {_feature}: loaded")
    except Exception as _exc:  # noqa: BLE001 - surface any load failure
        LOAD_ERRORS[_feature] = str(_exc)
        print(f"[startup] {_feature}: UNAVAILABLE - {_exc}")


def unavailable(feature):
    """Standard 503 payload for a feature whose artifacts did not load."""
    return jsonify({
        "success": False,
        "error": f"{feature.replace('_', ' ').title()} is unavailable on this server: "
                 f"{LOAD_ERRORS.get(feature, 'artifact not loaded')}",
    }), 503


def read_payload():
    """Accept both a JSON body (fetch/API clients) and a classic form POST."""
    return request.get_json(silent=True) or request.form


def require_fields(data, fields):
    """Return (values, error_response). Exactly one of the two is None."""
    out = {}
    for f in fields:
        if f not in data or data[f] in (None, ""):
            return None, (jsonify({"success": False,
                                   "error": f"Missing required field: {f}"}), 400)
        out[f] = data[f]
    return out, None


def as_float(values, fields):
    try:
        return {f: float(values[f]) for f in fields}, None
    except (TypeError, ValueError) as exc:
        return None, (jsonify({"success": False,
                               "error": f"Invalid numeric value: {exc}"}), 400)


# ---------------------------------------------------------------------------
# Pages and health
# ---------------------------------------------------------------------------
@app.route("/")
def home():
    return render_template("index.html")


@app.route("/health")
def health():
    """Per-feature health check — useful for verifying a cloud deployment."""
    features = {name: ("ok" if name in ARTIFACTS else "unavailable") for name in _SPEC}
    all_ok = not LOAD_ERRORS
    return jsonify({
        "status": "ok" if all_ok else "degraded",
        "features": features,
        "errors": LOAD_ERRORS,
    }), (200 if all_ok else 503)


@app.route("/api/options")
def options():
    """
    Everything the dropdowns need, in one request, fetched once when the page
    loads. Keeping this server-side means the option lists always match the
    data the models were actually trained on — there is no hand-maintained
    list of districts in the frontend that can drift out of sync.
    """
    out = {"available": {name: (name in ARTIFACTS) for name in _SPEC}}

    if "crop_prediction" in ARTIFACTS:
        cp = ARTIFACTS["crop_prediction"][0]
        out["crop_prediction"] = {
            "states": cp["states"],
            "districts_by_state": cp["districts_by_state"],
            "seasons_by_district": cp["seasons_by_district"],
            "seasons": cp["seasons"],
        }

    if "yield" in ARTIFACTS:
        ym = ARTIFACTS["yield"][1]
        out["yield"] = {
            "states": ym["states"],
            "districts_by_state": ym["districts_by_state"],
            "crops_by_state": ym["crops_by_state"],
            "seasons": ym["seasons"],
            "year_min": ym["year_min"],
            "year_max": ym["year_max"],
            "test_r2": ym["test_r2"],
        }

    if "fertilizer" in ARTIFACTS:
        fm = ARTIFACTS["fertilizer"][1]
        out["fertilizer"] = {
            "soil_types": fm["soil_types"],
            "crop_types": fm["crop_types"],
            "fertilizers": fm["fertilizers"],
        }

    if "rainfall" in ARTIFACTS:
        rf = ARTIFACTS["rainfall"][0]
        out["rainfall"] = {
            "subdivisions": rf["subdivisions"],
            "periods": rf["periods"],
            "year_min": rf["year_min"],
            "year_max": rf["year_max"],
        }

    return jsonify(out)


# ---------------------------------------------------------------------------
# 1. Crop Recommendation — N, P, K, temperature, humidity, pH, rainfall
# ---------------------------------------------------------------------------
@app.route("/predict", methods=["POST"])
def predict():
    if "crop_recommendation" not in ARTIFACTS:
        return unavailable("crop_recommendation")

    model, scaler, label_encoder = ARTIFACTS["crop_recommendation"]
    data = read_payload()

    raw, err = require_fields(data, FEATURE_ORDER)
    if err:
        return err
    nums, err = as_float(raw, FEATURE_ORDER)
    if err:
        return err

    try:
        X = pd.DataFrame([[nums[f] for f in FEATURE_ORDER]], columns=FEATURE_ORDER)
        X_scaled = scaler.transform(X)
        pred_idx = model.predict(X_scaled)[0]
        crop_name = label_encoder.inverse_transform([pred_idx])[0]

        # Top-3 probabilities for a nicer, more informative UI
        probs = model.predict_proba(X_scaled)[0]
        top3_idx = np.argsort(probs)[::-1][:3]
        top3 = [
            {"crop": label_encoder.inverse_transform([i])[0],
             "confidence": round(float(probs[i]) * 100, 2)}
            for i in top3_idx
        ]

        return jsonify({"success": True, "recommended_crop": crop_name, "top_3": top3})
    except Exception as exc:  # noqa: BLE001
        return jsonify({"success": False, "error": f"Prediction failed: {exc}"}), 500


@app.route("/api/crop-recommendation", methods=["POST"])
def crop_recommendation_alias():
    """Alias so every feature has a matching /api/... route."""
    return predict()


# ---------------------------------------------------------------------------
# 2. Crop Prediction — which crops are actually grown in a place and season
# ---------------------------------------------------------------------------
@app.route("/api/crop-prediction", methods=["POST"])
def crop_prediction():
    if "crop_prediction" not in ARTIFACTS:
        return unavailable("crop_prediction")

    cp = ARTIFACTS["crop_prediction"][0]
    data = read_payload()

    raw, err = require_fields(data, ["state", "district", "season"])
    if err:
        return err

    state = str(raw["state"]).strip()
    district = str(raw["district"]).strip()
    season = str(raw["season"]).strip()

    matches = cp["lookup"].get(f"{state}|{district}|{season}")

    if not matches:
        # Give a useful message rather than an empty result: say which seasons
        # this district does have records for.
        available = cp["seasons_by_district"].get(f"{state}|{district}")
        if available:
            return jsonify({
                "success": False,
                "error": f"No records for {district} in the {season} season. "
                         f"Seasons on record for this district: {', '.join(available)}.",
            }), 404
        return jsonify({
            "success": False,
            "error": f"No records found for {district}, {state}.",
        }), 404

    return jsonify({
        "success": True,
        "state": state,
        "district": district,
        "season": season,
        "recommended_crop": matches[0]["crop"],
        "crops": matches,
    })


# ---------------------------------------------------------------------------
# 3. Fertilizer Recommendation
# ---------------------------------------------------------------------------
def nutrient_note(nums):
    """
    A short, plain-language reading of the nutrient balance to sit beside the
    fertilizer name. Rules of thumb only — it explains why the suggestion
    leans the way it does; it is not a substitute for a soil test.
    """
    parts = []
    if nums["nitrogen"] < 12:
        parts.append("nitrogen is low")
    elif nums["nitrogen"] > 30:
        parts.append("nitrogen is already high")
    if nums["phosphorous"] < 10:
        parts.append("phosphorous is low")
    elif nums["phosphorous"] > 30:
        parts.append("phosphorous is already high")
    if nums["potassium"] < 5:
        parts.append("potassium is low")
    elif nums["potassium"] > 15:
        parts.append("potassium is already high")
    if not parts:
        return "Nutrient levels look reasonably balanced for this crop."
    return "Reading of the inputs: " + ", ".join(parts) + "."


@app.route("/api/fertilizer", methods=["POST"])
def fertilizer():
    if "fertilizer" not in ARTIFACTS:
        return unavailable("fertilizer")

    model, meta = ARTIFACTS["fertilizer"]
    data = read_payload()

    numeric = ["temperature", "humidity", "moisture",
               "nitrogen", "potassium", "phosphorous"]
    categorical = ["soil_type", "crop_type"]

    raw, err = require_fields(data, numeric + categorical)
    if err:
        return err
    nums, err = as_float(raw, numeric)
    if err:
        return err

    soil = str(raw["soil_type"]).strip()
    crop = str(raw["crop_type"]).strip()

    if soil not in meta["soil_types"]:
        return jsonify({"success": False,
                        "error": f"Unknown soil type '{soil}'. Choose one of: "
                                 f"{', '.join(meta['soil_types'])}."}), 400
    if crop not in meta["crop_types"]:
        return jsonify({"success": False,
                        "error": f"Unknown crop type '{crop}'. Choose one of: "
                                 f"{', '.join(meta['crop_types'])}."}), 400

    try:
        soil_enc = int(meta["le_soil"].transform([soil])[0])
        crop_enc = int(meta["le_crop"].transform([crop])[0])

        # Column order must match model/train_fertilizer.py exactly; it is
        # stored in the metadata rather than repeated here.
        row = {
            "Temparature": nums["temperature"],
            "Humidity": nums["humidity"],
            "Soil Moisture": nums["moisture"],
            "Soil Type": soil_enc,
            "Crop Type": crop_enc,
            "Nitrogen": nums["nitrogen"],
            "Potassium": nums["potassium"],
            "Phosphorous": nums["phosphorous"],
        }
        X = pd.DataFrame([[row[c] for c in meta["features"]]], columns=meta["features"])

        pred = model.predict(X)[0]

        alternatives = []
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(X)[0]
            order = np.argsort(probs)[::-1][:3]
            alternatives = [
                {"fertilizer": str(model.classes_[i]),
                 "confidence": round(float(probs[i]) * 100, 2)}
                for i in order if probs[i] > 0
            ]

        return jsonify({
            "success": True,
            "fertilizer": str(pred),
            "top_3": alternatives,
            "note": nutrient_note(nums),
        })
    except Exception as exc:  # noqa: BLE001
        return jsonify({"success": False, "error": f"Prediction failed: {exc}"}), 500


# ---------------------------------------------------------------------------
# 4. Rainfall Prediction
# ---------------------------------------------------------------------------
@app.route("/api/rainfall", methods=["POST"])
def rainfall():
    if "rainfall" not in ARTIFACTS:
        return unavailable("rainfall")

    rf = ARTIFACTS["rainfall"][0]
    data = read_payload()

    raw, err = require_fields(data, ["subdivision", "period"])
    if err:
        return err

    subdivision = str(raw["subdivision"]).strip()
    period = str(raw["period"]).strip()
    year_in = data.get("year")

    stats = rf["stats"].get(subdivision)
    if stats is None:
        return jsonify({"success": False,
                        "error": f"Unknown subdivision '{subdivision}'."}), 404

    entry = stats.get(period)
    if entry is None:
        return jsonify({"success": False,
                        "error": f"No data for period '{period}' in {subdivision}."}), 404

    try:
        year = int(year_in) if year_in not in (None, "") else rf["year_max"] + 1
    except (TypeError, ValueError):
        return jsonify({"success": False, "error": f"Invalid year: {year_in}"}), 400

    # Trend-adjusted estimate: the fitted line evaluated at the requested year,
    # floored at zero because negative rainfall is meaningless.
    trend = entry["slope"] * year + entry["intercept"]
    predicted = max(0.0, float(trend))

    # Years outside the record are extrapolation; say so rather than presenting
    # the number as if it were as solid as an in-range estimate.
    extrapolated = year > rf["year_max"] or year < rf["year_min"]

    low = max(0.0, entry["mean"] - entry["std"])
    high = entry["mean"] + entry["std"]

    return jsonify({
        "success": True,
        "subdivision": subdivision,
        "period": period,
        "year": year,
        "predicted_rainfall_mm": round(predicted, 1),
        "historical_average_mm": entry["mean"],
        "typical_range_mm": [round(low, 1), round(high, 1)],
        "recorded_min_mm": entry["min"],
        "recorded_max_mm": entry["max"],
        "trend_mm_per_decade": round(entry["slope"] * 10, 1),
        "years_of_data": entry["n_years"],
        "extrapolated": extrapolated,
    })


# ---------------------------------------------------------------------------
# 5. Yield / Production Prediction
# ---------------------------------------------------------------------------
@app.route("/api/yield", methods=["POST"])
def yield_prediction():
    if "yield" not in ARTIFACTS:
        return unavailable("yield")

    model, meta = ARTIFACTS["yield"]
    data = read_payload()

    raw, err = require_fields(data, ["state", "district", "season", "crop", "area"])
    if err:
        return err

    state = str(raw["state"]).strip()
    district = str(raw["district"]).strip()
    season = str(raw["season"]).strip()
    crop = str(raw["crop"]).strip()

    try:
        area = float(raw["area"])
    except (TypeError, ValueError):
        return jsonify({"success": False, "error": f"Invalid area: {raw['area']}"}), 400
    if area <= 0:
        return jsonify({"success": False, "error": "Area must be greater than zero."}), 400

    year_in = data.get("year")
    try:
        year = int(year_in) if year_in not in (None, "") else meta["year_max"]
    except (TypeError, ValueError):
        return jsonify({"success": False, "error": f"Invalid year: {year_in}"}), 400

    maps = meta["maps"]
    for label, value, col in (("state", state, "State_Name"),
                              ("district", district, "District_Name"),
                              ("season", season, "Season"),
                              ("crop", crop, "Crop")):
        if value not in maps[col]:
            return jsonify({
                "success": False,
                "error": f"Unknown {label} '{value}' — it does not appear in the training data.",
            }), 400

    try:
        prior = meta["prior"]
        row = {
            "State_Name": maps["State_Name"][state],
            "Season": maps["Season"][season],
            "Crop": maps["Crop"][crop],
            "District_code": maps["District_Name"][district],
            "Crop_Year": year,
            "Area": area,
            "log_Area": math.log1p(area),
            "te_district": meta["enc_district"].get(district, prior),
            "te_district_crop": meta["enc_district_crop"].get(f"{district}|{crop}", prior),
        }
        X = pd.DataFrame([[row[c] for c in meta["columns"]]], columns=meta["columns"])

        predicted = max(0.0, float(np.expm1(model.predict(X)[0])))

        # Historical production-per-unit-area for this exact combination, shown
        # beside the model output so the number can be sanity-checked.
        hist = meta["hist_yield"].get(f"{state}|{district}|{season}|{crop}")
        historical = None
        if hist:
            ratio, n = hist
            historical = {
                "yield_per_unit_area": round(ratio, 3),
                "estimated_production": round(ratio * area, 2),
                "records": n,
            }

        return jsonify({
            "success": True,
            "state": state,
            "district": district,
            "season": season,
            "crop": crop,
            "year": year,
            "area": area,
            "predicted_production": round(predicted, 2),
            "predicted_yield_per_unit_area": round(predicted / area, 3),
            "historical": historical,
            "model_r2": meta["test_r2"],
        })
    except Exception as exc:  # noqa: BLE001
        return jsonify({"success": False, "error": f"Prediction failed: {exc}"}), 500


# ---------------------------------------------------------------------------
# Local dev entry point.
# In production, gunicorn imports the `app` object directly (see Procfile)
# and this block is never executed — but it's kept so `python app.py`
# still works fine for local testing.
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    debug_mode = os.environ.get("FLASK_DEBUG", "false").lower() == "true"
    app.run(host="0.0.0.0", port=port, debug=debug_mode)
