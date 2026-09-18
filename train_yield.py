"""
train_yield.py
--------------
Trains the Yield / Production Prediction model.

Input : data/crop_production_india.csv
        (State_Name, District_Name, Crop_Year, Season, Crop, Area, Production)

Output: model/yield_model.pkl   -> HistGradientBoostingRegressor
        model/yield_meta.pkl    -> ordinal category maps + option lists used
                                   by the dropdowns in the UI

Why HistGradientBoostingRegressor instead of the RandomForest + one-hot
combination: the dataset has ~246k rows, 646 districts and 124 crops, so
one-hot encoding explodes the feature count and the pickled forest runs to
hundreds of megabytes. HistGradientBoosting handles the categories natively,
trains in seconds and pickles down to a couple of megabytes, which matters on
a free cloud tier with a small slug size.

State, Season and Crop are passed as native categoricals. District has 646
levels, above the 255 that native categorical splitting allows, so it is fed
in two ways instead: as an ordinal code, and as a smoothed target encoding
(the district's typical production-per-hectare). The target encoding is fitted
on the training split only and stored in the metadata so the API can apply the
identical transform at request time.

The target is log1p(Production) because production spans several orders of
magnitude (a few tonnes to millions); training on the log makes the errors
comparable across small and large districts. Predictions are converted back
with expm1.

Run:    python model/train_yield.py
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import train_test_split

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV = os.path.join(BASE_DIR, "data", "crop_production_india.csv")
OUT_MODEL = os.path.join(BASE_DIR, "model", "yield_model.pkl")
OUT_META = os.path.join(BASE_DIR, "model", "yield_meta.pkl")

GROUP_COLS = ["State_Name", "District_Name", "Season", "Crop"]
NATIVE_CAT_COLS = ["State_Name", "Season", "Crop"]   # <= 255 levels each
SMOOTHING = 10.0  # prior weight for the target encodings


def clean(df):
    """Trim the padded season strings and drop rows that cannot be learned from."""
    for c in GROUP_COLS:
        df[c] = df[c].astype(str).str.strip()
    df = df.dropna(subset=["Area", "Production"])
    df = df[(df["Area"] > 0) & (df["Production"] >= 0)]
    return df


def target_encode(keys, values, prior, smoothing=SMOOTHING):
    """Smoothed mean of `values` per key, pulled toward `prior` for rare keys."""
    agg = pd.DataFrame({"k": keys, "v": values}).groupby("k", observed=True)["v"].agg(["mean", "count"])
    enc = (agg["mean"] * agg["count"] + prior * smoothing) / (agg["count"] + smoothing)
    return enc.to_dict()


def main():
    df = clean(pd.read_csv(CSV))

    # Ordinal codes for every categorical column. The mapping is saved so the
    # API can encode a user's selection exactly the same way at request time.
    maps = {}
    for c in GROUP_COLS:
        maps[c] = {v: i for i, v in enumerate(sorted(df[c].unique()))}

    # log yield ratio = log1p(production per unit area) — the quantity the
    # target encodings summarise.
    df["_ratio"] = np.log1p(df["Production"] / df["Area"])

    tr_idx, te_idx = train_test_split(df.index, test_size=0.2, random_state=42)
    train = df.loc[tr_idx]

    prior = float(train["_ratio"].mean())
    enc_district = target_encode(train["District_Name"], train["_ratio"], prior)
    enc_district_crop = target_encode(
        train["District_Name"] + "|" + train["Crop"], train["_ratio"], prior
    )

    def build_features(frame):
        X = pd.DataFrame(index=frame.index)
        for c in NATIVE_CAT_COLS:
            X[c] = frame[c].map(maps[c]).astype("int32")
        X["District_code"] = frame["District_Name"].map(maps["District_Name"]).astype("int32")
        X["Crop_Year"] = frame["Crop_Year"].astype("int32")
        X["Area"] = frame["Area"].astype("float64")
        X["log_Area"] = np.log1p(frame["Area"].astype("float64"))
        X["te_district"] = frame["District_Name"].map(enc_district).fillna(prior)
        X["te_district_crop"] = (
            (frame["District_Name"] + "|" + frame["Crop"]).map(enc_district_crop).fillna(prior)
        )
        return X

    X = build_features(df)
    y = np.log1p(df["Production"].astype("float64"))

    X_tr, y_tr = X.loc[tr_idx], y.loc[tr_idx]
    X_te, y_te = X.loc[te_idx], y.loc[te_idx]

    model = HistGradientBoostingRegressor(
        max_iter=400,
        learning_rate=0.08,
        max_leaf_nodes=63,
        min_samples_leaf=10,
        l2_regularization=1.0,
        categorical_features=[X.columns.get_loc(c) for c in NATIVE_CAT_COLS],
        random_state=42,
    )
    model.fit(X_tr, y_tr)

    r2 = model.score(X_te, y_te)

    # A plain historical yield table (production per unit area) kept alongside
    # the model: shown next to the prediction as a sanity figure, and used as a
    # fallback for combinations the model has never seen.
    hist = (
        df.assign(ratio=df["Production"] / df["Area"])
          .groupby(GROUP_COLS, observed=True)["ratio"]
          .agg(["mean", "count"])
          .reset_index()
    )
    hist_lookup = {
        "|".join([r.State_Name, r.District_Name, r.Season, r.Crop]):
            [round(float(r.mean), 4), int(r.count)]
        for r in hist.itertuples()
    }

    # Cascading dropdown options: state -> districts, state -> crops
    districts = (
        df.groupby("State_Name", observed=True)["District_Name"]
          .apply(lambda s: sorted(s.unique().tolist())).to_dict()
    )
    crops_by_state = (
        df.groupby("State_Name", observed=True)["Crop"]
          .apply(lambda s: sorted(s.unique().tolist())).to_dict()
    )

    joblib.dump(model, OUT_MODEL, compress=3)
    joblib.dump(
        {
            "columns": X.columns.tolist(),
            "native_cat_cols": NATIVE_CAT_COLS,
            "maps": maps,
            "prior": prior,
            "enc_district": enc_district,
            "enc_district_crop": enc_district_crop,
            "states": sorted(df["State_Name"].unique().tolist()),
            "districts_by_state": districts,
            "crops_by_state": crops_by_state,
            "seasons": sorted(df["Season"].unique().tolist()),
            "crops": sorted(df["Crop"].unique().tolist()),
            "year_min": int(df["Crop_Year"].min()),
            "year_max": int(df["Crop_Year"].max()),
            "hist_yield": hist_lookup,
            "test_r2": round(float(r2), 4),
        },
        OUT_META,
        compress=3,
    )

    print(f"[yield] trained on {len(df)} rows | test R² (log scale) = {r2:.4f}")
    print(f"[yield] -> {OUT_MODEL}")


if __name__ == "__main__":
    main()
