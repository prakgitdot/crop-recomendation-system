"""
train_fertilizer.py
-------------------
Trains the Fertilizer Recommendation model.

Input : data/fertilizer_recommendation.csv
        (Temparature, Humidity, Soil Moisture, Soil Type, Crop Type,
         Nitrogen, Potassium, Phosphorous, Fertilizer Name)

Output: model/fertilizer_model.pkl   -> DecisionTreeClassifier
        model/fertilizer_meta.pkl    -> {soil encoder, crop encoder, column order}

Run:    python model/train_fertilizer.py
"""

import os
import joblib
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV = os.path.join(BASE_DIR, "data", "fertilizer_recommendation.csv")
OUT_MODEL = os.path.join(BASE_DIR, "model", "fertilizer_model.pkl")
OUT_META = os.path.join(BASE_DIR, "model", "fertilizer_meta.pkl")

# Column order the model is trained on — the API must build rows in this
# exact order, so it is stored alongside the model instead of duplicated.
FEATURES = [
    "Temparature", "Humidity", "Soil Moisture",
    "Soil Type", "Crop Type",
    "Nitrogen", "Potassium", "Phosphorous",
]


def main():
    df = pd.read_csv(CSV)

    le_soil = LabelEncoder().fit(df["Soil Type"])
    le_crop = LabelEncoder().fit(df["Crop Type"])

    X = df[FEATURES].copy()
    X["Soil Type"] = le_soil.transform(X["Soil Type"])
    X["Crop Type"] = le_crop.transform(X["Crop Type"])
    y = df["Fertilizer Name"]

    clf = DecisionTreeClassifier(random_state=0)
    clf.fit(X, y)

    joblib.dump(clf, OUT_MODEL)
    joblib.dump(
        {
            "features": FEATURES,
            "soil_types": list(le_soil.classes_),
            "crop_types": list(le_crop.classes_),
            "le_soil": le_soil,
            "le_crop": le_crop,
            "fertilizers": sorted(y.unique().tolist()),
        },
        OUT_META,
    )

    print(f"[fertilizer] trained on {len(df)} rows, "
          f"{y.nunique()} fertilizer classes -> {OUT_MODEL}")


if __name__ == "__main__":
    main()
