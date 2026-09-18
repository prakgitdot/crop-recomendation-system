"""
train_model.py
---------------
Trains the crop recommendation model and saves it to disk.

Usage:
    python train_model.py

Produces (in this same "model/" folder):
    crop_model.pkl     -> trained RandomForestClassifier
    scaler.pkl         -> fitted StandardScaler used to preprocess inputs
    label_encoder.pkl  -> fitted LabelEncoder mapping crop name <-> class index

All paths are relative to this script's own location (os.path.dirname(__file__)),
so this works identically on Windows, Linux, and any cloud host — no hardcoded
local paths.
"""

import os
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report

# ---------------------------------------------------------------------------
# Paths (relative to this file — works on any OS / any cloud host)
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "..", "data", "Crop_recommendation.csv")

MODEL_PATH = os.path.join(BASE_DIR, "crop_model.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "scaler.pkl")
ENCODER_PATH = os.path.join(BASE_DIR, "label_encoder.pkl")

FEATURES = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]
TARGET = "label"


def main():
    print(f"Loading dataset from: {DATA_PATH}")
    df = pd.read_csv(DATA_PATH)

    X = df[FEATURES]
    y_raw = df[TARGET]

    # Encode crop names -> integers
    encoder = LabelEncoder()
    y = encoder.fit_transform(y_raw)

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    print(f"\nTest Accuracy: {acc * 100:.2f}%\n")
    print(classification_report(y_test, preds, target_names=encoder.classes_))

    # Save artifacts
    joblib.dump(model, MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    joblib.dump(encoder, ENCODER_PATH)

    print(f"\nSaved model      -> {MODEL_PATH}")
    print(f"Saved scaler     -> {SCALER_PATH}")
    print(f"Saved encoder    -> {ENCODER_PATH}")


if __name__ == "__main__":
    main()
