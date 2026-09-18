"""
generate_dataset.py
--------------------
Generates a synthetic-but-realistic agricultural dataset for crop
recommendation, matching the schema used by the well-known Kaggle
"Crop Recommendation Dataset":

    N, P, K, temperature, humidity, ph, rainfall, label

Each crop's samples are drawn from a normal distribution centered on
published agronomic requirement ranges (FAO / ICAR guideline midpoints),
so the resulting classifier learns sensible, explainable decision
boundaries. This lets the project run end-to-end without depending on
an external dataset download.

Run:
    python generate_dataset.py
Produces:
    Crop_recommendation.csv  (in this same folder)
"""

import numpy as np
import pandas as pd

np.random.seed(42)

# Approximate agronomic requirement ranges (mean, std) per crop for each feature.
# Ranges are informed by general agronomy references and are meant to produce
# a realistic, separable dataset for demonstration / production use.
CROP_PROFILES = {
    "rice":        dict(N=(80, 15), P=(45, 10), K=(40, 10), temperature=(25, 3), humidity=(82, 5), ph=(6.2, 0.5), rainfall=(230, 40)),
    "maize":       dict(N=(80, 15), P=(40, 10), K=(20, 8),  temperature=(23, 4), humidity=(63, 8), ph=(6.3, 0.5), rainfall=(85, 20)),
    "chickpea":    dict(N=(40, 10), P=(65, 10), K=(80, 10), temperature=(19, 3), humidity=(16, 5), ph=(7.3, 0.4), rainfall=(80, 15)),
    "kidneybeans": dict(N=(20, 8),  P=(65, 10), K=(20, 8),  temperature=(18, 3), humidity=(21, 5), ph=(5.7, 0.4), rainfall=(105, 20)),
    "pigeonpeas":  dict(N=(20, 8),  P=(65, 10), K=(20, 8),  temperature=(27, 4), humidity=(48, 8), ph=(5.8, 0.5), rainfall=(150, 30)),
    "mothbeans":   dict(N=(20, 8),  P=(45, 10), K=(20, 8),  temperature=(28, 3), humidity=(53, 8), ph=(6.8, 0.5), rainfall=(50, 12)),
    "mungbean":    dict(N=(20, 8),  P=(45, 10), K=(20, 8),  temperature=(28, 3), humidity=(85, 5), ph=(6.7, 0.4), rainfall=(48, 10)),
    "blackgram":   dict(N=(40, 10), P=(65, 10), K=(20, 8),  temperature=(29, 3), humidity=(65, 8), ph=(7.1, 0.4), rainfall=(68, 14)),
    "lentil":      dict(N=(20, 8),  P=(65, 10), K=(20, 8),  temperature=(24, 3), humidity=(65, 8), ph=(6.9, 0.4), rainfall=(45, 10)),
    "pomegranate": dict(N=(19, 8),  P=(19, 8),  K=(40, 10), temperature=(21, 4), humidity=(90, 5), ph=(6.4, 0.4), rainfall=(107, 20)),
    "banana":      dict(N=(100, 15),P=(82, 10), K=(50, 10), temperature=(27, 3), humidity=(80, 5), ph=(6.0, 0.4), rainfall=(100, 20)),
    "mango":       dict(N=(20, 8),  P=(27, 10), K=(30, 10), temperature=(31, 3), humidity=(50, 8), ph=(5.8, 0.4), rainfall=(95, 20)),
    "grapes":      dict(N=(23, 8),  P=(132, 10),K=(200, 10),temperature=(24, 3), humidity=(82, 5), ph=(6.0, 0.4), rainfall=(70, 15)),
    "watermelon":  dict(N=(100, 15),P=(17, 8),  K=(50, 10), temperature=(25, 3), humidity=(85, 5), ph=(6.5, 0.4), rainfall=(50, 12)),
    "muskmelon":   dict(N=(100, 15),P=(17, 8),  K=(50, 10), temperature=(28, 3), humidity=(92, 5), ph=(6.4, 0.4), rainfall=(25, 8)),
    "apple":       dict(N=(20, 8),  P=(130, 10),K=(200, 10),temperature=(22, 3), humidity=(92, 5), ph=(5.9, 0.4), rainfall=(112, 20)),
    "orange":      dict(N=(19, 8),  P=(16, 8),  K=(10, 5),  temperature=(22, 3), humidity=(92, 5), ph=(7.0, 0.4), rainfall=(110, 20)),
    "papaya":      dict(N=(50, 12), P=(59, 10), K=(50, 10), temperature=(33, 3), humidity=(92, 5), ph=(6.7, 0.4), rainfall=(142, 25)),
    "coconut":     dict(N=(22, 8),  P=(16, 8),  K=(30, 10), temperature=(27, 2), humidity=(94, 3), ph=(5.9, 0.4), rainfall=(175, 25)),
    "cotton":      dict(N=(120, 15),P=(46, 10), K=(20, 8),  temperature=(24, 3), humidity=(80, 6), ph=(6.9, 0.4), rainfall=(80, 15)),
    "jute":        dict(N=(80, 15), P=(47, 10), K=(40, 10), temperature=(25, 2), humidity=(80, 5), ph=(6.7, 0.4), rainfall=(175, 25)),
    "coffee":      dict(N=(101, 15),P=(28, 10), K=(30, 10), temperature=(25, 3), humidity=(58, 8), ph=(6.8, 0.4), rainfall=(150, 25)),
}

SAMPLES_PER_CROP = 100
FEATURES = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]

rows = []
for crop, profile in CROP_PROFILES.items():
    for _ in range(SAMPLES_PER_CROP):
        row = {}
        for feat in FEATURES:
            mean, std = profile[feat]
            val = np.random.normal(mean, std)
            # Clip to sensible physical bounds
            if feat in ("N", "P", "K", "rainfall"):
                val = max(val, 0)
            if feat == "humidity":
                val = min(max(val, 0), 100)
            if feat == "ph":
                val = min(max(val, 3.5), 9.5)
            if feat == "temperature":
                val = max(val, 0)
            row[feat] = round(val, 2)
        row["label"] = crop
        rows.append(row)

df = pd.DataFrame(rows)
df = df.sample(frac=1, random_state=42).reset_index(drop=True)  # shuffle

out_path = __file__.replace("generate_dataset.py", "Crop_recommendation.csv")
df.to_csv(out_path, index=False)
print(f"Dataset written to {out_path}")
print(df.shape)
print(df["label"].value_counts())
