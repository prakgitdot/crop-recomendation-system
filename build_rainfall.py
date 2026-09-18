"""
build_rainfall.py
-----------------
Builds the Rainfall Prediction artifact.

Input : data/rainfall_in_india_1901-2015.csv
        (SUBDIVISION, YEAR, JAN..DEC, ANNUAL, and seasonal columns)
Output: model/rainfall_model.pkl

The original project answered a rainfall query by averaging the chosen
month's column over every year on record. That is a reasonable baseline but
it ignores the year the user asked about and gives no sense of how variable
the month is. This build keeps that long-run mean and adds two things:

- a least-squares trend line (rainfall vs. year) fitted per subdivision and
  per month, so a requested year produces a trend-adjusted figure rather than
  the same number regardless of year;
- the standard deviation, minimum and maximum, so the UI can show a likely
  range instead of a single misleadingly precise number.

Everything is precomputed here, so the API only does a dictionary lookup and
a multiplication at request time.

Run:    python model/build_rainfall.py
"""

import os
import joblib
import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV = os.path.join(BASE_DIR, "data", "rainfall_in_india_1901-2015.csv")
OUT = os.path.join(BASE_DIR, "model", "rainfall_model.pkl")

MONTHS = ["JAN", "FEB", "MAR", "APR", "MAY", "JUN",
          "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"]
PERIODS = MONTHS + ["ANNUAL", "Jan-Feb", "Mar-May", "Jun-Sep", "Oct-Dec"]


def main():
    df = pd.read_csv(CSV)
    df["SUBDIVISION"] = df["SUBDIVISION"].astype(str).str.strip()

    stats = {}
    for sub, block in df.groupby("SUBDIVISION", observed=True):
        entry = {}
        for period in PERIODS:
            series = block[["YEAR", period]].dropna()
            if len(series) < 3:
                continue
            years = series["YEAR"].to_numpy(dtype=float)
            vals = series[period].to_numpy(dtype=float)

            # Least-squares trend: rainfall ≈ slope * year + intercept
            slope, intercept = np.polyfit(years, vals, 1)

            entry[period] = {
                "mean": round(float(vals.mean()), 1),
                "std": round(float(vals.std(ddof=1)), 1),
                "min": round(float(vals.min()), 1),
                "max": round(float(vals.max()), 1),
                "slope": float(slope),
                "intercept": float(intercept),
                "n_years": int(len(series)),
            }
        stats[sub] = entry

    joblib.dump(
        {
            "stats": stats,
            "subdivisions": sorted(stats.keys()),
            "months": MONTHS,
            "periods": PERIODS,
            "year_min": int(df["YEAR"].min()),
            "year_max": int(df["YEAR"].max()),
        },
        OUT,
        compress=3,
    )

    print(f"[rainfall] {len(stats)} subdivisions, {len(PERIODS)} periods each -> {OUT}")


if __name__ == "__main__":
    main()
