"""
build_crop_prediction.py
------------------------
Builds the Crop Prediction artifact: given State + District + Season, which
crops are actually grown there, and in what proportion.

Input : data/crop_production_india.csv
Output: model/crop_prediction.pkl

The original project trained a hand-written decision tree over
(State, District, Season) and then read the class counts off the matching
leaf, printing them as percentages. Because all three inputs are categorical
and the tree splits on all of them, the leaf for a given combination is
exactly the set of rows with that State, District and Season — so the counts
it returns are just the grouped crop frequencies. This script computes those
groupings directly, which is exact rather than approximate and reduces a 12 MB
CSV plus a 380 KB pickled tree to a small lookup table that loads instantly at
startup.

One change on top of that: crops are ranked by the share of cropped area they
occupy in the district, not by how many rows mention them. Every crop grown in
a district tends to appear once per year, so a row count puts them all on an
equal footing and the "top" crop ends up being whichever one ties first
alphabetically. Area share separates a staple from a token half-hectare.

Run:    python model/build_crop_prediction.py
"""

import os
import joblib
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV = os.path.join(BASE_DIR, "data", "crop_production_india.csv")
OUT = os.path.join(BASE_DIR, "model", "crop_prediction.pkl")

TOP_N = 8  # how many crops to keep per combination


def main():
    df = pd.read_csv(CSV)
    for c in ["State_Name", "District_Name", "Season", "Crop"]:
        df[c] = df[c].astype(str).str.strip()

    grouped = (
        df.groupby(["State_Name", "District_Name", "Season", "Crop"], observed=True)
          .agg(records=("Crop", "size"),
               total_area=("Area", "sum"),
               avg_area=("Area", "mean"),
               avg_production=("Production", "mean"))
          .reset_index()
    )

    lookup = {}
    for (state, district, season), block in grouped.groupby(
        ["State_Name", "District_Name", "Season"], observed=True
    ):
        total_area = float(block["total_area"].sum())
        block = block.sort_values("total_area", ascending=False).head(TOP_N)
        lookup["|".join([state, district, season])] = [
            {
                "crop": r.Crop,
                "share": round(float(r.total_area) / total_area * 100, 1) if total_area > 0 else 0.0,
                "years": int(r.records),
                "avg_area": None if pd.isna(r.avg_area) else round(float(r.avg_area), 1),
                "avg_production": None if pd.isna(r.avg_production) else round(float(r.avg_production), 1),
            }
            for r in block.itertuples()
        ]

    # Options for the cascading dropdowns
    districts = (
        df.groupby("State_Name", observed=True)["District_Name"]
          .apply(lambda s: sorted(s.unique().tolist())).to_dict()
    )
    seasons_by_district = (
        df.groupby(["State_Name", "District_Name"], observed=True)["Season"]
          .apply(lambda s: sorted(s.unique().tolist()))
    )
    seasons_map = {f"{k[0]}|{k[1]}": v for k, v in seasons_by_district.items()}

    joblib.dump(
        {
            "lookup": lookup,
            "states": sorted(df["State_Name"].unique().tolist()),
            "districts_by_state": districts,
            "seasons_by_district": seasons_map,
            "seasons": sorted(df["Season"].unique().tolist()),
        },
        OUT,
        compress=3,
    )

    print(f"[crop-prediction] {len(lookup)} state/district/season combinations -> {OUT}")


if __name__ == "__main__":
    main()
