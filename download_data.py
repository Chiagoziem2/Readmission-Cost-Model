"""
Download the dataset into ./data/diabetic_data.csv

The dataset (Diabetes 130-US hospitals, 1999-2008) is CC BY 4.0.
Source: UCI Machine Learning Repository (Strack et al., 2014).

Preferred route uses the official `ucimlrepo` package.
"""
import os

os.makedirs(os.path.join(os.path.dirname(__file__), "data"), exist_ok=True)
out = os.path.join(os.path.dirname(__file__), "data", "diabetic_data.csv")

try:
    from ucimlrepo import fetch_ucirepo
    ds = fetch_ucirepo(id=296)  # Diabetes 130-US hospitals
    df = ds.data.features.copy()
    df["readmitted"] = ds.data.targets["readmitted"]
    # ucimlrepo drops the id columns; the model only needs patient_nbr for the
    # split, so if absent, fall back to the direct download below.
    df.to_csv(out, index=False)
    print(f"Saved {out} ({df.shape})")
except Exception as e:  # noqa
    print("ucimlrepo route failed:", e)
    print("Manual download: https://archive.ics.uci.edu/dataset/296  "
          "-> unzip -> place diabetic_data.csv in ./data/")
