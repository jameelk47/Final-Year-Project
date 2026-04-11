"""
Diagnostic script to discover which inputs land in GREEN, YELLOW, and RED tiers.
Run this ONCE to find confirmed examples for each tier, then use them in integration tests.

Candidate selection rationale (based on feature importance analysis):
  - Subcat_te (rank 1): all subcategories use confirmed valid Category+Subcat pairs
  - name_length (rank 2): GREEN titles are descriptive (40-70 chars), RED are very short
  - votes (rank 3): GREEN=100+, YELLOW=5-30, RED=0-1
  - TF-IDF keywords (rank 4-10): GREEN titles contain top tokens (create, video, write, app)
  - stars (rank 6): GREEN=4.7+, YELLOW=3.0-4.0, RED=0.0 or contradictory (5.0, 1 vote)

Usage: python -m Tests.tier_diagnostic
"""

import os
import pandas as pd
import numpy as np
import joblib
import kagglehub
import tensorflow as tf
from Models.hnn import HeteroscedasticKerasRegressor
from Models.gating import UncertaintyGater

def run_subcategory_tiers():
    """
    Groups subcategories into LOW / MEDIUM / HIGH price std dev bands
    using the 33rd and 66th percentiles of per-subcategory std dev.
    """
    path = kagglehub.dataset_download("kirilspiridonov/freelancers-offers-on-fiverr")
    csv_file = os.path.join(path, "fiverr_clean.csv")
    df = pd.read_csv(csv_file, encoding="latin-1")
    df = df.rename(columns={"ï..Category": "Category"})
    df["price"] = (
        df["price"]
        .str.replace(",", "", regex=False)
        .str.replace("$", "", regex=False)
        .astype(float)
    )

    # Map each subcategory to its parent category
    subcat_to_cat = df.groupby("Subcat")["Category"].first()

    stats = (
        df.groupby("Subcat")["price"]
        .agg(["mean", "std", "count"])
        .dropna()
        .sort_values("std")
    )

    low_cut = stats["std"].quantile(0.33)
    high_cut = stats["std"].quantile(0.66)

    low = stats[stats["std"] <= low_cut]
    mid = stats[(stats["std"] > low_cut) & (stats["std"] <= high_cut)]
    high = stats[stats["std"] > high_cut]

    print("\n" + "=" * 70)
    print("  SUBCATEGORY PRICE VARIANCE TIERS")
    print(f"  LOW  <= {low_cut:.2f}  |  MEDIUM <= {high_cut:.2f}  |  HIGH > {high_cut:.2f}")
    print("=" * 70)

    print(f"\n{'─'*70}")
    print(f"  LOW std dev ({len(low)} subcategories) — narrow price ranges expected")
    print(f"{'─'*70}")
    for subcat, row in low.iterrows():
        print(f"  {subcat:<45} mean=${row['mean']:>8.2f}  std=${row['std']:>8.2f}  n={int(row['count'])}")

    print(f"\n{'─'*70}")
    print(f"  MEDIUM std dev ({len(mid)} subcategories) — moderate price ranges")
    print(f"{'─'*70}")
    for subcat, row in mid.iterrows():
        print(f"  {subcat:<45} mean=${row['mean']:>8.2f}  std=${row['std']:>8.2f}  n={int(row['count'])}")

    print(f"\n{'─'*70}")
    print(f"  HIGH std dev ({len(high)} subcategories) — wide price ranges expected")
    print(f"{'─'*70}")
    for subcat, row in high.iterrows():
        print(f"  {subcat:<45} mean=${row['mean']:>8.2f}  std=${row['std']:>8.2f}  n={int(row['count'])}")

    print(f"\n{'='*70}")
    print(f"  Total subcategories: {len(stats)}")
    print(f"  LOW: {len(low)}  |  MEDIUM: {len(mid)}  |  HIGH: {len(high)}")
    print(f"{'='*70}")

    # Representative picks for integration testing
    lowest = stats.iloc[0]
    middle_idx = len(stats) // 2
    middle = stats.iloc[middle_idx]
    highest = stats.iloc[-1]

    print(f"\n{'='*70}")
    print("  REPRESENTATIVE SUBCATEGORIES FOR INTEGRATION TESTS")
    print(f"{'='*70}")
    print(f"  LOWEST  std:  {lowest.name}")
    print(f"    Category:   {subcat_to_cat[lowest.name]}")
    print(f"    mean=${lowest['mean']:>8.2f}  std=${lowest['std']:>8.2f}  n={int(lowest['count'])}")
    print()
    print(f"  MIDDLE  std:  {middle.name}")
    print(f"    Category:   {subcat_to_cat[middle.name]}")
    print(f"    mean=${middle['mean']:>8.2f}  std=${middle['std']:>8.2f}  n={int(middle['count'])}")
    print()
    print(f"  HIGHEST std:  {highest.name}")
    print(f"    Category:   {subcat_to_cat[highest.name]}")
    print(f"    mean=${highest['mean']:>8.2f}  std=${highest['std']:>8.2f}  n={int(highest['count'])}")
    print(f"{'='*70}")


if __name__ == "__main__":
    run_subcategory_tiers()
