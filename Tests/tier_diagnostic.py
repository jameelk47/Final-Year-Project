"""
Diagnostic script to discover which inputs land in GREEN, YELLOW, and RED tiers.
Run this ONCE to find confirmed examples for each tier, then use them in integration tests.

Usage: python -m Tests.tier_diagnostic
"""

import pandas as pd
import numpy as np
import joblib
import tensorflow as tf
from Models.hnn import HeteroscedasticKerasRegressor
from Models.gating import UncertaintyGater

def run_diagnostic():
    # 1. Load assets
    print("Loading models...")
    preprocessor = joblib.load('preprocessor.pkl')
    lgbm = joblib.load('lgbm_model.pkl')
    hnn = joblib.load('hnn_wrapper.pkl')
    hnn.model_ = tf.keras.models.load_model('hnn_weights.keras', compile=False)

    gater = UncertaintyGater()  # Uses new defaults: 0.888, 0.453

    # 2. Candidate inputs to probe
    candidates = [
        # (title, category, subcategory, stars, votes)
        # --- Likely GREEN: clear niche, strong reputation ---
        ("design a professional minimalist logo", "Graphics & Design", "Logo Design", 4.9, 150),
        ("I will create a wordpress website", "Programming & Tech", "WordPress", 4.8, 200),
        ("professional SEO audit for your website", "Digital Marketing", "SEO", 4.5, 80),
        ("I will write a professional resume and cover letter", "Writing & Translation", "Resume Writing", 4.7, 120),
        ("I will do professional video editing", "Video & Animation", "Video Editing", 4.6, 95),

        # --- Likely YELLOW: vague title, weak signals ---
        ("do a high quality task for you", "Digital Marketing", "Social Media Marketing", 0.0, 0),
        ("I will help you with your project", "Programming & Tech", "Other", 3.0, 5),
        ("I will do something creative", "Graphics & Design", "Other", 2.0, 2),
        ("quick and affordable service for you", "Lifestyle", "Other", 0.0, 0),
        ("I will be your virtual assistant", "Business", "Virtual Assistant", 3.5, 10),

        # --- Likely RED: gibberish, contradictory signals ---
        ("asdfghjkl12345 !!!!!!!!", "Writing & Translation", "Proofreading & Editing", 5.0, 1),
        ("xyzzy plugh nothing happens", "Programming & Tech", "WordPress", 0.0, 0),
        ("aaa bbb ccc ddd eee fff", "Graphics & Design", "Logo Design", 1.0, 0),
        ("I will sing happy birthday in a chicken suit", "Lifestyle", "Other", 1.0, 0),
        ("free money guaranteed results no scam", "Digital Marketing", "SEO", 5.0, 500),
    ]

    # 3. Run each through the pipeline and report
    print(f"\n{'Title':<55} {'Status':<8} {'Sigma':<8} {'Diverg':<8}")
    print("=" * 85)

    results_by_tier = {"GREEN": [], "YELLOW": [], "RED": []}

    for title, cat, subcat, stars, votes in candidates:
        raw = pd.DataFrame([{
            'name': title, 'Category': cat, 'Subcat': subcat,
            'stars': stars, 'votes': votes
        }])
        X_proc = preprocessor.transform(raw).toarray()

        lgbm_mu = lgbm.predict(X_proc)[0]
        hnn_mu, hnn_sigma = hnn.predict(X_proc, return_std=True)
        disagreement = abs(lgbm_mu - hnn_mu[0])

        result = gater.get_advanced_recommendation(lgbm_mu, hnn_mu[0], hnn_sigma[0])
        status = result['status']

        print(f"{title:<55} {status:<8} {hnn_sigma[0]:<8.3f} {disagreement:<8.3f}")

        results_by_tier[status].append({
            'title': title, 'category': cat, 'subcategory': subcat,
            'stars': stars, 'votes': votes,
            'sigma': hnn_sigma[0], 'divergence': disagreement
        })

    # 4. Summary
    print("\n" + "=" * 85)
    print("TIER SUMMARY")
    print("=" * 85)
    for tier in ["GREEN", "YELLOW", "RED"]:
        count = len(results_by_tier[tier])
        print(f"\n{tier}: {count} candidate(s)")
        for r in results_by_tier[tier]:
            print(f"  - \"{r['title']}\" (sigma={r['sigma']:.3f}, div={r['divergence']:.3f})")

    if not results_by_tier["GREEN"]:
        print("\n No GREEN candidates found — try adding inputs with stronger signals.")
    if not results_by_tier["YELLOW"]:
        print("\n No YELLOW candidates found — try adding inputs with vague titles and low reviews.")
    if not results_by_tier["RED"]:
        print("\n No RED candidates found — try adding gibberish or contradictory inputs.")


if __name__ == "__main__":
    run_diagnostic()
