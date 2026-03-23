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

    gater = UncertaintyGater()  # Uses tightened defaults: 0.75, 0.50

    # 2. Candidate inputs to probe
    # Format: (title, category, subcategory, stars, votes)
    # All Category+Subcategory pairs are validated against the training dataset.
    # Candidates deliberately vary votes, stars, title keywords and length to
    # cover the full range of the model's uncertainty output.
    candidates = [
        # ── GREEN: high votes, high stars, descriptive keyword-rich title ──
        # votes=100-300 → well-established gig, stable price signal
        # title contains top TF-IDF tokens (create, video, write)
        ("I will create a professional mobile app for your business",
         "Programming & Tech", "Mobile Apps", 4.9, 200),

        ("professional video editing for youtube content creators",
         "Video & Animation", "Video Editing", 4.8, 150),

        ("I will write articles and blog posts for your website",
         "Writing & Translation", "Articles & Blog Posts", 4.9, 300),

        ("professional resume writing and cover letter service",
         "Writing & Translation", "Resume Writing", 4.7, 180),

        ("I will do voice over recording for your commercial project",
         "Music & Audio", "Voice Over", 4.8, 120),

        # ── YELLOW: low-moderate votes, average stars, generic title ──
        # votes=5-20 → limited market validation
        # title is real but generic — fewer TF-IDF keyword matches
        ("I will help with your business presentation",
         "Business", "Presentations", 3.5, 15),

        ("I will translate your documents for you",
         "Writing & Translation", "Translation", 3.8, 10),

        ("mixing and mastering service for your music",
         "Music & Audio", "Mixing & Mastering", 4.0, 8),

        ("website content writing service",
         "Writing & Translation", "Website Content", 3.5, 5),

        ("I will create animated content for you",
         "Video & Animation", "Animated GIFs", 3.0, 20),

        # ── RED: zero votes, contradictory signals, or gibberish title ──
        # cold_start=1 (stars=0, votes=0) → preprocessor cold-start flag triggered
        # OR: 5 stars + 1 review → contradictory signal the model has rarely seen
        # OR: gibberish → no TF-IDF matches, extreme name_length signal
        ("I will do anything for you",
         "Business", "Data Entry", 5.0, 1),

        ("asdfghjkl12345",
         "Writing & Translation", "Proofreading & Editing", 5.0, 1),

        ("quick service",
         "Lifestyle", "Gaming", 0.0, 0),

        ("services available now",
         "Digital Marketing", "Web Traffic", 0.0, 0),

        ("help",
         "Music & Audio", "DJ Drops & Tags", 0.0, 0),
    ]

    # 3. Run each through the pipeline and report
    print(f"\n{'Title':<58} {'Status':<8} {'Sigma':<8} {'Diverg':<8}")
    print("=" * 88)

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

        result = gater.get_recommendation(lgbm_mu, hnn_mu[0], hnn_sigma[0])
        status = result['status']

        print(f"{title:<58} {status:<8} {hnn_sigma[0]:<8.3f} {disagreement:<8.3f}")

        results_by_tier[status].append({
            'title': title, 'category': cat, 'subcategory': subcat,
            'stars': stars, 'votes': votes,
            'sigma': hnn_sigma[0], 'divergence': disagreement,
            'price': result['price'], 'range': result['range']
        })

    # 4. Summary
    print("\n" + "=" * 88)
    print("TIER SUMMARY")
    print("=" * 88)
    for tier in ["GREEN", "YELLOW", "RED"]:
        count = len(results_by_tier[tier])
        print(f"\n{tier}: {count} candidate(s)")
        for r in results_by_tier[tier]:
            lo, hi = r['range']
            print(
                f"  - \"{r['title']}\"\n"
                f"    sigma={r['sigma']:.3f}, div={r['divergence']:.3f}, "
                f"price=${r['price']:.2f}, range=(${lo:.2f}–${hi:.2f})"
            )

    if not results_by_tier["GREEN"]:
        print("\n⚠ No GREEN candidates found — increase votes or add stronger keyword titles.")
    if not results_by_tier["YELLOW"]:
        print("\n⚠ No YELLOW candidates found — try moderate votes (5-30) and generic titles.")
    if not results_by_tier["RED"]:
        print("\n⚠ No RED candidates found — try cold-start inputs (0 stars, 0 votes).")


if __name__ == "__main__":
    run_diagnostic()
