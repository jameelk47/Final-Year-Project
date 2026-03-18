"""
Integration tests for SHAPAdvisor.

One test per confidence tier (GREEN / YELLOW / RED), using the same
confirmed inputs from tier_diagnostic.py. These tests verify that the
full SHAP pipeline runs end-to-end and returns valid, well-structured
output — not that specific SHAP values are correct (those are
non-deterministic).

Note: KernelExplainer (used for HNN uncertainty) is slow.
      nsamples=10 is used here to keep test run time reasonable.
      Production inference uses the default nsamples=100.

Usage: pytest Tests/shap_integration.py -v
"""

import pytest
import numpy as np
import pandas as pd
import joblib
import tensorflow as tf

from Models.hnn import HeteroscedasticKerasRegressor
from Models.shap_advisor import SHAPAdvisor
from Dataset.preprocessing import (
    fiverr_dss_pipeline,
    X_train,
    text_col, cat_col, target_enc_col, num_cols
)


# ============================================================
# Shared fixture — load everything once per test session
# ============================================================

@pytest.fixture(scope="module")
def advisor():
    """
    Initialises SHAPAdvisor with real trained models and a small
    background sample from the training set.
    """
    lgbm = joblib.load('lgbm_model.pkl')
    hnn = joblib.load('hnn_wrapper.pkl')
    hnn.model_ = tf.keras.models.load_model('hnn_weights.keras', compile=False)

    # Background dataset: 50 rows of processed training data for KernelExplainer
    X_train_proc = fiverr_dss_pipeline.transform(X_train).toarray()
    X_background = X_train_proc[:50]

    # Reconstruct feature names the same way ensemble.py does
    enc = fiverr_dss_pipeline.named_steps["encoding"]
    tfidf_features = enc.named_transformers_["tfidf"].get_feature_names_out()
    ohe_features = enc.named_transformers_["ohe"].get_feature_names_out(cat_col)
    target_features = np.array([f"{col}_te" for col in target_enc_col])
    num_features = np.array(num_cols)
    feature_names = np.concatenate([tfidf_features, ohe_features, target_features, num_features])

    return SHAPAdvisor(lgbm, hnn, X_background, feature_names)


@pytest.fixture(scope="module")
def preprocessor():
    return joblib.load('preprocessor.pkl')


def _process(preprocessor, title, category, subcat, stars, votes):
    """Helper: transforms raw input to a dense numpy array."""
    raw = pd.DataFrame([{
        'name': title, 'Category': category, 'Subcat': subcat,
        'stars': stars, 'votes': votes
    }])
    return preprocessor.transform(raw).toarray()


# ============================================================
# SI-01: GREEN tier — clear niche, strong reputation
# Confirmed: sigma=0.696, divergence=0.075
# ============================================================

def test_shap_green_tier(advisor, preprocessor):
    X = _process(preprocessor,
                 title="design a professional minimalist logo",
                 category="Graphics & Design",
                 subcat="Logo Design",
                 stars=4.9,
                 votes=150)

    # --- explain_price ---
    price_exp = advisor.explain_price(X)

    assert isinstance(price_exp, dict)
    assert set(price_exp.keys()) == {'shap_values', 'base_value', 'top_drivers'}
    assert isinstance(price_exp['base_value'], float)
    assert len(price_exp['top_drivers']) > 0

    for feat_name, shap_val in price_exp['top_drivers']:
        assert isinstance(feat_name, str)
        assert isinstance(shap_val, float)

    # --- explain_uncertainty (fast mode) ---
    unc_exp = advisor.explain_uncertainty(X, nsamples=10)

    assert isinstance(unc_exp, dict)
    assert set(unc_exp.keys()) == {'shap_values', 'base_value', 'top_drivers'}

    # --- generate_explanation ---
    text = advisor.generate_explanation(price_exp, unc_exp)

    assert isinstance(text, str)
    assert len(text) > 0
    assert "What shaped your price estimate:" in text
    print(f"\nSI-01 GREEN Explanation:\n{text}")


# ============================================================
# SI-02: YELLOW tier — high sigma triggers uncertainty
# Confirmed: sigma=1.092, divergence=0.344
# ============================================================

def test_shap_yellow_tier(advisor, preprocessor):
    X = _process(preprocessor,
                 title="I will create a wordpress website",
                 category="Programming & Tech",
                 subcat="WordPress",
                 stars=4.8,
                 votes=200)

    price_exp = advisor.explain_price(X)
    unc_exp = advisor.explain_uncertainty(X, nsamples=10)

    assert isinstance(price_exp, dict)
    assert len(price_exp['top_drivers']) > 0

    assert isinstance(unc_exp, dict)
    assert len(unc_exp['top_drivers']) > 0

    text = advisor.generate_explanation(price_exp, unc_exp)

    assert isinstance(text, str)
    assert "What shaped your price estimate:" in text
    # YELLOW has high sigma so at least one uncertainty driver should be positive
    # (uncertainty section appears when any driver has positive SHAP on sigma)
    print(f"\nSI-02 YELLOW Explanation:\n{text}")


# ============================================================
# SI-03: RED tier — high divergence, system conflict
# Confirmed: sigma=0.911, divergence=1.024
# ============================================================

def test_shap_red_tier(advisor, preprocessor):
    X = _process(preprocessor,
                 title="do a high quality task for you",
                 category="Digital Marketing",
                 subcat="Social Media Marketing",
                 stars=0.0,
                 votes=0)

    # SHAP should still run successfully even on a RED-tier input
    price_exp = advisor.explain_price(X)
    unc_exp = advisor.explain_uncertainty(X, nsamples=10)

    assert isinstance(price_exp, dict)
    assert set(price_exp.keys()) == {'shap_values', 'base_value', 'top_drivers'}

    assert isinstance(unc_exp, dict)
    assert set(unc_exp.keys()) == {'shap_values', 'base_value', 'top_drivers'}

    text = advisor.generate_explanation(price_exp, unc_exp)

    assert isinstance(text, str)
    assert len(text) > 0
    assert "What shaped your price estimate:" in text
    print(f"\nSI-03 RED Explanation:\n{text}")
