import pytest
import pandas as pd
from Models.predictor import FiverrPricePredictor # The wrapper we discussed
import joblib
import tensorflow as tf
from Models.gating import UncertaintyGater
from Dataset.preprocessing import X_test


@pytest.fixture
def predictor():
    """Initializes the full system with real trained components."""

    preprocessor = joblib.load('preprocessor.pkl')
    lgbm = joblib.load('lgbm_model.pkl')
    hnn = joblib.load('hnn_wrapper.pkl')
    hnn.model_ = tf.keras.models.load_model('hnn_weights.keras', compile=False)
    
    # Initialize the gater with data-driven thresholds from Risk-Coverage analysis
    gater = UncertaintyGater(confidence_threshold=0.888, divergence_threshold=0.453)
    return FiverrPricePredictor(preprocessor, lgbm, hnn, gater)

# IT-01: GREEN — Clear niche, strong keyword signals, low sigma, low divergence
# Confirmed: sigma=0.696 (<0.888), divergence=0.075 (<0.453)
def test_integration_green_tier(predictor):
    result = predictor.predict_strategy(
        title="design a professional minimalist logo",
        category="Graphics & Design",
        sub_category="Logo Design",
        stars=4.9,
        votes=150
    )
    assert result['status'] == "GREEN"
    assert result['recommended_tier'] == "Balanced"
    assert "balanced" in result['tiers']
    assert "conservative" in result['tiers']
    assert "aggressive" in result['tiers']
    print(f"IT-01 GREEN: ${result['tiers']['balanced']} — {result['advice']}")

# IT-02: YELLOW — High sigma triggers uncertainty flag (models agree but market noisy)
# Confirmed: sigma=1.092 (>0.888), divergence=0.344 (<0.453)
def test_integration_yellow_tier(predictor):
    result = predictor.predict_strategy(
        title="I will create a wordpress website",
        category="Programming & Tech",
        sub_category="WordPress",
        stars=4.8,
        votes=200
    )
    assert result['status'] == "YELLOW"
    assert result['recommended_tier'] == "Conservative"
    assert "tiers" in result
    print(f"IT-02 YELLOW: ${result['tiers']['conservative']} — {result['advice']}")

# IT-03: RED — High divergence triggers system conflict flag
# Confirmed: sigma=0.911, divergence=1.024 (>>0.453)
def test_integration_red_tier(predictor):
    result = predictor.predict_strategy(
        title="do a high quality task for you",
        category="Digital Marketing",
        sub_category="Social Media Marketing",
        stars=0.0,
        votes=0
    )
    assert result['status'] == "RED"
    assert result['tier'] == "Manual Review Required"
    assert result['range'] == (None, None)
    print(f"IT-03 RED: {result['advice']}")

def test_integration_golden_path(predictor):
    # 1. Grab a real row from the test set (one that the model should know well)
    # We'll take the first row that actually exists in your data
    sample_row = X_test.iloc[0]
    
    # 2. Feed it into the predictor
    result = predictor.predict_strategy(
        title=sample_row['name'],
        category=sample_row['Category'],
        sub_category=sample_row['Subcat'],
        stars=sample_row['stars'],
        votes=sample_row['votes']
    )
    
    # 3. Validation
    # Since this came from the real test set, it SHOULD be Green or Yellow
    # based on your 95th percentile calibration.
    assert result['status'] in ["GREEN", "YELLOW"]
    print(f"Golden Path Sigma: {result.get('uncertainty_score', 'N/A')}")