import pytest
import pandas as pd
from Models.predictor import FiverrPricePredictor
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

    gater = UncertaintyGater(confidence_threshold=0.75, divergence_threshold=0.50)
    return FiverrPricePredictor(preprocessor, lgbm, hnn, gater)


# IT-01: sigma=0.696 (<0.75), div=0.075 (<0.50) → GREEN
def test_integration_logo_gig(predictor):
    result = predictor.predict_strategy(
        title="design a professional minimalist logo",
        category="Graphics & Design",
        sub_category="Logo Design",
        stars=4.9,
        votes=150
    )
    assert result['status'] == "GREEN"
    assert 'price' in result
    assert 'range' in result
    assert 'advice' in result
    lower, upper = result['range']
    assert lower < result['price'] < upper
    print(f"IT-01: ${result['price']} — {result['status']} — sigma={result['sigma']}")


# IT-02: sigma=1.092 (>0.75, <1.50), div=0.344 (<0.50) → YELLOW
def test_integration_wordpress_gig(predictor):
    result = predictor.predict_strategy(
        title="I will create a wordpress website",
        category="Programming & Tech",
        sub_category="WordPress",
        stars=4.8,
        votes=200
    )
    assert result['status'] == "YELLOW"
    assert 'price' in result
    assert 'range' in result
    lower, upper = result['range']
    assert lower < result['price'] < upper
    print(f"IT-02: ${result['price']} — {result['status']} — sigma={result['sigma']}")


# IT-03: div=1.024 (>>0.50) → RED (model disagreement)
def test_integration_gibberish_gig(predictor):
    result = predictor.predict_strategy(
        title="do a high quality task for you",
        category="Digital Marketing",
        sub_category="Social Media Marketing",
        stars=0.0,
        votes=0
    )
    assert result['status'] == "RED"
    assert "disagree" in result['advice'].lower() or "volatility" in result['advice'].lower()
    print(f"IT-03: {result['status']} — {result['advice']}")


def test_integration_golden_path(predictor):
    sample_row = X_test.iloc[0]

    result = predictor.predict_strategy(
        title=sample_row['name'],
        category=sample_row['Category'],
        sub_category=sample_row['Subcat'],
        stars=sample_row['stars'],
        votes=sample_row['votes']
    )

    assert result['status'] in ["GREEN", "YELLOW", "RED"]
    assert result['price'] > 0
    lower, upper = result['range']
    assert lower < result['price'] < upper
    print(f"Golden Path: ${result['price']} — {result['status']} — sigma={result['sigma']}")
