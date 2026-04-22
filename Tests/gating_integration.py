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

    gater = UncertaintyGater()
    return FiverrPricePredictor(preprocessor, lgbm, hnn, gater)


# IT-01: LOW variance subcategory (Article to VideoNew, std=$3.29)
# Expect GREEN — tight pricing segment, strong seller signals
def test_integration_low_variance(predictor):
    result = predictor.predict_strategy(
        title="I will turn your article into an engaging video",
        category="Video & Animation",
        sub_category="Article to VideoNew",
        stars=4.9,
        votes=150
    )
    assert result['status'] in ["GREEN_PLUS", "GREEN"]
    assert 'price' in result
    assert 'range' in result
    assert 'advice' in result
    lower, upper = result['range']
    assert lower < result['price'] < upper
    print(f"IT-01 (LOW var): ${result['price']} — {result['status']} — sigma={result['sigma']}")


# IT-02: MEDIUM variance subcategory (Unboxing VideosNew, std=$53.74)
# Expect YELLOW — moderate pricing spread
def test_integration_medium_variance(predictor):
    result = predictor.predict_strategy(
        title="I will create a professional unboxing video for your product",
        category="Video & Animation",
        sub_category="Unboxing VideosNew",
        stars=4.5,
        votes=50
    )
    assert result['status'] in ["YELLOW"]
    assert 'price' in result
    assert 'range' in result
    lower, upper = result['range']
    assert lower < result['price'] < upper
    print(f"IT-02 (MED var): ${result['price']} — {result['status']} — sigma={result['sigma']}")


# IT-03: HIGH variance subcategory (Game Development, std=$1046)
# Expect YELLOW or RED — extreme pricing spread
def test_integration_high_variance(predictor):
    result = predictor.predict_strategy(
        title="I will develop a mobile game for you",
        category="Programming & Tech",
        sub_category="Game Development",
        stars=4.0,
        votes=10
    )
    assert result['status'] in ["RED"]
    assert 'price' in result
    assert 'range' in result
    lower, upper = result['range']
    assert lower < result['price'] < upper
    print(f"IT-03 (HIGH var): ${result['price']} — {result['status']} — sigma={result['sigma']}")


def test_integration_golden_path(predictor):
    sample_row = X_test.iloc[0]

    result = predictor.predict_strategy(
        title=sample_row['name'],
        category=sample_row['Category'],
        sub_category=sample_row['Subcat'],
        stars=sample_row['stars'],
        votes=sample_row['votes']
    )

    assert result['status'] in ["GREEN_PLUS", "GREEN", "YELLOW", "RED"]
    assert result['price'] > 0
    lower, upper = result['range']
    assert lower < result['price'] < upper
    print(f"Golden Path: ${result['price']} — {result['status']} — sigma={result['sigma']}")
