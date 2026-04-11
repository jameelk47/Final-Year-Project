import pytest
import numpy as np
from Models.gating import UncertaintyGater

@pytest.fixture
def gater():
    """Initializes the gater with percentile-calibrated thresholds."""
    return UncertaintyGater(green_threshold=0.7538, yellow_threshold=0.9558, divergence_threshold=0.4609)

# TC-01: GREEN — low sigma, models agree
def test_green_light_consensus(gater):
    lgbm_mu = 4.0  # ~$54
    hnn_mu = 4.1   # ~$60  (disagreement 0.1 < 0.4609)
    hnn_sigma = 0.5  # well below green_threshold 0.7538

    result = gater.get_recommendation(lgbm_mu, hnn_mu, hnn_sigma)

    assert result['status'] == "GREEN"
    assert result['price'] == pytest.approx(54.60, abs=0.01)
    lower, upper = result['range']
    assert lower < result['price'] < upper

# TC-02: YELLOW — sigma between green and yellow thresholds, models agree
def test_yellow_light_uncertainty(gater):
    lgbm_mu = 4.0
    hnn_mu = 4.1   # disagreement 0.1 < 0.4609
    hnn_sigma = 0.85  # above green (0.7538), below yellow (0.9558)

    result = gater.get_recommendation(lgbm_mu, hnn_mu, hnn_sigma)

    assert result['status'] == "YELLOW"
    lower, upper = result['range']
    assert lower < result['price'] < upper

# TC-03: RED — models disagree (divergence > threshold)
def test_red_light_divergence(gater):
    lgbm_mu = 4.0  # ~$54
    hnn_mu = 5.0   # ~$148  (disagreement 1.0 > 0.4609)
    hnn_sigma = 0.5

    result = gater.get_recommendation(lgbm_mu, hnn_mu, hnn_sigma)

    assert result['status'] == "RED"
    assert "disagree" in result['advice'].lower()

# TC-04: RED — models disagree AND high sigma (divergence takes priority)
def test_red_light_total_failure(gater):
    lgbm_mu = 4.0
    hnn_mu = 5.5   # disagreement 1.5 > 0.4609
    hnn_sigma = 0.85

    result = gater.get_recommendation(lgbm_mu, hnn_mu, hnn_sigma)

    assert result['status'] == "RED"
    assert "disagree" in result['advice'].lower()

# TC-05: RED — models agree but sigma >= yellow_threshold (extreme volatility)
def test_red_light_extreme_volatility(gater):
    lgbm_mu = 4.0
    hnn_mu = 4.1   # disagreement 0.1 < 0.4609
    hnn_sigma = 1.2  # above yellow_threshold 0.9558

    result = gater.get_recommendation(lgbm_mu, hnn_mu, hnn_sigma)

    assert result['status'] == "RED"
    assert "volatility" in result['advice'].lower()

# TC-06: YELLOW at the boundary — sigma just above green_threshold
def test_yellow_at_green_boundary(gater):
    lgbm_mu = 4.0
    hnn_mu = 4.0
    hnn_sigma = 0.76  # just above green_threshold 0.7538

    result = gater.get_recommendation(lgbm_mu, hnn_mu, hnn_sigma)

    assert result['status'] == "YELLOW"

# TC-07: RED at the boundary — sigma exactly at yellow_threshold
def test_red_at_yellow_boundary(gater):
    lgbm_mu = 4.0
    hnn_mu = 4.0
    hnn_sigma = 0.9558  # exactly at yellow_threshold → falls to else (RED)

    result = gater.get_recommendation(lgbm_mu, hnn_mu, hnn_sigma)

    assert result['status'] == "RED"

# TC-08: Divergence takes priority over low sigma
def test_divergence_overrides_low_sigma(gater):
    lgbm_mu = 4.0
    hnn_mu = 4.5   # disagreement 0.5 > 0.4609
    hnn_sigma = 0.5  # would be GREEN by sigma alone

    result = gater.get_recommendation(lgbm_mu, hnn_mu, hnn_sigma)

    assert result['status'] == "RED"
    assert "disagree" in result['advice'].lower()

def test_log_normal_asymmetry(gater):
    """Verifies that the price range is asymmetric (wider above than below)."""
    lgbm_mu = 4.0
    hnn_mu = 4.0
    hnn_sigma = 0.5

    result = gater.get_recommendation(lgbm_mu, hnn_mu, hnn_sigma)

    price = result['price']
    lower, upper = result['range']

    lower_diff = price - lower
    upper_diff = upper - price

    assert upper_diff > lower_diff
