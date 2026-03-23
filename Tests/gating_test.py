import pytest
import numpy as np
from Models.gating import UncertaintyGater

@pytest.fixture
def gater():
    """Initializes the gater with tightened thresholds."""
    return UncertaintyGater(confidence_threshold=0.75, divergence_threshold=0.50)

# TC-01: THE "STRONG CONSENSUS" SCENARIO
def test_green_light_consensus(gater):
    # Inputs: Low Sigma (0.1 < 0.30) and Low Disagreement (0.1 < 0.30)
    lgbm_mu = 4.0  # ~$54
    hnn_mu = 4.1   # ~$60
    hnn_sigma = 0.1

    result = gater.get_recommendation(lgbm_mu, hnn_mu, hnn_sigma)

    assert result['status'] == "GREEN"
    assert result['price'] == pytest.approx(54.60, abs=0.01)
    lower, upper = result['range']
    assert lower < result['price'] < upper

# TC-02: THE "MARKET VOLATILITY" SCENARIO
def test_yellow_light_uncertainty(gater):
    # Inputs: High Sigma (0.9 > 0.75, < 1.50) but Low Disagreement (0.1 < 0.50)
    lgbm_mu = 4.0
    hnn_mu = 4.1
    hnn_sigma = 0.9

    result = gater.get_recommendation(lgbm_mu, hnn_mu, hnn_sigma)

    assert result['status'] == "YELLOW"
    lower, upper = result['range']
    assert lower < result['price'] < upper

# TC-03: THE "MODEL CONTRADICTION" SCENARIO
def test_red_light_divergence(gater):
    # Inputs: Low Sigma (0.1) but High Disagreement (1.0 > 0.30)
    lgbm_mu = 4.0  # ~$54
    hnn_mu = 5.0   # ~$148
    hnn_sigma = 0.1

    result = gater.get_recommendation(lgbm_mu, hnn_mu, hnn_sigma)

    assert result['status'] == "RED"
    assert "disagree" in result['advice'].lower()

# TC-04: THE "TOTAL CHAOS" SCENARIO
def test_red_light_total_failure(gater):
    # Inputs: High Sigma (0.8) and High Disagreement (1.5)
    lgbm_mu = 4.0
    hnn_mu = 5.5
    hnn_sigma = 0.8

    result = gater.get_recommendation(lgbm_mu, hnn_mu, hnn_sigma)

    assert result['status'] == "RED"

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

    # In log-normal pricing, the upper gap is always larger in dollars
    assert upper_diff > lower_diff
