import pytest
import numpy as np
from Models.gating import UncertaintyGater # Ensure this matches your filename

@pytest.fixture
def gater():
    """Initializes the gater with your BSc thesis thresholds."""
    return UncertaintyGater(confidence_threshold=0.25, divergence_threshold=0.5)

# TC-01: THE "STRONG CONSENSUS" SCENARIO
def test_green_light_consensus(gater):
    # Inputs: Low Sigma (0.1 < 0.25) and Low Disagreement (0.1 < 0.5)
    lgbm_mu = 4.0  # ~$54
    hnn_mu = 4.1   # ~$60
    hnn_sigma = 0.1
    
    result = gater.get_advanced_recommendation(lgbm_mu, hnn_mu, hnn_sigma)
    
    assert result['status'] == "GREEN"
    assert result['recommended_tier'] == "Balanced"
    assert result['tiers']['balanced'] == 54.6  # exp(4.0)

    assert result['tiers']['conservative'] == pytest.approx(51.94, abs=0.01)
    assert result['tiers']['aggressive'] == pytest.approx(57.40, abs=0.01)

# TC-02: THE "MARKET VOLATILITY" SCENARIO
def test_yellow_light_uncertainty(gater):
    # Inputs: High Sigma (0.4 > 0.25) but Low Disagreement (0.1 < 0.5)
    lgbm_mu = 4.0
    hnn_mu = 4.1
    hnn_sigma = 0.4
    
    result = gater.get_advanced_recommendation(lgbm_mu, hnn_mu, hnn_sigma)
    
    assert result['status'] == "YELLOW"
    assert result['recommended_tier'] == "Conservative"
    # Ensure Conservative is actually lower than Balanced
    assert result['tiers']['conservative'] < result['tiers']['balanced']

# TC-03: THE "MODEL CONTRADICTION" SCENARIO
def test_red_light_divergence(gater):
    # Inputs: Low Sigma (0.1) but High Disagreement (1.0 > 0.5)
    # This simulates the "Engine Failure" where models fight.
    lgbm_mu = 4.0  # ~$54
    hnn_mu = 5.0   # ~$148
    hnn_sigma = 0.1
    
    result = gater.get_advanced_recommendation(lgbm_mu, hnn_mu, hnn_sigma)
    
    assert result['status'] == "RED"
    assert result['tier'] == "Manual Review Required"
    assert result['range'] == (None, None)

# TC-04: THE "TOTAL CHAOS" SCENARIO
def test_red_light_total_failure(gater):
    # Inputs: High Sigma (0.8) and High Disagreement (1.5)
    # Simulates gibberish input where nothing makes sense.
    lgbm_mu = 4.0
    hnn_mu = 5.5
    hnn_sigma = 0.8
    
    result = gater.get_advanced_recommendation(lgbm_mu, hnn_mu, hnn_sigma)
    
    assert result['status'] == "RED"
    assert "volatility" in result['advice'] or "disagree" in result['advice']

def test_log_normal_asymmetry(gater):
    """Verifies that the price gap is larger on the aggressive side than conservative."""
    lgbm_mu = 4.0
    hnn_mu = 4.0
    hnn_sigma = 0.5 # Wide sigma makes the asymmetry obvious
    
    result = gater.get_advanced_recommendation(lgbm_mu, hnn_mu, hnn_sigma)
    
    consv = result['tiers']['conservative'] # exp(3.75) ≈ 42.52
    bal = result['tiers']['balanced']       # exp(4.00) ≈ 54.60
    aggr = result['tiers']['aggressive']    # exp(4.25) ≈ 70.11
    
    lower_diff = bal - consv  # ~12.08
    upper_diff = aggr - bal   # ~15.51
    
    # In log-normal pricing, the 'Aggressive' jump is always larger in dollars
    # than the 'Conservative' drop. This proves your math is in log-space!
    assert upper_diff > lower_diff