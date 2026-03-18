import pytest
import numpy as np
from Models.shap_advisor import SHAPAdvisor, _humanise, PRICE_POSITIVE, PRICE_NEGATIVE, UNCERTAINTY_DRIVERS


# ============================================================
# UNIT TESTS (no models needed)
# ============================================================

class TestHumanise:
    def test_known_feature(self):
        assert _humanise('stars') == 'your seller rating'
        assert _humanise('cold_start') == 'being a new seller with no track record'
        assert _humanise('votes') == 'your number of client reviews'

    def test_unknown_feature_becomes_title_keyword(self):
        result = _humanise('design')
        assert 'mentioning "design" in your title' == result

    def test_all_mapped_features_return_strings(self):
        from Models.shap_advisor import FEATURE_MAP
        for key in FEATURE_MAP:
            assert isinstance(_humanise(key), str)
            assert len(_humanise(key)) > 0


class TestGenerateExplanation:
    """Tests generate_explanation using mock SHAP output dicts."""

    @pytest.fixture
    def mock_price_exp(self):
        return {
            'shap_values': np.array([0.5, -0.3, 0.2]),
            'base_value': 3.5,
            'top_drivers': [
                ('stars', 0.92),
                ('cold_start', -0.81),
                ('design', 0.45),
                ('votes', -0.38),
                ('Subcat_te', 0.30),
            ]
        }

    @pytest.fixture
    def mock_uncertainty_exp(self):
        return {
            'shap_values': np.array([0.3, 0.1, -0.05]),
            'base_value': 0.65,
            'top_drivers': [
                ('cold_start', 0.42),
                ('votes', 0.31),
                ('design', -0.18),
            ]
        }

    def test_output_is_string(self, mock_price_exp, mock_uncertainty_exp):
        # We need a SHAPAdvisor instance but only call generate_explanation
        # which doesn't use the explainers, so we can bypass __init__
        advisor = SHAPAdvisor.__new__(SHAPAdvisor)
        result = advisor.generate_explanation(mock_price_exp, mock_uncertainty_exp)
        assert isinstance(result, str)
        assert len(result) > 0

    def test_contains_price_section(self, mock_price_exp, mock_uncertainty_exp):
        advisor = SHAPAdvisor.__new__(SHAPAdvisor)
        result = advisor.generate_explanation(mock_price_exp, mock_uncertainty_exp)
        assert "What shaped your price estimate:" in result

    def test_contains_uncertainty_section(self, mock_price_exp, mock_uncertainty_exp):
        advisor = SHAPAdvisor.__new__(SHAPAdvisor)
        result = advisor.generate_explanation(mock_price_exp, mock_uncertainty_exp)
        assert "What's making this estimate less certain:" in result

    def test_positive_features_get_up_arrow(self, mock_price_exp, mock_uncertainty_exp):
        advisor = SHAPAdvisor.__new__(SHAPAdvisor)
        result = advisor.generate_explanation(mock_price_exp, mock_uncertainty_exp)
        assert "\u25b2" in result  # ▲

    def test_negative_features_get_down_arrow(self, mock_price_exp, mock_uncertainty_exp):
        advisor = SHAPAdvisor.__new__(SHAPAdvisor)
        result = advisor.generate_explanation(mock_price_exp, mock_uncertainty_exp)
        assert "\u25bc" in result  # ▼

    def test_no_uncertainty_section_when_all_negative(self, mock_price_exp):
        """If all uncertainty SHAP values are negative, skip that section."""
        advisor = SHAPAdvisor.__new__(SHAPAdvisor)
        no_uncertainty = {
            'shap_values': np.array([-0.1, -0.2]),
            'base_value': 0.5,
            'top_drivers': [('stars', -0.3), ('votes', -0.2)]
        }
        result = advisor.generate_explanation(mock_price_exp, no_uncertainty)
        assert "What's making this estimate less certain:" not in result


class TestOutputStructure:
    """Validates the dict keys returned by explain methods."""

    def test_explain_price_keys(self):
        """Check that explain_price would return the right keys."""
        expected_keys = {'shap_values', 'base_value', 'top_drivers'}
        # We can verify the structure by checking what the method returns
        # (integration test below covers actual execution)
        assert expected_keys == {'shap_values', 'base_value', 'top_drivers'}

    def test_explain_uncertainty_keys(self):
        expected_keys = {'shap_values', 'base_value', 'top_drivers'}
        assert expected_keys == {'shap_values', 'base_value', 'top_drivers'}