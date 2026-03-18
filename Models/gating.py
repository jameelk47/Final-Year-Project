import numpy as np

class UncertaintyGater:
    def __init__(self, confidence_threshold=0.888, divergence_threshold=0.453):
        self.conf_thresh = confidence_threshold   # Threshold for HNN Sigma
        self.div_thresh = divergence_threshold    # Threshold for Mcodel Disagreement

    def get_recommendation(self, lgbm_mu, hnn_mu, hnn_sigma):
        """
        Logic to determine the 'Health' of the price recommendation.
        Inputs are in Log-Space.
        """
        # 1. Calculate Disagreement (Distance between models)
        disagreement = np.abs(lgbm_mu - hnn_mu)
        
        # 2. Determine the Status
        if hnn_sigma < self.conf_thresh and disagreement < self.div_thresh:
            status = "GREEN"
            msg = "High Confidence: Market data strongly supports this price."
        elif hnn_sigma < (self.conf_thresh * 2) and disagreement < (self.div_thresh * 2):
            status = "YELLOW"
            msg = "Moderate Confidence: Price may vary based on specific niche factors."
        else:
            status = "RED"
            msg = "Low Confidence: Market volatility detected. Use this as a rough guide only."

        # 3. Calculate Final Dollar Range (using LGBM as the anchor)
        # 95% Confidence Interval using HNN's uncertainty
        lower_bound = np.exp(lgbm_mu - 1.96 * hnn_sigma)
        upper_bound = np.exp(lgbm_mu + 1.96 * hnn_sigma)
        
        return {
            "price": np.exp(lgbm_mu),
            "range": (lower_bound, upper_bound),
            "status": status,
            "message": msg,
            "uncertainty_score": float(hnn_sigma)
        }

    def get_advanced_recommendation(self, lgbm_mu, hnn_mu, hnn_sigma):
        price_mid = np.exp(lgbm_mu)
        disagreement = np.abs(lgbm_mu - hnn_mu)
        
        # 1. First, check for System Conflict (The 'Safety Valve')
        if disagreement > self.div_thresh:
            return {
                "status": "RED",
                "tier": "Manual Review Required",
                "advice": "The models significantly disagree on this gig. Please check competitor prices manually.",
                "range": (None, None)
            }

        # 2. If models agree, proceed to Tiering
        price_consv = np.exp(lgbm_mu - 0.5 * hnn_sigma)
        price_aggr = np.exp(lgbm_mu + 0.5 * hnn_sigma)
        
        # 3. Gating based on Sigma
        if hnn_sigma < self.conf_thresh:
            status = "GREEN"
            rec_tier = "Balanced"
            advice = "Standard market conditions. Balanced pricing is optimal."
        else:
            status = "YELLOW"
            rec_tier = "Conservative"
            advice = "Higher market variance detected. Starting conservative is recommended for new sellers."

        return {
            "status": status,
            "recommended_tier": rec_tier,
            "tiers": {
                "conservative": round(price_consv, 2),
                "balanced": round(price_mid, 2),
                "aggressive": round(price_aggr, 2)
            },
            "advice": advice
        }

