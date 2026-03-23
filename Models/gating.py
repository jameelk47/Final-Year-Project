import numpy as np

class UncertaintyGater:
    def __init__(self, confidence_threshold=0.75, divergence_threshold=0.50):
        self.conf_thresh = confidence_threshold   # Threshold for HNN Sigma
        self.div_thresh = divergence_threshold    # Threshold for Model Disagreement

    def get_recommendation(self, lgbm_mu, hnn_mu, hnn_sigma):
        """
        Decision-support recommendation with consistent output structure.
        Always returns: predicted price, range, status, and graduated advice.
        Inputs are in Log-Space.
        """
        # 1. Calculate Disagreement (Distance between models)
        disagreement = np.abs(lgbm_mu - hnn_mu)

        # 2. Calculate Price Range (using LGBM as anchor, HNN sigma for bounds)
        price = round(float(np.exp(lgbm_mu)), 2)
        lower_bound = round(float(np.exp(lgbm_mu - 1.96 * hnn_sigma)), 2)
        upper_bound = round(float(np.exp(lgbm_mu + 1.96 * hnn_sigma)), 2)

        # 3. Determine Status and Graduated Advice
        #    - Divergence (epistemic): do the models agree on this input?
        #    - Sigma (aleatoric): how volatile is pricing in this market segment?
        if disagreement > self.div_thresh:
            # Safety valve: models fundamentally disagree on this input
            status = "RED"
            advice = (
                "The models disagree significantly on this gig. "
                "The predicted price may not be reliable. "
                "Manual review of competitor prices is recommended before listing."
            )
        elif hnn_sigma < self.conf_thresh and disagreement < self.div_thresh:
            status = "GREEN"
            advice = (
                "Stable market — prices in this segment are consistent and "
                "both models agree. You can use the predicted price as a "
                "strong starting point. The price range is narrow."
            )
        elif hnn_sigma < (self.conf_thresh * 2):
            status = "YELLOW"
            advice = (
                "This market segment has volatile pricing — gigs like yours "
                "vary widely in price. Both models agree on direction, "
                "but consider starting closer to the lower bound and adjusting "
                "as you gain reviews."
            )
        else:
            status = "RED"
            advice = (
                "High market volatility — prices in this segment vary "
                "enormously. The wide range reflects genuine market uncertainty. "
                "Use the lower bound as a starting point and review competitor "
                "prices manually."
            )

        return {
            "price": price,
            "range": (lower_bound, upper_bound),
            "status": status,
            "advice": advice,
            "sigma": round(float(hnn_sigma), 4),
            "divergence": round(float(disagreement), 4)
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

