import numpy as np

class UncertaintyGater:
    def __init__(
        self,
        green_plus_threshold=0.7670,
        green_threshold=0.8993,
        yellow_threshold=1.0571,
        divergence_threshold=0.3211,
    ):
        # Calibrated on held-out test predictions (see Tests/calibration_threshold.py).
        # Sigma tiers: percentiles of HNN total σ (MC-dropout) — P10 / P30 / P70.
        # Divergence: P70 of |LGBM μ − HNN μ| (preferable to risk-curve elbow here:
        # divergence vs MAE is noisy; P70 flags the worst ~30% disagreements without
        # over-triggering RED relative to the elbow heuristic).
        self.green_plus_thresh = green_plus_threshold
        self.green_thresh = green_threshold
        self.yellow_thresh = yellow_threshold
        self.div_thresh = divergence_threshold

    def get_recommendation(self, lgbm_mu, hnn_mu, hnn_sigma):
        """
        Decision-support recommendation with consistent output structure.
        Always returns: predicted price, range, status, and graduated advice.
        Inputs are in Log-Space.

        Logic:
          1. disagreement > div_thresh           → RED  (models disagree)
          2. Models agree AND sigma < green_plus → GREEN_PLUS (very high confidence)
          3. Models agree AND sigma < green      → GREEN
          4. Models agree AND sigma < yellow     → YELLOW
          5. Models agree AND sigma >= yellow    → RED  (extreme volatility)
        """
        disagreement = np.abs(lgbm_mu - hnn_mu)

        price = round(float(np.exp(lgbm_mu)), 2)
        lower_bound = round(float(np.exp(lgbm_mu - 1.96 * hnn_sigma)), 2)
        upper_bound = round(float(np.exp(lgbm_mu + 1.96 * hnn_sigma)), 2)

        if disagreement > self.div_thresh:
            status = "RED"
            advice = (
                "The models disagree significantly on this gig. "
                "The predicted price may not be reliable. "
                "Manual review of competitor prices is recommended before listing."
            )
        elif hnn_sigma < self.green_plus_thresh:
            status = "GREEN_PLUS"
            advice = (
                "Very high confidence — this segment shows unusually stable pricing "
                "and both models agree closely. Use the predicted price as your main "
                "anchor. If your portfolio, delivery, or niche clearly justify a "
                "premium, you may also consider pricing toward the upper bound of the range."
            )
        elif hnn_sigma < self.green_thresh:
            status = "GREEN"
            advice = (
                "Stable market — prices in this segment are consistent and "
                "both models agree. You can use the predicted price as a "
                "strong starting point."
            )
        elif hnn_sigma < self.yellow_thresh:
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

        if disagreement > self.div_thresh:
            return {
                "status": "RED",
                "tier": "Manual Review Required",
                "advice": "The models significantly disagree on this gig. Please check competitor prices manually.",
                "range": (None, None)
            }

        price_consv = np.exp(lgbm_mu - 0.5 * hnn_sigma)
        price_aggr = np.exp(lgbm_mu + 0.5 * hnn_sigma)

        if hnn_sigma < self.green_plus_thresh:
            status = "GREEN_PLUS"
            rec_tier = "Balanced (premium allowed)"
            advice = (
                "Very stable conditions. Balanced pricing is optimal; a justified "
                "premium may move you toward the aggressive tier."
            )
        elif hnn_sigma < self.green_thresh:
            status = "GREEN"
            rec_tier = "Balanced"
            advice = "Standard market conditions. Balanced pricing is optimal."
        elif hnn_sigma < self.yellow_thresh:
            status = "YELLOW"
            rec_tier = "Conservative"
            advice = "Higher market variance detected. Starting conservative is recommended for new sellers."
        else:
            return {
                "status": "RED",
                "tier": "Manual Review Required",
                "advice": "Extreme market volatility. Review competitor prices manually.",
                "range": (None, None)
            }

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
