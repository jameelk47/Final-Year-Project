import shap
import numpy as np
import random


# --- Phrase pools for natural variation ---

PRICE_POSITIVE = [
    "{feature} is working in your favour, pushing your predicted price higher.",
    "{feature} is a strong positive signal for your pricing.",
    "The model sees {feature} as a value booster for this gig.",
    "{feature} contributes positively — similar gigs with this trait tend to earn more.",
]

PRICE_NEGATIVE = [
    "{feature} is pulling your predicted price down.",
    "{feature} is a limiting factor in your pricing right now.",
    "The model associates {feature} with lower-priced gigs.",
    "{feature} is working against your price estimate.",
]

UNCERTAINTY_DRIVERS = [
    "{feature} is a key source of pricing uncertainty for this gig.",
    "The model finds {feature} hard to price reliably.",
    "{feature} introduces noise — similar gigs vary a lot on this trait.",
    "{feature} is making the prediction less stable.",
]

FEATURE_MAP = {
    'cold_start':   'being a new seller with no track record',
    'votes':        'your number of client reviews',
    'stars':        'your seller rating',
    'Subcat_te':    'the typical pricing in your subcategory',
    'name_length':  'how descriptive your gig title is',
    'votes_capped': 'having hit the 1,000+ reviews milestone',
}


def _humanise(feature_name):
    """Maps raw feature names to human-readable phrases."""
    if feature_name in FEATURE_MAP:
        return FEATURE_MAP[feature_name]
    # TF-IDF features are already words from the gig title
    return f'mentioning "{feature_name}" in your title'


class SHAPAdvisor:
    """
    Explains predictions from the LGBM (price) and HNN (uncertainty)
    using SHAP values.

    - explain_price():       which features drove the LGBM price prediction
    - explain_uncertainty():  which features drove the HNN uncertainty (sigma)
    """

    def __init__(self, lgbm_model, hnn_model, X_background, feature_names):
        """
        Parameters
        ----------
        lgbm_model : fitted LGBMRegressor
        hnn_model  : fitted HeteroscedasticKerasRegressor (with .predict(X, return_std=True))
        X_background : np.ndarray
            A small sample of processed training data (e.g. 100 rows).
            Used as the baseline reference for KernelExplainer.
        feature_names : array-like
            Human-readable names for each of the processed features.
        """
        self.feature_names = np.array(feature_names)

        # LGBM: TreeExplainer (fast, exact for tree-based models)
        self.lgbm_explainer = shap.TreeExplainer(lgbm_model)

        # HNN: KernelExplainer (model-agnostic, works with any predict function)
        # We wrap the HNN predict to return only sigma (uncertainty)
        def hnn_uncertainty_fn(X):
            _, sigma = hnn_model.predict(X, return_std=True)
            return sigma

        self.hnn_explainer = shap.KernelExplainer(
            hnn_uncertainty_fn,
            shap.kmeans(X_background, 10)  # summarise background into 10 centroids
        )

    def _filter_active_drivers(self, X_processed, shap_values_flat, top_n=10):
        """
        Filter SHAP drivers to only include features that are active
        (non-zero) in the input. This prevents misleading explanations
        from inactive one-hot or TF-IDF features — e.g. reporting
        "Category_Lifestyle is pushing price down" when the user
        selected a different category.

        SHAP assigns values to zero-valued features because it measures
        impact relative to a baseline. While technically valid, showing
        the effect of an absent feature is not actionable advice.
        """
        input_values = X_processed[0] if X_processed.ndim == 2 else X_processed

        # Rank ALL features by absolute SHAP impact
        ranked_indices = np.argsort(np.abs(shap_values_flat))[::-1]

        # Keep only features with non-zero input values
        active_drivers = []
        for i in ranked_indices:
            if input_values[i] != 0:
                active_drivers.append(
                    (self.feature_names[i], float(shap_values_flat[i]))
                )
            if len(active_drivers) == top_n:
                break

        return active_drivers

    def explain_price(self, X_processed):
    
        shap_values = self.lgbm_explainer.shap_values(X_processed)
        base_value = self.lgbm_explainer.expected_value

        # For a single row, flatten to 1D
        if shap_values.ndim == 2 and shap_values.shape[0] == 1:
            shap_values_flat = shap_values[0]
        else:
            shap_values_flat = shap_values

        # Filter to only active (non-zero) features in the input
        top_drivers = self._filter_active_drivers(X_processed, shap_values_flat)

        return {
            'shap_values': shap_values,
            'base_value': float(base_value),
            'top_drivers': top_drivers
        }

    def explain_uncertainty(self, X_processed, nsamples=100):

        shap_values = self.hnn_explainer.shap_values(X_processed, nsamples=nsamples, l1_reg=0)
        base_value = self.hnn_explainer.expected_value

        # For a single row, flatten to 1D
        if isinstance(shap_values, list):
            shap_values = shap_values[0]
        if shap_values.ndim == 2 and shap_values.shape[0] == 1:
            shap_values_flat = shap_values[0]
        else:
            shap_values_flat = shap_values

        # Filter to only active (non-zero) features in the input
        top_drivers = self._filter_active_drivers(X_processed, shap_values_flat)

        return {
            'shap_values': shap_values,
            'base_value': float(base_value) if np.isscalar(base_value) else float(base_value[0]),
            'top_drivers': top_drivers
        }

    def generate_price_explanation(self, price_exp):
        """
        Human-readable explanation of what drove the price prediction.

        Parameters
        ----------
        price_exp : dict returned by explain_price()

        Returns
        -------
        str : natural-language explanation text
        """
        lines = ["What shaped your price estimate:", ""]

        for feature, value in price_exp['top_drivers'][:5]:
            name = _humanise(feature)
            if value > 0:
                lines.append(f"  \u25b2 {random.choice(PRICE_POSITIVE).format(feature=name)}")
            else:
                lines.append(f"  \u25bc {random.choice(PRICE_NEGATIVE).format(feature=name)}")

        return "\n".join(lines)

    def generate_uncertainty_explanation(self, uncertainty_exp):
        """
        Human-readable explanation of what drove prediction uncertainty.
        Only surfaces features that *increased* uncertainty (positive SHAP on sigma).

        Parameters
        ----------
        uncertainty_exp : dict returned by explain_uncertainty()

        Returns
        -------
        str or None : explanation text, or None if no features increased uncertainty
        """
        uncertainty_drivers = [
            (f, v) for f, v in uncertainty_exp['top_drivers'][:5] if v > 0
        ]

        if not uncertainty_drivers:
            return None

        lines = ["What's making this estimate less certain:", ""]
        for feature, value in uncertainty_drivers[:3]:
            name = _humanise(feature)
            lines.append(f"  ? {random.choice(UNCERTAINTY_DRIVERS).format(feature=name)}")

        return "\n".join(lines)

    def generate_explanation(self, price_exp, uncertainty_exp):
        """
        Convenience wrapper that combines both explanations into one block.
        Prefer calling generate_price_explanation / generate_uncertainty_explanation
        individually when you need to display them separately.
        """
        parts = [self.generate_price_explanation(price_exp)]
        unc = self.generate_uncertainty_explanation(uncertainty_exp)
        if unc:
            parts.append(unc)
        return "\n\n".join(parts)