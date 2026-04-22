import streamlit as st
import joblib
import numpy as np
import pandas as pd
import os
import sys
import kagglehub
import shap

# Project root so we can import Models/ and load .pkl files
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)

import tensorflow as tf
from Models.hnn import HeteroscedasticKerasRegressor, aleatoric_loss
from Models.gating import UncertaintyGater
from Models.shap_advisor import SHAPAdvisor, _humanise

# Page Config
st.set_page_config(
    page_title="Fiverr Price Advisor",
    page_icon="💰",
    layout="centered",
)


# ── Load Models + SHAP Advisor (cached, loaded once) ──
@st.cache_resource
def load_models():
    preprocessor = joblib.load(os.path.join(ROOT_DIR, "preprocessor.pkl"))
    lgbm = joblib.load(os.path.join(ROOT_DIR, "lgbm_model.pkl"))
    hnn = joblib.load(os.path.join(ROOT_DIR, "hnn_wrapper.pkl"))
    hnn.model_ = tf.keras.models.load_model(
        os.path.join(ROOT_DIR, "hnn_weights.keras"),
        compile=False,
    )
    gater = UncertaintyGater()
    return preprocessor, lgbm, hnn, gater


@st.cache_resource
def load_shap_advisor(_preprocessor, _lgbm, _hnn):
    """Build SHAPAdvisor with feature names and background data."""
    # Extract feature names from the fitted preprocessor
    enc = _preprocessor.named_steps["encoding"]
    tfidf_features = enc.named_transformers_["tfidf"].get_feature_names_out()
    ohe_features = enc.named_transformers_["ohe"].get_feature_names_out(["Category"])
    target_features = np.array(["Subcat_te"])
    num_features = np.array(["votes", "stars", "votes_capped", "name_length", "cold_start"])
    feature_names = np.concatenate([tfidf_features, ohe_features, target_features, num_features])

    # Build background sample from the raw CSV
    path = kagglehub.dataset_download("kirilspiridonov/freelancers-offers-on-fiverr")
    csv_file = os.path.join(path, "fiverr_clean.csv")
    df = pd.read_csv(csv_file, encoding="latin-1")
    df = df.rename(columns={"ï..Category": "Category"})
    sample = df.sample(100, random_state=42).drop(columns=["price", "Unnamed: 0"])
    X_background = _preprocessor.transform(sample).toarray()

    advisor = SHAPAdvisor(_lgbm, _hnn, X_background, feature_names)
    return advisor


@st.cache_data
def load_category_subcat_mapping():
    """Returns a dict: { "Graphics & Design": ["Logo Design", ...], ... }"""
    path = kagglehub.dataset_download("kirilspiridonov/freelancers-offers-on-fiverr")
    csv_file = os.path.join(path, "fiverr_clean.csv")
    df = pd.read_csv(csv_file, encoding="latin-1")
    df = df.rename(columns={"ï..Category": "Category"})
    mapping = (
        df.groupby("Category")["Subcat"]
        .apply(lambda s: sorted(s.dropna().unique().tolist()))
        .to_dict()
    )
    return mapping


preprocessor, lgbm, hnn, gater = load_models()
shap_advisor = load_shap_advisor(preprocessor, lgbm, hnn)
cat_subcat_map = load_category_subcat_mapping()
categories = sorted(cat_subcat_map.keys())


# ════════════════════════════════════════════════════════════════
# SECTION 1 — Title
# ════════════════════════════════════════════════════════════════
st.title("💰 Fiverr Gig Price Advisor")
st.caption(
    "AI-powered pricing recommendations backed by an ensemble of models "
    "with built-in uncertainty estimation."
)
st.divider()


# ════════════════════════════════════════════════════════════════
# SECTION 2 — Gig Details (Inputs)
# ════════════════════════════════════════════════════════════════
st.subheader("📝 Gig Details")

title = st.text_input(
    "Gig Title",
    placeholder="e.g. design a professional minimalist logo",
)

col_cat, col_sub = st.columns(2)
with col_cat:
    category = st.selectbox("Category", options=categories)
with col_sub:
    subcat_options = cat_subcat_map.get(category, [])
    sub_category = st.selectbox(
        "Subcategory",
        options=subcat_options,
        index=None,
        placeholder="Type to search...",
    )

col_stars, col_votes = st.columns(2)
with col_stars:
    stars = st.slider("Rating ⭐", 0.0, 5.0, 4.5, 0.1)
with col_votes:
    votes = st.number_input(
        "Reviews", min_value=0, max_value=10000, value=50, step=10
    )

predict_btn = st.button(
    "🔍 Get Price Recommendation", type="primary", use_container_width=True
)

st.divider()


# ════════════════════════════════════════════════════════════════
# PREDICTION LOGIC (runs when button is pressed)
# ════════════════════════════════════════════════════════════════
if predict_btn and title and sub_category:
    with st.spinner("Analysing market data..."):
        # Build input DataFrame
        raw_data = pd.DataFrame(
            [
                {
                    "name": title,
                    "Category": category,
                    "Subcat": sub_category,
                    "stars": stars,
                    "votes": votes,
                }
            ]
        )

        # Preprocess → Model predictions
        X_processed = preprocessor.transform(raw_data).toarray()
        lgbm_mu = lgbm.predict(X_processed)[0]
        hnn_mu, hnn_sigma = hnn.predict(X_processed, return_std=True)

        # Gating recommendation
        result = gater.get_recommendation(lgbm_mu, hnn_mu[0], hnn_sigma[0])

        # SHAP explanations
        price_exp = shap_advisor.explain_price(X_processed)
        uncertainty_exp = shap_advisor.explain_uncertainty(X_processed)

    # ════════════════════════════════════════════════════════════
    # SECTION 3 — Predicted Price
    # ════════════════════════════════════════════════════════════
    st.subheader("💵 Predicted Price")

    price = result["price"]
    lower, upper = result["range"]

    m1, m2, m3 = st.columns(3)
    m1.metric("Lower Bound", f"${lower:,.2f}")
    m2.metric("Predicted Price", f"${price:,.2f}")
    m3.metric("Upper Bound", f"${upper:,.2f}")

    st.divider()

    # ════════════════════════════════════════════════════════════
    # SECTION 4 — Confidence Level + Advice
    # ════════════════════════════════════════════════════════════
    st.subheader("📊 Confidence & Advice")

    status = result["status"]
    if status == "GREEN_PLUS":
        st.success(
            f"🟢 **{status}** — Very High Confidence, Anchor on Predicted Price"
        )
    elif status == "GREEN":
        st.success(f"🟢 **{status}** — Stable Market, Models Agree")
    elif status == "YELLOW":
        st.warning(f"🟡 **{status}** — Volatile Market, Models Agree")
    elif status == "RED":
        if "disagree" in result["advice"].lower():
            st.error(f"🔴 **{status}** — Models Disagree")
        else:
            st.error(f"🔴 **{status}** — Extreme Market Volatility")
    else:
        st.error(f"🔴 **{status}**")

    st.markdown(f"**💡 Advice:** {result['advice']}")

    st.divider()

    # ════════════════════════════════════════════════════════════
    # SECTION 5 — SHAP: What Shaped Your Price
    # ════════════════════════════════════════════════════════════
    st.subheader("🔍 What Shaped Your Price")
    st.caption(
        "These are the features with the largest impact on your predicted price, "
        "ranked by importance."
    )

    top_drivers = price_exp["top_drivers"][:5]

    for feature_name, shap_value in top_drivers:
        human_name = _humanise(feature_name)
        impact = abs(shap_value)

        if shap_value > 0:
            st.markdown(f"▲ **{human_name}** — pushing price **up** (+{impact:.3f})")
        else:
            st.markdown(f"▼ **{human_name}** — pushing price **down** (−{impact:.3f})")

    st.divider()

    # ════════════════════════════════════════════════════════════
    # SECTION 6 — SHAP: What's Driving Uncertainty
    # ════════════════════════════════════════════════════════════
    uncertainty_drivers = [
        (f, v) for f, v in uncertainty_exp["top_drivers"][:5] if v > 0
    ]

    if uncertainty_drivers:
        st.subheader("⚠️ What's Driving Uncertainty")
        st.caption(
            "These features are making the price estimate less certain — "
            "similar gigs with these traits show wider price variation."
        )

        for feature_name, shap_value in uncertainty_drivers[:3]:
            human_name = _humanise(feature_name)
            st.markdown(f"? **{human_name}** — increasing uncertainty (+{shap_value:.3f})")

        st.divider()

    # ════════════════════════════════════════════════════════════
    # Technical Details (collapsible)
    # ════════════════════════════════════════════════════════════
    with st.expander("🔬 Technical Details"):
        detail_col1, detail_col2 = st.columns(2)
        with detail_col1:
            st.write(f"**HNN Sigma (market volatility):** {result['sigma']}")
            st.write(f"**Model Divergence:** {result['divergence']}")
        with detail_col2:
            st.write(f"**LGBM prediction (log-price):** {lgbm_mu:.4f}")
            st.write(f"**HNN prediction (log-price):** {hnn_mu[0]:.4f}")
        st.caption(
            "Sigma measures aleatoric uncertainty (inherent market price variance). "
            "Divergence measures how much the LGBM and HNN models disagree on this input."
        )

elif predict_btn and not sub_category:
    st.warning("⚠️ Please select a subcategory to get a recommendation.")
elif predict_btn and not title:
    st.warning("⚠️ Please enter a gig title to get a recommendation.")


# ── Footer ──
st.divider()
st.caption(
    "Built with LightGBM + Heteroscedastic Neural Network | "
    "Uncertainty-aware gating | SHAP explainability | Fiverr freelancer dataset"
)
