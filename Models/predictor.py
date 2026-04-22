import pandas as pd
import joblib
import numpy as np
import tensorflow as tf
from Models.gating import UncertaintyGater
from Models.hnn import HeteroscedasticKerasRegressor, aleatoric_loss


my_gater = UncertaintyGater()

class FiverrPricePredictor:
    def __init__(self, preprocessor, lgbm_model, hnn_model, gater):
        self.preprocessor = preprocessor
        self.lgbm = lgbm_model
        self.hnn = hnn_model
        self.gater = gater

    def predict_strategy(self, title, category, sub_category, stars, votes):
        # 1. Transform raw data to 300+ features

        raw_data = pd.DataFrame([{
            'name': title,
            'Category': category,
            'Subcat': sub_category,
            'stars': stars,
            'votes': votes
        }])

        X_processed = self.preprocessor.transform(raw_data).toarray()
        
        # 2. Get Model Outputs
        lgbm_mu = self.lgbm.predict(X_processed)[0]
        hnn_mu, hnn_sigma = self.hnn.predict(X_processed, return_std=True)
        
        # 3. Pass to your validated Gating Logic
        return self.gater.get_recommendation(lgbm_mu, hnn_mu[0], hnn_sigma[0])

loaded_preprocessor = joblib.load('preprocessor.pkl')
loaded_lgbm = joblib.load('lgbm_model.pkl')
loaded_hnn = joblib.load('hnn_wrapper.pkl')

# Load the entire Keras model (custom loss is automatically resolved via the decorator)
loaded_hnn.model_ = tf.keras.models.load_model("hnn_weights.keras", compile=False)

final_predictor = FiverrPricePredictor(
    preprocessor=loaded_preprocessor,
    lgbm_model=loaded_lgbm,
    hnn_model=loaded_hnn,
    gater=my_gater
)

