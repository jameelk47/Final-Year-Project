import kagglehub
import os
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from category_encoders import TargetEncoder


# Download fiverr dataset
path = kagglehub.dataset_download("kirilspiridonov/freelancers-offers-on-fiverr")

print("Path to dataset files:", path)

csv_file = os.path.join(path, "fiverr_clean.csv")  

df = pd.read_csv(csv_file, encoding='latin-1')

print(df.info())


# Clean votes, stars, price and add votes_capped flag
class FiverrPreProcessor(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X = X.copy()
        X['votes_capped'] = (X['votes'] == '1k+').astype(int)
        
        X['votes'] = X['votes'].replace('1k+', '1000').astype(float)
        
        X['stars'] = X['stars'].astype(float)
        
        return X

# Define the column groups
text_col = 'name'
cat_col = ['ï..Category	']
target_enc_col = ['Subcat']
num_cols = ['votes', 'stars', 'votes_capped']

# Build the processing steps
preprocessor = ColumnTransformer(
    transformers=[
        ('tfidf', TfidfVectorizer(max_features=500, stop_words='english'), text_col),
        ('ohe', OneHotEncoder(handle_unknown='ignore'), cat_col),
        ('target_enc', TargetEncoder(), target_enc_col),
        ('scaler', StandardScaler(), num_cols)
    ]
)

# Final Pipeline Object
fiverr_dss_pipeline = Pipeline([
    ('cleaning', FiverrPreProcessor()),
    ('encoding', preprocessor)
])

X = df.drop(columns=['price', 'Unnamed: 0'])

y = np.log1p(df['price']).astype(float)
    



