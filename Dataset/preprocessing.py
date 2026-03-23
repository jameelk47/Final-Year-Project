import kagglehub
import os
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, PowerTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from category_encoders import TargetEncoder
from sklearn.model_selection import train_test_split

# Download fiverr dataset
path = kagglehub.dataset_download("kirilspiridonov/freelancers-offers-on-fiverr")


print("Path to dataset files:", path)
csv_file = os.path.join(path, "fiverr_clean.csv")  
df = pd.read_csv(csv_file, encoding='latin-1')
print(df.info())
df = df.rename(columns={'ï..Category': 'Category'})
mapping = df.groupby('Category')['Subcat'].apply(lambda s: sorted(s.dropna().unique().tolist())).to_dict()
for cat, subcats in sorted(mapping.items()):
    print(f'\n{cat}:')
    for s in subcats:
        print(f'  - {s}')


# Clean votes, stars, price and add votes_capped flag, cold_start flag
class FiverrPreProcessor(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X = X.copy()
        X['votes_capped'] = (X['votes'] == '1k+').astype(int)
        X['votes'] = X['votes'].replace('1k+', '1000')
        X['cold_start'] = ((X['stars'] == 'nul') & (X['votes'] == 'nul')).astype(int)
        X['votes'] = X['votes'].replace('nul', '0').astype(float)
        X['stars'] = X['stars'].replace('nul', '0').astype(float)
        X['name_length'] = X['name'].str.len()
        return X

# Define the column groups
text_col = 'name'
cat_col = ['Category']
target_enc_col = ['Subcat']
num_cols = ['votes', 'stars', 'votes_capped', 'name_length', 'cold_start']

num_pipeline = Pipeline([
    ('power_trans', PowerTransformer(method='yeo-johnson')), # Handles the skew
    ('scaler', StandardScaler())                             # Centers at 0
])

# Build the processing steps
preprocessor = ColumnTransformer(
    transformers=[
        ('tfidf', TfidfVectorizer(max_features=300, stop_words='english'), text_col),
        ('ohe', OneHotEncoder(handle_unknown='ignore'), cat_col),
        ('target_enc', TargetEncoder(), target_enc_col),
        ('num_pipeline', num_pipeline, num_cols)
    ]
)

# Final Pipeline Object
fiverr_dss_pipeline = Pipeline([
    ('cleaning', FiverrPreProcessor()),
    ('encoding', preprocessor)
])

df['price'] = (
    df['price']
      .str.replace(',', '', regex=False)   # "2,222" -> "2222"
      .str.replace('$', '', regex=False)   # remove currency symbols if present
      .astype(float)
)

df['votes_clean'] = df['votes'].replace('1k+', '1000').replace('nul', '0').astype(float)


X = df.drop(columns=['price', 'Unnamed: 0'])
y = np.log1p(df['price'])
    
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Now fit ONLY on training data
fitted_pipeline = fiverr_dss_pipeline.fit(X_train, y_train)


# ── Price variance by subcategory ──
subcat_stats = (
    df.groupby("Subcat")["price"]
    .agg(["mean", "std", "count"])
    .sort_values("std")
)

print("\n\n════════════════════════════════════════════════")
print("  Subcategory Price Variance (sorted by std dev)")
print("════════════════════════════════════════════════\n")

print("NARROWEST price ranges (low std → model should be most confident):\n")
print(subcat_stats.head(10).to_string())

print("\n\nWIDEST price ranges (high std → model will show wider bands):\n")
print(subcat_stats.tail(10).to_string())
