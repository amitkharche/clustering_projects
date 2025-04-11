"""
Model training script for city traffic pattern clustering to optimize transport routes.
"""

import os
import logging
import pickle
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# Logging setup
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Configuration
DATA_PATH = "data/simulated_city_traffic.csv"
MODEL_DIR = "model"
MODEL_FILE = os.path.join(MODEL_DIR, "kmeans_model.pkl")
PREPROCESSOR_FILE = os.path.join(MODEL_DIR, "preprocessor.pkl")
N_CLUSTERS = 5
RANDOM_STATE = 42

def load_data(path):
    """Load traffic data from CSV."""
    if not os.path.exists(path):
        logger.error("Dataset not found.")
        raise FileNotFoundError("Dataset is missing.")
    df = pd.read_csv(path)
    logger.info(f"Loaded dataset with shape: {df.shape}")
    return df

def preprocess_data():
    """Create preprocessing pipeline."""
    numeric_features = ["Avg_Speed", "Vehicle_Count", "Latitude", "Longitude"]
    categorical_features = ["Time_of_Day", "Day_of_Week", "Weather_Condition"]

    numeric_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(drop="first", handle_unknown="ignore")

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features)
        ]
    )
    return preprocessor

def train_model(X):
    """Train KMeans clustering model."""
    model = KMeans(n_clusters=N_CLUSTERS, random_state=RANDOM_STATE)
    model.fit(X)
    return model

def evaluate_model(model, X):
    """Evaluate clustering model."""
    labels = model.predict(X)
    score = silhouette_score(X, labels)
    logger.info(f"Silhouette Score: {score:.4f}")

def save_pickle(obj, path):
    """Save object to disk."""
    with open(path, "wb") as f:
        pickle.dump(obj, f)
    logger.info(f"Saved object to {path}")

def main():
    os.makedirs(MODEL_DIR, exist_ok=True)
    df = load_data(DATA_PATH)

    preprocessor = preprocess_data()
    X_processed = preprocessor.fit_transform(df)

    model = train_model(X_processed)
    evaluate_model(model, X_processed)

    save_pickle(model, MODEL_FILE)
    save_pickle(preprocessor, PREPROCESSOR_FILE)
    logger.info("Training pipeline completed.")

if __name__ == "__main__":
    main()
