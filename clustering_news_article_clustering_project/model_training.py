"""
Model training script for clustering news articles by topic or tone using TF-IDF and KMeans.
"""

import os
import logging
import pickle
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Configuration
DATA_PATH = "data/simulated_news.csv"
MODEL_DIR = "model"
MODEL_PATH = os.path.join(MODEL_DIR, "kmeans_model.pkl")
VECTORIZER_PATH = os.path.join(MODEL_DIR, "tfidf_vectorizer.pkl")
N_CLUSTERS = 5
RANDOM_STATE = 42

def load_data(path):
    """Load dataset from a CSV file."""
    if not os.path.exists(path):
        logger.error("Dataset not found.")
        raise FileNotFoundError("Dataset file missing.")
    df = pd.read_csv(path)
    logger.info(f"Loaded dataset with shape: {df.shape}")
    return df

def preprocess_text(df):
    """Convert articles into TF-IDF vectors."""
    logger.info("Vectorizing text using TF-IDF...")
    vectorizer = TfidfVectorizer(stop_words="english")
    X = vectorizer.fit_transform(df["Article"])
    return X, vectorizer

def train_model(X):
    """Train a KMeans clustering model."""
    logger.info("Training KMeans model...")
    model = KMeans(n_clusters=N_CLUSTERS, random_state=RANDOM_STATE)
    model.fit(X)
    return model

def evaluate_model(model, X):
    """Evaluate clustering quality."""
    labels = model.predict(X)
    score = silhouette_score(X, labels)
    logger.info(f"Silhouette Score: {score:.4f}")

def save_object(obj, path):
    """Save object to file using pickle."""
    with open(path, "wb") as f:
        pickle.dump(obj, f)
    logger.info(f"Saved object to {path}")

def main():
    os.makedirs(MODEL_DIR, exist_ok=True)
    df = load_data(DATA_PATH)
    X, vectorizer = preprocess_text(df)
    model = train_model(X)
    evaluate_model(model, X)
    save_object(model, MODEL_PATH)
    save_object(vectorizer, VECTORIZER_PATH)
    logger.info("Training pipeline complete.")

if __name__ == "__main__":
    main()
