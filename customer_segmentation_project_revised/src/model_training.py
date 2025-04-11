"""
Train clustering model (KMeans), evaluate using silhouette score, save model.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import joblib
import os

def train_model(X, n_clusters=5, model_dir="models"):
    model = KMeans(n_clusters=n_clusters, random_state=42)
    model.fit(X)
    score = silhouette_score(X, model.labels_)
    os.makedirs(model_dir, exist_ok=True)
    joblib.dump(model, f"{model_dir}/kmeans_model.pkl")
    return model, score
