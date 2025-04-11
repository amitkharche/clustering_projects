"""
Streamlit app for clustering news articles by topic or tone using KMeans and TF-IDF.
"""

import streamlit as st
import pandas as pd
import pickle
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="🗞️ News Article Clustering", layout="wide")
st.title("🗞️ Cluster News Articles by Topic or Tone")

@st.cache_resource
def load_model():
    try:
        with open("model/kmeans_model.pkl", "rb") as f:
            model = pickle.load(f)
        with open("model/tfidf_vectorizer.pkl", "rb") as f:
            vectorizer = pickle.load(f)
        return model, vectorizer
    except Exception as e:
        st.error(f"Error loading model or vectorizer: {e}")
        return None, None

def run_clustering(df, model, vectorizer):
    tfidf = vectorizer.transform(df["Article"])
    labels = model.predict(tfidf)
    df["Cluster"] = labels

    pca = PCA(n_components=2)
    reduced = pca.fit_transform(tfidf.toarray())
    df["PCA1"], df["PCA2"] = reduced[:, 0], reduced[:, 1]
    return df

def plot_clusters(df):
    fig, ax = plt.subplots()
    sns.scatterplot(data=df, x="PCA1", y="PCA2", hue="Cluster", palette="tab10", s=100, ax=ax)
    ax.set_title("Article Clusters (PCA View)")
    st.pyplot(fig)

def main():
    uploaded_file = st.file_uploader("Upload a CSV with 'Article' column", type="csv")
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        if "Article" not in df.columns:
            st.error("CSV must contain an 'Article' column.")
            return
        st.subheader("Uploaded Articles")
        st.dataframe(df.head())

        model, vectorizer = load_model()
        if model and vectorizer:
            df = run_clustering(df, model, vectorizer)
            st.subheader("Clustered Articles")
            st.dataframe(df[["Article", "Cluster"]].head())

            plot_clusters(df)

            csv = df.to_csv(index=False).encode("utf-8")
            st.download_button("Download Clustered Articles", csv, "clustered_articles.csv", "text/csv")

if __name__ == "__main__":
    main()
