"""
Streamlit app for city traffic pattern clustering using KMeans.
"""

import streamlit as st
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA

st.set_page_config(page_title="🚦 City Traffic Clustering", layout="wide")
st.title("🚦 City Traffic Pattern Clustering App")

@st.cache_resource
def load_model():
    try:
        with open("model/kmeans_model.pkl", "rb") as f:
            model = pickle.load(f)
        with open("model/preprocessor.pkl", "rb") as f:
            preprocessor = pickle.load(f)
        return model, preprocessor
    except Exception as e:
        st.error(f"Error loading model or preprocessor: {e}")
        return None, None

def process_and_predict(df, model, preprocessor):
    X = preprocessor.transform(df)
    df["Cluster"] = model.predict(X)

    pca = PCA(n_components=2)
    reduced = pca.fit_transform(X.toarray() if hasattr(X, "toarray") else X)
    df["PCA1"], df["PCA2"] = reduced[:, 0], reduced[:, 1]
    return df

def plot_clusters(df):
    fig, ax = plt.subplots()
    sns.scatterplot(data=df, x="PCA1", y="PCA2", hue="Cluster", palette="tab10", s=100, ax=ax)
    ax.set_title("Traffic Pattern Clusters (PCA View)")
    st.pyplot(fig)

def main():
    uploaded_file = st.file_uploader("Upload a CSV with traffic data", type="csv")
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.subheader("Uploaded Data")
        st.dataframe(df.head())

        model, preprocessor = load_model()
        if model and preprocessor:
            df = process_and_predict(df, model, preprocessor)
            st.subheader("Clustered Output")
            st.dataframe(df[["Avg_Speed", "Vehicle_Count", "Cluster"]].head())
            plot_clusters(df)

            csv = df.to_csv(index=False).encode("utf-8")
            st.download_button("Download Clustered Results", csv, "clustered_traffic.csv", "text/csv")

if __name__ == "__main__":
    main()
