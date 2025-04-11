# 🗞️ News Article Clustering Project

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.22.0-red)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.2.2-orange)

## 📌 Business Use Case

This project demonstrates how to group news articles based on their content using unsupervised learning (clustering). This can be used in:

- News personalization systems
- Media monitoring
- Topic tracking and categorization

## 🧠 Features Used

- Text content (`Article`) converted using **TF-IDF Vectorization**
- Clustering done using **KMeans**

## 🧪 Pipeline Steps

### Training (`model_training.py`)
- Load and clean simulated news articles
- Convert text to TF-IDF vectors
- Cluster articles using KMeans
- Evaluate with silhouette score
- Save model and vectorizer

### Inference (`app.py`)
- Upload CSV of articles
- Predict clusters using trained model
- Visualize with PCA
- Download results

## 🚀 How to Use

1. **Install dependencies**
```bash
pip install -r requirements.txt
```

2. **Run model training**
```bash
python model_training.py
```

3. **Launch the app**
```bash
streamlit run app.py
```

## 🗂 Project Structure
```
news_article_clustering_project/
├── data/
│   └── simulated_news.csv
├── model/
│   └── kmeans_model.pkl
│   └── tfidf_vectorizer.pkl
├── app.py
├── model_training.py
├── requirements.txt
├── README.md
└── .github/ISSUE_TEMPLATE/
```

## 📄 License

This project is licensed under the MIT License.
