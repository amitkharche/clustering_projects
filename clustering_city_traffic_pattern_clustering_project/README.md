# 🚦 City Traffic Pattern Clustering Project

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.22.0-red)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.2.2-orange)

## 📌 Business Use Case

Urban planners and transportation companies need to optimize routes and reduce congestion. By clustering traffic patterns based on time, location, volume, and weather, cities can:
- Design efficient traffic signal timings
- Plan better public transport schedules
- Reduce travel time and emissions

## 🧠 Features Used
- Avg Speed
- Vehicle Count
- Time of Day, Day of Week
- Weather Condition
- Latitude & Longitude

## 🧪 Pipeline Steps

### Training (`model_training.py`)
- Load traffic data
- Preprocess using OneHotEncoding + StandardScaling
- Train KMeans clustering model
- Evaluate with silhouette score
- Save model and preprocessor

### App (`app.py`)
- Upload new traffic data
- Predict cluster labels
- Visualize clusters via PCA
- Download results as CSV

## 🚀 How to Use

1. Install packages
```bash
pip install -r requirements.txt
```

2. Run model training
```bash
python model_training.py
```

3. Launch the app
```bash
streamlit run app.py
```

## 📁 Project Structure
```
city_traffic_pattern_clustering_project/
├── data/                       # Simulated traffic data
├── model/                      # Saved model and preprocessor
├── app.py                      # Streamlit app
├── model_training.py           # Training pipeline
├── README.md                   # Documentation
├── requirements.txt            # Dependencies
└── .github/ISSUE_TEMPLATE/     # Issue templates
```

## 📄 License
This project is licensed under the MIT License.
