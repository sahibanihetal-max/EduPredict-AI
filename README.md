# EduPredict AI — Smart Student Performance Analytics

EduPredict AI is a machine learning project that estimates student final grades using historical student performance data. It combines predictive modelling with an interactive dashboard to explore educational data and identify useful study insights.

## Features
- Student final-grade prediction using machine learning
- Random Forest and Extra Trees model comparison
- Interactive dashboard built with Streamlit
- Data visualisation using Plotly
- Student data filtering and CSV export
- Study suggestions based on entered information

## Tech Stack
- Python
- Pandas and NumPy
- Scikit-learn
- Streamlit
- Plotly
- Joblib

## Dataset
UCI Student Performance Dataset — Mathematics records.

## Run Locally

1. Clone or download this repository.
2. Install dependencies:

   `pip install -r requirements.txt`

3. Train the model:

   `python train_model.py`

4. Launch the dashboard:

   `streamlit run app.py`

## Project Purpose
This project demonstrates data preprocessing, regression modelling, model evaluation, and interactive data visualisation.

## Disclaimer
Predictions are estimates based on historical data. They are not guaranteed results and should not be used to judge a student's ability or make high-stakes decisions.