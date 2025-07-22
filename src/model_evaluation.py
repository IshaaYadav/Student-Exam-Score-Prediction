# src/model_evaluation.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, r2_score
from joblib import load
import os

def evaluate_model(model_path='models/student_score_model.pkl', df=None):
    model = load(model_path)
    
    X = df.drop(columns=['average_score'])
    y = df['average_score']
    y_pred = model.predict(X)

    rmse = np.sqrt(mean_squared_error(y, y_pred))
    r2 = r2_score(y, y_pred)

    os.makedirs('outputs/figures', exist_ok=True)

    # Prediction vs Actual
    plt.figure(figsize=(8,5))
    sns.scatterplot(x=y, y=y_pred, color='mediumpurple', s=40)
    plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')
    plt.xlabel("Actual Average Score")
    plt.ylabel("Predicted Score")
    plt.title("Predicted vs Actual Scores")
    plt.savefig('outputs/figures/predicted_vs_actual.png')
    plt.close()

    # Residuals
    residuals = y - y_pred
    plt.figure(figsize=(8,5))
    sns.histplot(residuals, bins=30, kde=True, color='coral')
    plt.title("Residuals Distribution")
    plt.xlabel("Prediction Error")
    plt.ylabel("Frequency")
    plt.savefig('outputs/figures/residuals_distribution.png')
    plt.close()

    return rmse, r2
