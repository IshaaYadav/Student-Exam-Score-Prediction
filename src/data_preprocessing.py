# src/data_preprocessing.py

import pandas as pd
import numpy as np

def load_and_prepare_data(file_path=r"C:\Users\ISHA\OneDrive\Documents\GitHub\Student-Exam-Score-Prediction\data\StudentsPerformance.csv", seed=42):
    # Load dataset
    df = pd.read_csv(file_path)
    
    # Create 'average_score' as the target
    df['average_score'] = df[['math score', 'reading score', 'writing score']].mean(axis=1)

    # Simulate 'hours_studied' (students with higher scores likely studied more)
    np.random.seed(seed)
    df['hours_studied'] = (df['average_score'] / 10) + np.random.normal(0, 1.5, size=len(df))
    df['hours_studied'] = df['hours_studied'].clip(lower=1).round(1)

    # Simulate 'attendance_ratio' (scaled between 0.6 to 1.0, correlated with score)
    df['attendance_ratio'] = (df['average_score'] / 100) * 0.4 + 0.6
    df['attendance_ratio'] = df['attendance_ratio'].round(2)

    # Simulate 'previous_score' (average_score minus random noise)
    df['previous_score'] = df['average_score'] - np.random.normal(5, 5, size=len(df))
    df['previous_score'] = df['previous_score'].clip(lower=0, upper=100).round(1)

    # Select final columns for modeling
    df_model = df[[
        'hours_studied',
        'previous_score',
        'attendance_ratio',
        'average_score'
    ]]

    return df_model
