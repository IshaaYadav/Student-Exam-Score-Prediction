# src/model_training.py

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from joblib import dump

def train_and_select_model(df, random_state=42):
    X = df.drop(columns=['average_score'])
    y = df['average_score']
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_state)

    # Models to compare
    models = {
        'LinearRegression': LinearRegression(),
        'Ridge': Ridge(alpha=1.0),
        'Lasso': Lasso(alpha=0.1),
        'RandomForest': RandomForestRegressor(n_estimators=100, random_state=random_state)
    }

    results = []

    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)

        results.append({
            'Model': name,
            'MAE': mae,
            'RMSE': rmse,
            'R2': r2
        })

    # Convert to DataFrame
    results_df = pd.DataFrame(results).sort_values(by='R2', ascending=False)
    
    # Save best model
    best_model_name = results_df.iloc[0]['Model']
    best_model = models[best_model_name]
    best_model.fit(X, y)
    
    dump(best_model, 'models/student_score_model.pkl')
    
    return results_df, best_model_name
