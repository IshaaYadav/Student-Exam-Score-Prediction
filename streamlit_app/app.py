# streamlit_app/app.py

import streamlit as st
import pandas as pd
import numpy as np
from joblib import load
from PIL import Image
import os

# Load model
model = load('models/student_score_model.pkl')

# Set page config
st.set_page_config(page_title="Student Score Predictor 🎓", layout="centered")

# Load visuals
pred_vs_actual = 'outputs/figures/predicted_vs_actual.png'
residuals = 'outputs/figures/residuals_distribution.png'
feature_plot = 'outputs/figures/feature_importance.png'

# Title
st.title("📚 Student Exam Score Predictor")
st.markdown("This app predicts a student’s **exam score** based on their study behavior and attendance. Try it out below!")

# Sidebar for input
st.sidebar.header("📥 Student Input")
hours = st.sidebar.slider("Hours Studied", min_value=0.0, max_value=12.0, step=0.5, value=4.0)
previous = st.sidebar.slider("Previous Exam Score", min_value=0.0, max_value=100.0, step=1.0, value=65.0)
attendance = st.sidebar.slider("Attendance Ratio", min_value=0.6, max_value=1.0, step=0.01, value=0.8)

# Make prediction
input_df = pd.DataFrame({
    'hours_studied': [hours],
    'previous_score': [previous],
    'attendance_ratio': [attendance]
})
predicted_score = model.predict(input_df)[0]

# Display prediction
st.subheader("🎯 Predicted Exam Score")
st.metric(label="Predicted Score", value=f"{predicted_score:.2f} / 100")

# Performance interpretation
if predicted_score < 50:
    st.warning("This student may need additional support and mentorship. 📉")
elif predicted_score > 85:
    st.success("Great job! This student is on track for excellent performance! 🌟")
else:
    st.info("The student is doing well, but there’s room to improve. 📈")

# Visuals
st.subheader("📊 Model Performance Visuals")

with st.expander("Predicted vs Actual"):
    if os.path.exists(pred_vs_actual):
        st.image(pred_vs_actual, use_column_width=True)
    else:
        st.info("Visual not found.")

with st.expander("Residual Distribution"):
    if os.path.exists(residuals):
        st.image(residuals, use_column_width=True)

with st.expander("Feature Importance"):
    if os.path.exists(feature_plot):
        st.image(feature_plot, use_column_width=True)

# Batch prediction
st.subheader("👩‍🏫 Teacher's Dashboard (Batch Prediction)")

uploaded_file = st.file_uploader("Upload a CSV with student data", type=['csv'])

if uploaded_file:
    try:
        df_input = pd.read_csv(uploaded_file)
        required_cols = ['hours_studied', 'previous_score', 'attendance_ratio']
        if all(col in df_input.columns for col in required_cols):
            df_input['predicted_score'] = model.predict(df_input[required_cols])
            st.write("📋 Predictions:", df_input)
            csv = df_input.to_csv(index=False).encode('utf-8')
            st.download_button("📥 Download Predictions as CSV", data=csv, file_name="student_predictions.csv", mime='text/csv')
        else:
            st.error(f"CSV must contain columns: {required_cols}")
    except Exception as e:
        st.error(f"Error reading file: {e}")
