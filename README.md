# Student-Exam-Score-Prediction
This Streamlit web app predicts students' exam scores using features like:
- Hours Studied
- Previous Exam Scores
- Attendance Percentage

☁️ Live App
[![View in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://student-exam-score-prediction-ishayadav.streamlit.app/)

---

## 📸 App Demo

![App Demo](demo.gif)

---

## 🚀 Features
- **Interactive Score Prediction** with real-time user input
- **Insightful Recommendations** based on predicted performance
- Clean, colorful and responsive **Streamlit UI**
- Deployed via **Streamlit Cloud**

---

## 🔍 Tech Stack

- **Frontend**: Streamlit
- **Backend/Modeling**: Scikit-learn, Pandas, NumPy
- **Deployment**: Streamlit Cloud
- **Dataset**: Student Performance Data (Kaggle)

---

## 🚀 Project Structure
Student-Exam-Score-Prediction<br>
├── data<br>
│   └── StudentsPerformance.csv   #dataset used<br>
├── models<br>
│   └── student_score_model.pkl      #Trained regression model<br>
├── notebooks<br>
│   └── EDA_Student_Performance.ipynb    #Exploratory Data Analysis and feature exploration<br>
├── outputs<br>
│   └── figures<br>
│       ├── predicted_vs_actual.png    # Prediction performance visualization<br>
│       └── residuals_distribution.png   # Model error distribution<br>
├── src<br>
│   ├── data_preprocessing.py<br>
│   ├── model_training.py<br>
│   └── model_evaluation.py<br>
├── streamlit_app<br>
│   └── app.py   # Streamlit web application<br>
├── requirements.txt<br>
├── demo.gif<br>
├── main.py     # Central runner <br>
├── LICENSE<br>
└── README.md<br>

---

🧠 Dataset
Source: Kaggle - Students Performance in Exams [!](https://www.kaggle.com/datasets/spscientist/students-performance-in-exams)

Selected Features:

Hours_Studied: Time dedicated to exam prep

Previous_Scores: Historical performance record

Attendance: Engagement and classroom presence

❗ Features like gender, race/ethnicity, lunch type, and parental education level were excluded to avoid demographic bias and focus strictly on academic predictors.

⚙️ How the Project Works <br>
🔹 1. Data Preprocessing (src/data_preprocessing.py)
Loads CSV dataset
Drops irrelevant features
Fills or removes missing values
Normalizes/standardizes numerical features
Splits data into training and testing sets

🔹 2. Model Training (src/model_training.py)
Trains a Random Forest
Saves the final trained model to /models/student_score_model.pkl using joblib

🔹 3. Model Evaluation (src/model_evaluation.py)
Evaluates model using:
MAE (Mean Absolute Error)
RMSE (Root Mean Square Error)
R² Score (Coefficient of Determination)
Saves visualizations for analysis:
📌 Predicted vs Actual Scores
📌 Residuals Distribution

🔹 4. Interactive Web App (streamlit_app/app.py)
Uses Streamlit to deploy an intuitive user interface.
Users input:
Hours studied
Previous scores
Attendance (%)
--Predicts final Exam Score

Flags if a student may require academic support, based on:
Low predicted score
Poor attendance (below threshold)

🧠 Project Insights
Attendance and consistent academic performance are critical success indicators.
Predictive models enable early identification of at-risk students.
The project supports data-driven decision making in academic settings.
 
📄 License
This project is licensed under the MIT License.


👩‍💻 Developed by: 
Isha Yadav
Btech CSE (AIML)

