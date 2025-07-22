# Student-Exam-Score-Prediction
This project predicts students' exam scores based on key academic performance indicators: hours studied, previous scores, and attendance. The aim is to provide actionable insights that can help identify students who may require additional academic support, while also understanding how different factors impact performance.


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
├── main.py     # Central runner <br>
├── LICENSE<br>
└── README.md<br>

---

🧠 Dataset
Source: Kaggle - Students Performance in Exams [](https://www.kaggle.com/datasets/spscientist/students-performance-in-exams)

Selected Features:

Hours_Studied: Time dedicated to exam prep

Previous_Scores: Historical performance record

Attendance: Engagement and classroom presence

❗ Features like gender, race/ethnicity, lunch type, and parental education level were excluded to avoid demographic bias and focus strictly on academic predictors.

⚙️ How the Project Works
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

📷 Output Visuals
<img width="1920" height="1008" alt="Screenshot 2025-07-22 223720" src="https://github.com/user-attachments/assets/34fa3966-db0e-4c80-8b17-4eaa76346d0d" />
<img width="1920" height="1008" alt="image" src="https://github.com/user-attachments/assets/a1ba3e6b-8279-4fb7-9fce-d9bf44517ff3" />
<img width="1920" height="1008" alt="image" src="https://github.com/user-attachments/assets/b5770f58-eadc-462d-977e-879eb9dff5ef" />
<img width="1920" height="1008" alt="image" src="https://github.com/user-attachments/assets/ea6deffa-7a77-4e94-a3bc-ef828eee9ad0" />


📄 License
This project is licensed under the MIT License.


👩‍💻 Developed by: 
Isha Yadav
Btech CSE (AIML)

