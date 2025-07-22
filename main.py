# main.py

from src.data_preprocessing import load_and_prepare_data
from src.model_training import train_and_select_model
from src.model_evaluation import evaluate_model

def main():
    print("🔄 Loading and preparing data...")
    df = load_and_prepare_data()
    print("✅ Data loaded and enhanced with new features.")
    print("\n📄 Sample Data:")
    print(df.head())

    print("\n🚀 Training models and selecting the best one...")
    results_df, best_model = train_and_select_model(df)
    print("\n📊 Model Performance Summary:")
    print(results_df)

    print(f"\n✅ Best performing model saved: {best_model}")

    print("\n📈 Evaluating and generating model visualizations...")
    rmse, r2 = evaluate_model(df=df)
    print(f"📉 RMSE: {rmse:.2f}")
    print(f"📈 R² Score: {r2:.2f}")

    print("\n🎉 All done! You're ready to launch the Streamlit app.")
    print("👉 Run it using: streamlit run streamlit_app/app.py")

if __name__ == "__main__":
    main()
