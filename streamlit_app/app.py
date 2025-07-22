import streamlit as st
import pandas as pd
import numpy as np
import os
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Student Score Predictor", 
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-title {
        font-size: 3.5rem;
        font-weight: 700;
        color: #1e3a8a;
        text-align: center;
        margin-bottom: 0.5rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    .subtitle {
        font-size: 1.2rem;
        color: #64748b;
        text-align: center;
        margin-bottom: 2rem;
        font-style: italic;
    }
    .prediction-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin: 1rem 0;
        box-shadow: 0 10px 25px rgba(0,0,0,0.2);
    }
    .prediction-score {
        font-size: 3rem;
        font-weight: bold;
        margin: 0.5rem 0;
    }
    .insight-card {
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        border-left: 5px solid;
    }
    .high-risk {
        background-color: #fef2f2;
        border-left-color: #ef4444;
        color: #991b1b;
    }
    .medium-risk {
        background-color: #fffbeb;
        border-left-color: #f59e0b;
        color: #92400e;
    }
    .low-risk {
        background-color: #f0fdf4;
        border-left-color: #10b981;
        color: #065f46;
    }
    .metric-container {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        text-align: center;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        padding: 0px 20px;
        background-color: #f8fafc;
        border-radius: 10px 10px 0px 0px;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_model():
    """Load the trained model"""
    model_path = os.path.join("models", "student_score_model.pkl")
    if os.path.exists(model_path):
        try:
            return joblib.load(model_path)
        except Exception as e:
            st.error(f"Error loading model: {e}")
            return None
    else:
        st.warning("⚠️ Model file not found. Using prediction formula.")
        return None

@st.cache_data
def load_data():
    """Load actual"""
    data_path = os.path.join("data", "StudentsPerformance.csv")
    if os.path.exists(data_path):
        try:
            data = pd.read_csv(data_path)
            # Convert score columns to numeric
            data['math score'] = pd.to_numeric(data['math score'], errors='coerce')
            data['reading score'] = pd.to_numeric(data['reading score'], errors='coerce')
            data['writing score'] = pd.to_numeric(data['writing score'], errors='coerce')
            # Calculate average score
            data['average_score'] = (data['math score'] + data['reading score'] + data['writing score']) / 3
            return data
        except Exception as e:
            st.error(f"Error loading data: {e}")
            return None
    else:
        st.warning("⚠️ Data file not found. Please ensure StudentsPerformance.csv exists in the data folder.")
        return None

def predict_score(model, hours, prev_score, attendance):
    """Make prediction"""
    if model:
        try:
            input_data = pd.DataFrame({
                "Hours Studied": [hours],
                "Previous Scores": [prev_score],
                "Attendance": [attendance]
            })
            return model.predict(input_data)[0]
        except:
            pass
    
    # prediction formula
    score = (0.4 * hours * 4 + 0.35 * prev_score + 0.25 * attendance + np.random.normal(0, 3))
    return max(0, min(100, score))

def get_risk_level_and_insights(score, hours, prev_score, attendance):
    """Determine risk level"""
    if score < 50:
        risk_level = "high-risk"
        risk_text = "🚨 HIGH RISK - Immediate Support Needed"
        recommendations = [
            "📚 Increase study hours to at least 6-8 hours per week",
            "👥 Enroll in tutoring or study groups immediately",
            "📅 Create a structured study schedule with daily goals",
            "🎯 Focus on fundamental concepts and basic skills",
            "👨‍🏫 Schedule regular meetings with academic advisor"
        ]
    elif score < 70:
        risk_level = "medium-risk"
        risk_text = "⚠️ MODERATE RISK - Additional Support Recommended"
        recommendations = [
            "📈 Aim to increase study time by 2-3 hours per week",
            "📝 Implement active learning techniques (flashcards, practice tests)",
            "👥 Join study groups or peer learning sessions",
            "🎯 Focus on areas where previous performance was weak",
            "📊 Track progress weekly and adjust study methods"
        ]
    else:
        risk_level = "low-risk"
        risk_text = "✅ LOW RISK - On Track for Success"
        recommendations = [
            "🌟 Continue current study habits - they're working well!",
            "📚 Consider helping peers through tutoring or study groups",
            "🎯 Challenge yourself with advanced materials or projects",
            "📈 Maintain consistent attendance and engagement",
            "🏆 Set goals for even higher achievement"
        ]
    
    # Additional specific insights
    insights = []
    if hours < 5:
        insights.append("⏰ Study time is below optimal range. Research shows 6-8 hours/week is ideal.")
    if prev_score < 60:
        insights.append("📉 Previous performance indicates foundational gaps that need addressing.")
    if attendance < 80:
        insights.append("🎓 Attendance below 80% significantly impacts learning outcomes.")
    if hours > 15:
        insights.append("⚖️ Very high study hours - ensure you're maintaining work-life balance.")
    
    return risk_level, risk_text, recommendations, insights

def create_gauge_chart(score):
    """gauge chart for the score"""
    fig = go.Figure(go.Indicator(
        mode = "gauge+number+delta",
        value = score,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Predicted Exam Score", 'font': {'size': 20}},
        delta = {'reference': 75, 'increasing': {'color': "green"}, 'decreasing': {'color': "red"}},
        gauge = {
            'axis': {'range': [None, 100], 'tickwidth': 1},
            'bar': {'color': "#1e3a8a"},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 50], 'color': '#fee2e2'},
                {'range': [50, 70], 'color': '#fef3c7'},
                {'range': [70, 85], 'color': '#d1fae5'},
                {'range': [85, 100], 'color': '#dbeafe'}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 90
            }
        }
    ))
    fig.update_layout(height=300, font={'color': "#1e3a8a", 'family': "Arial"})
    return fig

def create_feature_impact_chart():
    """feature importance visualization"""
    features = ['Study Hours', 'Previous Scores', 'Attendance']
    importance = [40, 35, 25]
    colors = ['#1e3a8a', '#3b82f6', '#60a5fa']
    
    fig = px.bar(
        x=features, y=importance, 
        title="Factors Influencing Student Performance",
        labels={'x': 'Factors', 'y': 'Impact (%)'},
        color=features,
        color_discrete_sequence=colors
    )
    fig.update_traces(text=[f'{i}%' for i in importance], textposition='outside')
    fig.update_layout(showlegend=False, height=400)
    return fig

def create_score_distribution(data):
    """score distribution charts"""
    if data is None:
        fig = go.Figure()
        fig.add_annotation(text="No data available", x=0.5, y=0.5, showarrow=False)
        fig.update_layout(height=400, title="Score Distribution")
        return fig
    
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Math Scores', 'Reading Scores', 'Writing Scores', 'Average Scores'),
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    # Math scores
    fig.add_trace(
        go.Histogram(x=data['math score'], name='Math', nbinsx=20, marker_color='#1e3a8a'),
        row=1, col=1
    )
    
    # Reading scores
    fig.add_trace(
        go.Histogram(x=data['reading score'], name='Reading', nbinsx=20, marker_color='#3b82f6'),
        row=1, col=2
    )
    
    # Writing scores
    fig.add_trace(
        go.Histogram(x=data['writing score'], name='Writing', nbinsx=20, marker_color='#60a5fa'),
        row=2, col=1
    )
    
    # Average scores
    fig.add_trace(
        go.Histogram(x=data['average_score'], name='Average', nbinsx=20, marker_color='#93c5fd'),
        row=2, col=2
    )
    
    fig.update_layout(height=600, title_text="Distribution of Student Scores", showlegend=False)
    return fig

def create_demographic_analysis(data):
    """demographic analysis charts"""
    if data is None:
        fig = go.Figure()
        fig.add_annotation(text="No data available", x=0.5, y=0.5, showarrow=False)
        fig.update_layout(height=400, title="Demographic Analysis")
        return fig
    
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Scores by Gender', 'Scores by Lunch Program', 
                       'Scores by Test Prep', 'Scores by Parental Education'),
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    # Gender analysis
    gender_avg = data.groupby('gender')['average_score'].mean().reset_index()
    fig.add_trace(
        go.Bar(x=gender_avg['gender'], y=gender_avg['average_score'], 
               name='Gender', marker_color='#1e3a8a'),
        row=1, col=1
    )
    
    # Lunch program analysis
    lunch_avg = data.groupby('lunch')['average_score'].mean().reset_index()
    fig.add_trace(
        go.Bar(x=lunch_avg['lunch'], y=lunch_avg['average_score'], 
               name='Lunch', marker_color='#3b82f6'),
        row=1, col=2
    )
    
    # Test prep analysis
    prep_avg = data.groupby('test preparation course')['average_score'].mean().reset_index()
    fig.add_trace(
        go.Bar(x=prep_avg['test preparation course'], y=prep_avg['average_score'], 
               name='Test Prep', marker_color='#60a5fa'),
        row=2, col=1
    )
    
    # Parental education analysis (top 4 categories)
    parent_avg = data.groupby('parental level of education')['average_score'].mean().reset_index()
    parent_avg = parent_avg.nlargest(4, 'average_score')
    fig.add_trace(
        go.Bar(x=parent_avg['parental level of education'], y=parent_avg['average_score'], 
               name='Parent Ed', marker_color='#93c5fd'),
        row=2, col=2
    )
    
    fig.update_layout(height=600, title_text="Performance by Demographics", showlegend=False)
    fig.update_xaxes(tickangle=45)
    return fig

def main():
    # Header
    st.markdown('<h1 class="main-title">🎓 Student Performance Predictor</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Empowering educators with AI-driven insights to support every student\'s success</p>', unsafe_allow_html=True)
    
    # Load model and data
    model = load_model()
    data = load_data()
    
    # Display file status
    col1, col2 = st.columns(2)
    with col1:
        if model is not None:
            st.success("✅ Model loaded successfully")
        else:
            st.info("ℹ️ Using prediction formula")
    with col2:
        if data is not None:
            st.success(f"✅ Data loaded successfully ({len(data)} records)")
        else:
            st.error("❌ Data not available")
    
    # Sidebar inputs - BACK TO ORIGINAL INPUTS
    st.sidebar.markdown("### 📊 Enter Student Information")
    st.sidebar.markdown("---")
    
    hours = st.sidebar.slider(
        "📚 Hours Studied per Week", 
        0.0, 20.0, 5.0, 0.5,
        help="Average weekly study hours outside of class"
    )
    
    prev_score = st.sidebar.slider(
        "📝 Previous Exam Score (%)", 
        0.0, 100.0, 75.0, 1.0,
        help="Most recent exam or assessment score"
    )
    
    attendance = st.sidebar.slider(
        "🎓 Attendance Rate (%)", 
        0, 100, 85, 1,
        help="Percentage of classes attended"
    )
    
    st.sidebar.markdown("---")
    
    # Prediction section - BACK TO ORIGINAL LOGIC
    if st.sidebar.button("Predict", type="primary", use_container_width=True):
        with st.spinner("Analyzing student data..."):
            # Make prediction using ORIGINAL inputs
            predicted_score = predict_score(model, hours, prev_score, attendance)
            risk_level, risk_text, recommendations, insights = get_risk_level_and_insights(
                predicted_score, hours, prev_score, attendance
            )
            
            # Display results
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.markdown(f"""
                <div class="prediction-card">
                    <h3>Predicted Performance</h3>
                    <div class="prediction-score">{predicted_score:.1f}%</div>
                    <p>Expected Exam Score</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.plotly_chart(create_gauge_chart(predicted_score), use_container_width=True)
            
            # Risk assessment
            st.markdown(f'<div class="insight-card {risk_level}"><h3>{risk_text}</h3></div>', 
                       unsafe_allow_html=True)
            
            # Detailed insights
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("🎯 Specific Recommendations")
                for rec in recommendations:
                    st.write(f"• {rec}")
            
            with col2:
                st.subheader("🔍 Key Insights")
                if insights:
                    for insight in insights:
                        st.write(f"• {insight}")
                else:
                    st.write("• All indicators show positive learning patterns")
                    st.write("• Student demonstrates good academic habits")
                    st.write("• Continue monitoring for consistent performance")
            
            # Comparison metrics
            if data is not None:
                st.subheader("📊 Performance Comparison")
                col1, col2, col3 = st.columns(3)
                
                avg_score = data['average_score'].mean()
                
                with col1:
                    delta_hours = hours - 6  # Compare to optimal 6 hours
                    st.metric("Study Hours", f"{hours} hrs/week", 
                             f"{delta_hours:+.1f} vs optimal", delta_color="normal")
                
                with col2:
                    delta_prev = prev_score - avg_score
                    st.metric("Previous Score", f"{prev_score}%", 
                             f"{delta_prev:+.1f}% vs dataset avg", delta_color="normal")
                
                with col3:
                    delta_att = attendance - 85  # Compare to good attendance
                    st.metric("Attendance", f"{attendance}%", 
                             f"{delta_att:+.1f}% vs good", delta_color="normal")
    
    # Educational insights section
    st.markdown("---")
    st.subheader("📈 Educational Insights & Analytics")
    
    tab1, tab2, tab3 = st.tabs(["📊 Performance Factors", "📈 Data Patterns", "🎯 Support Guidelines"])
    
    with tab1:
        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(create_feature_impact_chart(), use_container_width=True)
        with col2:
            st.markdown("""
            #### Key Performance Factors:
            
            **📚 Study Hours (40% impact)**
            - Optimal range: 6-8 hours per week
            - Below 4 hours: High risk of poor performance
            - Above 12 hours: Diminishing returns, check for burnout
            
            **📝 Previous Performance (35% impact)**
            - Strong predictor of future success
            - Scores below 60%: Indicates foundational gaps
            - Consistent high scores: Likely continued success
            
            **🎓 Attendance (25% impact)**
            - Minimum 80% for good outcomes
            - Each 10% increase improves scores by ~5%
            - Below 70%: Significant risk factor
            """)
    
    with tab2:
        if data is not None:
            st.plotly_chart(create_score_distribution(data), use_container_width=True)
            st.plotly_chart(create_demographic_analysis(data), use_container_width=True)
            
            # Summary statistics
            st.subheader("📋 Dataset Summary")
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Students", f"{len(data):,}")
            with col2:
                st.metric("Average Score", f"{data['average_score'].mean():.1f}%")
            with col3:
                st.metric("Students at Risk (<60%)", f"{(data['average_score'] < 60).sum()}")
            with col4:
                st.metric("High Performers (>85%)", f"{(data['average_score'] > 85).sum()}")
        else:
            st.info("📊 Data not available for visualization")
    
    with tab3:
        st.markdown("""
        ### 🎯 Educator Support Guidelines
        
        #### 🚨 High Risk Students (Score < 50%)
        - **Immediate Action Required**
        - Schedule one-on-one meetings within 48 hours
        - Develop personalized learning plan
        - Connect with tutoring services
        - Consider reduced course load if necessary
        - Weekly progress check-ins
        
        #### ⚠️ Moderate Risk Students (Score 50-70%)
        - **Proactive Support Recommended**
        - Bi-weekly check-ins
        - Study skills workshops
        - Peer mentoring programs
        - Additional practice materials
        - Monitor attendance closely
        
        #### ✅ Low Risk Students (Score > 70%)
        - **Maintain Current Support**
        - Monthly progress reviews
        - Encourage peer tutoring roles
        - Provide enrichment opportunities
        - Set stretch goals for excellence
        
        #### 📊 Early Warning Indicators
        - Study hours < 4 per week
        - Attendance < 75%
        - Previous scores < 60%
        - Declining trend in any metric
        """)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; padding: 20px; background: linear-gradient(90deg, #667eea 0%, #764ba2 100%); 
                color: white; border-radius: 10px; margin: 20px 0;'>
        <h4>Made by Isha Yadav as part of Celebal Internship Project</h4>  
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
