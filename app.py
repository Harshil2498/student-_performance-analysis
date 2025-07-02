import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Load data and model
try:
    data = pd.read_csv('stud.csv')
    rf_model = joblib.load('rf_model.pkl')
    scaler = joblib.load('scaler.pkl')
    le_gender = joblib.load('le_gender.pkl')
    le_lunch = joblib.load('le_lunch.pkl')
    le_test_prep = joblib.load('le_test_prep.pkl')
    feature_names = joblib.load('feature_names.pkl')  # Load feature names
except FileNotFoundError as e:
    st.error(f"Error: Missing file {str(e)}. Ensure 'stud.csv' and all .pkl files are in the project directory.")
    st.stop()

# App title
st.title('Student Performance Analysis Dashboard')

# Sidebar for navigation
st.sidebar.title('Navigation')
page = st.sidebar.selectbox('Choose a Section', ['Data Overview', 'EDA Visualizations', 'Math Score Predictor', 'Insights'])

# Cache data to improve performance
@st.cache_data
def load_data():
    return pd.read_csv('stud.csv')

# Section 1: Data Overview
if page == 'Data Overview':
    st.header('Data Overview')
    st.write('Dataset contains 1000 rows and 8 columns: gender, race_ethnicity, parental_level_of_education, lunch, test_preparation_course, math_score, reading_score, writing_score.')
    st.write(data.head(1000))
    st.write('Summary Statistics:')
    st.write(data.describe())

# Section 2: EDA Visualizations
elif page == 'EDA Visualizations':
    st.header('Exploratory Data Analysis')
    vis_option = st.selectbox('Select Visualization', [
        'Score Distribution',
        'Math Score by Gender',
        'Math Score by Lunch Type',
        'Math Score by Test Preparation',
        'Correlation Heatmap',
        'Feature Importance'
    ])

    plt.figure(figsize=(10, 6))
    if vis_option == 'Score Distribution':
        sns.histplot(data['math_score'], kde=True)
        plt.title('Distribution of Math Scores')
    elif vis_option == 'Math Score by Gender':
        sns.boxplot(x='gender', y='math_score', data=data)
        plt.title('Math Score by Gender')
    elif vis_option == 'Math Score by Lunch Type':
        sns.boxplot(x='lunch', y='math_score', data=data)
        plt.title('Math Score by Lunch Type')
    elif vis_option == 'Math Score by Test Preparation':
        sns.boxplot(x='test_preparation_course', y='math_score', data=data)
        plt.title('Math Score by Test Preparation')
    elif vis_option == 'Correlation Heatmap':
        sns.heatmap(data[['math_score', 'reading_score', 'writing_score']].corr(), annot=True, cmap='coolwarm')
        plt.title('Correlation Matrix')
    elif vis_option == 'Feature Importance':
        feature_importance = pd.DataFrame(rf_model.feature_importances_, index=feature_names, columns=['Importance'])
        feature_importance = feature_importance.sort_values('Importance', ascending=False)
        sns.barplot(x='Importance', y=feature_importance.index, data=feature_importance)
        plt.title('Feature Importance for Math Score Prediction')
    
    st.pyplot(plt)

# Section 3: Math Score Predictor
elif page == 'Math Score Predictor':
    st.header('Predict Math Score')
    st.write('Enter student details to predict their math score.')

    # Input fields
    gender = st.selectbox('Gender', ['female', 'male'])
    race_ethnicity = st.selectbox('Race/Ethnicity', ['group A', 'group B', 'group C', 'group D', 'group E'])
    parental_education = st.selectbox('Parental Education', [
        'some high school', 'high school', 'some college',
        'associate\'s degree', 'bachelor\'s degree', 'master\'s degree'
    ])
    lunch = st.selectbox('Lunch', ['standard', 'free/reduced'])
    test_prep = st.selectbox('Test Preparation', ['none', 'completed'])

    # Prepare input data
    try:
        input_data = pd.DataFrame({
            'gender': [gender],
            'race_ethnicity': [race_ethnicity],
            'parental_level_of_education': [parental_education],
            'lunch': [lunch],
            'test_preparation_course': [test_prep]
        })

        # Encode inputs
        input_data['gender'] = le_gender.transform(input_data['gender'])
        input_data['lunch'] = le_lunch.transform(input_data['lunch'])
        input_data['test_preparation_course'] = le_test_prep.transform(input_data['test_preparation_course'])
        input_data = pd.get_dummies(input_data, columns=['race_ethnicity', 'parental_level_of_education'], drop_first=True)
        input_data = input_data.reindex(columns=feature_names, fill_value=0)  # Use feature_names instead of feature_names_in_
        input_data = scaler.transform(input_data)

        # Predict
        if st.button('Predict Math Score'):
            prediction = rf_model.predict(input_data)[0]
            st.success(f'Predicted Math Score: {prediction:.2f}')
    except Exception as e:
        st.error(f"Error in prediction: {str(e)}")

# Section 4: Insights
elif page == 'Insights':
    st.header('Key Insights and Recommendations')
    st.write("""
    - **Test Preparation Impact:** Students who completed the test preparation course score ~10-15 points higher on average in math (based on EDA).
    - **Socioeconomic Status:** Students with free/reduced lunch (indicating lower socioeconomic status) tend to score lower, suggesting a need for targeted support.
    - **Parental Education:** Higher parental education (e.g., master's degree) correlates with better student performance.
    - **Recommendations:**
      - Expand access to test preparation programs, especially for low-income students.
      - Provide academic support for students with parents of lower education levels.
      - Address socioeconomic barriers through nutrition or tutoring programs.
    """)

# Footer
st.sidebar.write('Built with Streamlit by Harshil')