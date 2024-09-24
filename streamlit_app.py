import streamlit as st
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model

# Load the pre-trained model
model = load_model('NN_via_carpatia.h5')

# Function to make predictions
def predict_psy_health(data):
    data = np.array(data).reshape(1, -1)
    prediction = model.predict(data)
    return prediction

# Add page title and description
st.title('🧠 Mental Health Prediction App')
st.write("""
#### Enter your information below to get a personalized prediction about your mental health status.
This tool helps in assessing potential mental health risks based on several lifestyle and historical factors.
""")

# Collect user inputs
st.markdown("### Please fill in the following details:")

gender = st.radio('👤 What is your gender?', ['Male', 'Female'])
occupation = st.selectbox('💼 What is your occupation?', ['Student', 'Business', 'Housewife', 'Others'])
self_employed = st.radio('🏢 Are you self-employed?', ['Yes', 'No'])
family_history = st.radio('👨‍👩‍👧 Do you have a family history of mental illness?', ['Yes', 'No'])
treatment = st.radio('💊 Are you currently receiving mental health treatment?', ['Yes', 'No'])
days_indoors = st.selectbox('🏠 How often do you stay indoors?', ['Go out Every day', '1-14 days', '15-30 days', 'More than 2 months'])
growing_stress = st.selectbox('😥 Are your stress levels increasing?', ['Yes', 'No', 'Maybe'])
changes_habits = st.selectbox('🔄 Have you noticed changes in your habits?', ['Yes', 'No'])
mental_health_history = st.selectbox('📋 Do you have a history of mental health issues?', ['Yes', 'No', 'Maybe'])
mood_swings = st.selectbox('⚖️ Do you experience mood swings?', ['Yes', 'No'])
coping_struggles = st.selectbox('🆘 How well do you cope with daily challenges?', ['Low', 'Medium', 'High'])
work_interest = st.selectbox('📈 How interested are you in your work?', ['Yes', 'No', 'Maybe'])
social_weakness = st.selectbox('🤝 Do you feel socially weak or disconnected?', ['Yes', 'No'])
mental_health_interview = st.selectbox('🗣️ Have you had a mental health interview?', ['Yes', 'No', 'Maybe'])
care_options = st.selectbox('💼 Do you have access to mental health care options?', ['Yes', 'No', 'Maybe'])

# Prepare data for prediction
input_data = {
    'Gender': gender,
    'Occupation': occupation,
    'self_employed': self_employed,
    'family_history': family_history,
    'treatment': treatment,
    'Days_Indoors': days_indoors,
    'Growing_Stress': growing_stress,
    'Changes_Habits': changes_habits,
    'Mental_Health_History': mental_health_history,
    'Mood_Swings': mood_swings,
    'Coping_Struggles': coping_struggles,
    'Work_Interest': work_interest,
    'Social_Weakness': social_weakness,
    'mental_health_interview': mental_health_interview,
    'care_options': care_options
}

# Function to preprocess input data
def preprocess_input(data):
    df = pd.DataFrame([data])
    
    # One-hot encoding for categorical variables
    dummies_columns = ['Days_Indoors', 'Gender', 'Occupation', 'Growing_Stress', 
                       'Changes_Habits', 'Mental_Health_History', 'Mood_Swings', 
                       'Work_Interest', 'Social_Weakness', 'mental_health_interview', 'care_options']
    
    df_dummies = pd.get_dummies(df[dummies_columns])
    df = df.drop(columns=dummies_columns)
    df = pd.concat([df, df_dummies], axis=1)

    # Convert yes/no to binary
    columns_to_replace = ['self_employed', 'family_history', 'treatment', 'Coping_Struggles']
    df[columns_to_replace] = df[columns_to_replace].replace({'Yes': 1, 'No': 0})
    df['Coping_Struggles'] = df['Coping_Struggles'].replace({'Low': 0, 'Medium': 1, 'High': 2})

    return df.values

# Prediction button with feedback
if st.button('🔍 Predict'):
    preprocessed_data = preprocess_input(input_data)
    prediction = predict_psy_health(preprocessed_data)

    if prediction[0] > 0.5:
        st.error('🚨 Warning! There is a possible risk of mental illness.')
    else:
        st.success('🎉 No risk of mental illness detected.')

# Footer
st.markdown("""
---
**Disclaimer:** This app provides a preliminary mental health risk assessment. For professional advice, consult a healthcare provider.
""")
