import streamlit as st
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model

model = load_model('NN_via_carpatia.h5')


def predict_psy_health(data):
    prediction = model.predict(data)
    return prediction

st.title('Mental Health Prediction App')


gender = st.selectbox('What is your gender?', ['Male', 'Female'])
occupation = st.selectbox('What is your occupation?', ['Student', 'Business', 'Housewife', 'Others', 'Corporate'])
self_employed = st.selectbox('Are you self-employed?', ['Yes', 'No'])
family_history = st.selectbox('Do you have a family history of mental illness?', ['Yes', 'No'])
treatment = st.selectbox('Are you currently receiving mental health treatment?', ['Yes', 'No'])
days_indoors = st.selectbox('How often do you stay indoors?', ['1-14 days', '15-30 days', '31-60 days', 'Go out Every day', 'More than 2 months'])
growing_stress = st.selectbox('Are your stress levels increasing?', ['Yes', 'No', 'Maybe'])
changes_habits = st.selectbox('Have you noticed changes in your habits?', ['Yes', 'No', 'Maybe'])
mental_health_history = st.selectbox('Do you have a history of mental health issues?', ['Yes', 'No', 'Maybe'])
mood_swings = st.selectbox('Do you experience mood swings?', ['High', 'Low', 'Medium'])
coping_struggles = st.selectbox('How well do you cope with daily challenges?', ['Yes', 'No'])
work_interest = st.selectbox('How interested are you in your work?', ['Yes', 'No', 'Maybe'])
social_weakness = st.selectbox('Do you feel socially weak or disconnected?', ['Yes', 'No', 'Maybe'])
mental_health_interview = st.selectbox('Have you had a mental health interview?', ['Yes', 'No', 'Maybe'])
care_options = st.selectbox('Do you have access to mental health care options?', ['Yes', 'No', 'Not sure'])

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

def preprocess_input(data):
    df = pd.DataFrame([data])
    binary_columns = ['self_employed', 'family_history', 'Coping_Struggles']
    df[binary_columns] = df[binary_columns].replace({'Yes': 1, 'No': 0})
    dummies_columns = ['Gender', 'Occupation', 'Days_Indoors', 'Growing_Stress', 
                       'Changes_Habits', 'Mental_Health_History', 'Mood_Swings', 
                       'Work_Interest', 'Social_Weakness', 'mental_health_interview', 'care_options']
    df = pd.get_dummies(df, columns=dummies_columns)
    expected_columns = ['self_employed', 'family_history', 'Coping_Struggles',
                        'Days_Indoors_1-14 days', 'Days_Indoors_15-30 days',
                        'Days_Indoors_31-60 days', 'Days_Indoors_Go out Every day',
                        'Days_Indoors_More than 2 months', 'Gender_Female', 'Gender_Male',
                        'Occupation_Business', 'Occupation_Corporate', 'Occupation_Housewife',
                        'Occupation_Others', 'Occupation_Student', 'Growing_Stress_Maybe',
                        'Growing_Stress_No', 'Growing_Stress_Yes', 'Changes_Habits_Maybe',
                        'Changes_Habits_No', 'Changes_Habits_Yes', 'Mental_Health_History_Maybe',
                        'Mental_Health_History_No', 'Mental_Health_History_Yes', 'Mood_Swings_High',
                        'Mood_Swings_Low', 'Mood_Swings_Medium', 'Work_Interest_Maybe',
                        'Work_Interest_No', 'Work_Interest_Yes', 'Social_Weakness_Maybe',
                        'Social_Weakness_No', 'Social_Weakness_Yes', 'mental_health_interview_Maybe',
                        'mental_health_interview_No', 'mental_health_interview_Yes', 'care_options_No',
                        'care_options_Not sure', 'care_options_Yes']
    for col in expected_columns:
        if col not in df.columns:
            df[col] = 0
    df = df[expected_columns]
    return df.values.astype(np.float32)


if st.button('Predict'):
    preprocessed_data = preprocess_input(input_data)
    #st.write(f"Shape of preprocessed data: {preprocessed_data.shape}")
    if preprocessed_data is not None:
        try:
            prediction = predict_psy_health(preprocessed_data)

            if prediction[0][0] > 0.5:
                st.info('There is a possible risk of mental illness.')
            else:
                st.info('No risk of mental illness detected.')
        except ValueError as e:
            st.error(f"Error during prediction: {e}")



    st.info(
        """
        **Author:** *Mateusz Walo*  
        **LinkedIn:** [My LinkedIn profile :)](https://www.linkedin.com/in/mateuszwalo/)  
        **Data:** [Mental Health Dataset](https://www.kaggle.com/datasets/bhavikjikadara/mental-health-dataset)
        """
    )
