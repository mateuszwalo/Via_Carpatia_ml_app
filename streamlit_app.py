import streamlit as st
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model


model = load_model('NN_via_carpatia.h5')


def predict_psy_health(data):
    data = np.array(data).reshape(1, -1)  
    prediction = model.predict(data)
    return prediction

st.title(' Mental Health Prediction App')
st.write("""
#### Enter your information below to get a personalized prediction about your mental health status.
This tool helps in assessing potential mental health risks based on several lifestyle and historical factors.
""")


gender = st.selectbox(' What is your gender?', ['Male', 'Female'])
occupation = st.selectbox('What is your occupation?', ['Student', 'Business', 'Housewife', 'Others', 'Corporate'])
self_employed = st.selectbox('Are you self-employed?', ['Yes', 'No'])
family_history = st.selectbox('Do you have a family history of mental illness?', ['Yes', 'No'])
treatment = st.selectbox('Are you currently receiving mental health treatment?', ['Yes', 'No'])
days_indoors = st.selectbox('How often do you stay indoors?', ['1-14 days', '15-30 days', '31-60 days', 'Go out Every day','More than 2 months'])
growing_stress = st.selectbox('Are your stress levels increasing?', ['Yes', 'No', 'Maybe'])
changes_habits = st.selectbox('Have you noticed changes in your habits?', ['Yes', 'No','Maybe'])
mental_health_history = st.selectbox('Do you have a history of mental health issues?', ['Yes', 'No', 'Maybe'])
mood_swings = st.selectbox('Do you experience mood swings?', ['High', 'Low','Medium'])
coping_struggles = st.selectbox('How well do you cope with daily challenges?', ['Yes','No'])
work_interest = st.selectbox('How interested are you in your work?', ['Yes', 'No', 'Maybe'])
social_weakness = st.selectbox('Do you feel socially weak or disconnected?', ['Yes', 'No','Maybe'])
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
    df2 = pd.DataFrame([data])
    dummies_columns = ['Days_Indoors', 'Gender', 'Occupation', 'Growing_Stress', 
                   'Changes_Habits', 'Mental_Health_History', 'Mood_Swings', 
                   'Work_Interest', 'Social_Weakness', 'mental_health_interview','care_options']
    
    df_dummies = pd.get_dummies(df2[dummies_columns])
    df2 = df2.drop(columns=dummies_columns)
    df_usa = pd.concat([df2, df_dummies], axis=1)
    columns_to_convert = df_usa.select_dtypes(include=['bool']).columns
    df_usa[columns_to_convert] = df_usa[columns_to_convert].apply(lambda x: x.astype(int))
    columns_to_replace = ['self_employed', 'family_history', 'treatment', 'Coping_Struggles']
    df_usa[columns_to_replace] = df_usa[columns_to_replace].replace({'Yes': 1, 'No': 0})
    df_usa[columns_to_replace] = df_usa[columns_to_replace].astype(int)
    
    return df_usa.values

if st.button('Predict'):
    preprocessed_data = preprocess_input(input_data)
    
    if preprocessed_data is not None:  
        prediction = predict_psy_health(preprocessed_data)

        if prediction[0][0] > 0.5: 
            st.error('Warning! There is a possible risk of mental illness.')
        else:
            st.success('No risk of mental illness detected.')

