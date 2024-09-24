import streamlit as st
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
model = load_model('NN_via_carpatia.h5')

def predict_psy_health(data):
    data = np.array(data).reshape(1, -1)
    prediction = model.predict(data)
    return prediction

st.title('Mental Health Prediction')

st.write("""
### Enter the following data to get a prediction of your mental health status.
""")

df = pd.DataFrame(columns=[
    'Gender', 'Occupation', 'self_employed', 'family_history', 'treatment', 'Days_Indoors',
    'Growing_Stress', 'Changes_Habits', 'Mental_Health_History', 'Mood_Swings',
    'Coping_Struggles', 'Work_Interest', 'Social_Weakness', 'mental_health_interview', 'care_options'
])


gender = st.selectbox('Gender', ['Male', 'Female'])
occupation = st.selectbox('Occupation', ['Student', 'Business', 'Housewife', 'Others'])
self_employed = st.selectbox('Self-employed?', ['Yes', 'No'])
family_history = st.selectbox('Family history of mental illness?', ['Yes', 'No'])
days_indoors = st.selectbox('Days spent indoors', ['Go out Every day', '1-14 days', '15-30 days', 'More than 2 months'])
growing_stress = st.selectbox('Growing stress levels?', ['Yes', 'No', 'Maybe'])
changes_habits = st.selectbox('Changes in habits?', ['Yes', 'No'])
mental_health_history = st.selectbox('Mental health history?', ['Yes', 'No', 'Maybe'])
mood_swings = st.selectbox('Mood swings?', ['Yes', 'No'])
coping_struggles = st.selectbox('Struggles with coping?', ['Low', 'Medium', 'High'])
work_interest = st.selectbox('Interest in work?', ['Yes', 'No', 'Maybe'])
social_weakness = st.selectbox('Social weakness?', ['Yes', 'No'])
mental_health_interview = st.selectbox('Mental health interview?', ['Yes', 'No', 'Maybe'])
care_options = st.selectbox('Access to care options?', ['Yes', 'No', 'Maybe'])


if st.button('Add Data'):
    new_data = {
        'Gender': gender,
        'Occupation': occupation,
        'self_employed': self_employed,
        'family_history': family_history,
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
    
    df = df.append(new_data, ignore_index=True)
    st.success("Data has been added!")

st.write("### Current Data:")
st.dataframe(df)

dummies_columns = ['Days_Indoors', 'Gender', 'Occupation', 'Growing_Stress', 
                   'Changes_Habits', 'Mental_Health_History', 'Mood_Swings', 
                   'Work_Interest', 'Social_Weakness', 'mental_health_interview','care_options']
df_dummies = pd.get_dummies(df[dummies_columns])
df = df.drop(columns=dummies_columns)
df = pd.concat([df, df_dummies], axis=1)
columns_to_replace = ['self_employed', 'family_history', 'treatment', 'Coping_Struggles']
df[columns_to_replace] = df[columns_to_replace].replace({'Yes': 1, 'No': 0})
df[columns_to_replace] = df[columns_to_replace].astype(int)

if st.button('Predict'):
    prediction = predict_psy_health(df)
    if prediction[0] > 0.5:
        st.error('Warning! There is a possible risk of mental illness.')
    else:
        st.success('No risk of mental illness detected.')

