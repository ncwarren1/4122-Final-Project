import pandas as pd
import altair as alt
import streamlit as st
import numpy as np

import pickle

pickle_in = open('linear.pkl', 'rb')
model = pickle.load(pickle_in)

@st.cache
def load_data():
    df = pd.read_csv('covid_education_impact.csv')
    return df
df = load_data()
def prediction(gender, financial_situation, geography,do_children_have_internet_connection):
  
    if gender == "Female":
        gender = 0
    else:
        gender = 1
 
 
    if financial_situation == "^Icannotafford" or "Prefernottoanswer":
        financial_situation = 0
    else:
        financial_situation = 1
  
    if geography == "Rural":
        geography = 0
    else:
        geography = 1
    if do_children_have_internet_connection == "Yes":
        do_children_have_internet_connection = 0
    else:
        do_children_have_internet_connection = 1
    
    prediction = model.predict( 
        [[gender, financial_situation,geography,do_children_have_internet_connection]])
    
    
    print(prediction)
    return prediction


fsChart = alt.Chart(df).mark_bar().encode(
    x='count(financial_situation):Q',
    y='financial_situation:N',
    color='geography:N'
).properties(
    width=1000,
    height=500
).configure_axis(labelLimit=6000)

esChart = alt.Chart(df).mark_bar().encode(
    x='count(employment_status):Q',
    y='employment_status:N',
    color='geography:N'
).properties(
    width=1000,
    height=500
).configure_axis(labelLimit=6000)

edChart = alt.Chart(df).mark_bar().encode(
    x='count(education):Q',
    y='education:N',
    color='geography:N'
).properties(
    width=1000,
    height=500
).configure_axis(labelLimit=6000)

choice = st.radio(
    "Which graph do you want to see?",
    ('financial_situation', 'employment_status', 'education')
)
if choice == 'financial_situation':
    st.write(fsChart)
elif choice == 'employment_status':
    st.write(esChart)
elif choice == 'education':
    st.write(edChart)
else:
    st.write('You must select a choice')
    
    
st.markdown('##Electricity Efficiency Prediction:')
st.caption("Make sure selection based on the following options:|| -Geography:Rural or Prefer not to answer - 0 and Suburban/Peri-urban or City center and metropolitan area=1   || -Gender:Female = 0 and Other=1  ||-Finacial Situation:I cannot afford = 0 and I can afford=1   ||-Do children have internet connection:Yes = 0 and No=1")

geography = st.radio('Geography',(0, 1))
gender = st.radio('Gender',(0, 1))
financial_situation = st.radio('Financial Situation',(0,1))
do_children_have_internet_connection = st.radio('Do your children have internet connection',(0,1))
if st.button('Predict'):
 result = prediction(geography, gender, financial_situation, do_children_have_internet_connection)
 st.success("Result is{}".format(result))

