import pandas as pd
import altair as alt
import streamlit as st


@st.cache
def load_data():
    df = pd.read_csv('covid_education_impact.csv')
    return df


df = load_data()


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