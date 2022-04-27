import pandas as pd
import altair as alt
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from numpy.core.numeric import True_
from sklearn import metrics
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import plot_confusion_matrix, plot_roc_curve, plot_precision_recall_curve
from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import confusion_matrix

import pickle

pickle_in = open('linear.pkl', 'rb')
t_model = pickle.load(pickle_in)
st.set_option('deprecation.showPyplotGlobalUse', False)


@st.cache(persist=True, allow_output_mutation=True)
def load_data():
    df = pd.read_csv('covid_education_impact.csv')
    return df


df = load_data()


def prediction(gender, financial_situation, geography, do_children_have_internet_connection):
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

    prediction = t_model.predict(
        [[gender, financial_situation, geography, do_children_have_internet_connection]])

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
st.caption(
    "Make sure selection based on the following options:|| -Geography:Rural or Prefer not to answer - 0 and Suburban/Peri-urban or City center and metropolitan area=1   || -Gender:Female = 0 and Other=1  ||-Finacial Situation:I cannot afford = 0 and I can afford=1   ||-Do children have internet connection:Yes = 0 and No=1")

geography = st.radio('Geography', (0, 1))
gender = st.radio('Gender', (0, 1))
financial_situation = st.radio('Financial Situation', (0, 1))
do_children_have_internet_connection = st.radio('Do your children have internet connection', (0, 1))
if st.button('Predict'):
    result = prediction(geography, gender, financial_situation, do_children_have_internet_connection)
    st.success("Result is{}".format(result))


def main():
    # st.title("Venezuela Predicition & Visualization WebApp")
    st.sidebar.title("Venezuela Prediction Visualization")
    st.sidebar.markdown("Choose a specific classification to run!!")

    def load():
        data = pd.read_csv('covid_education_impact.csv')
        label = LabelEncoder()

        for i in data.columns:
            data[i] = label.fit_transform(data[i])
        # label.fit(['geography','gender','financial_situation','do_children_have_internet_connection','does_home_shows_severe_deficit_of_electricity'])
        return data

    df_1 = load()
    class_names = ['financial_situation', 'geogrpahy']
    if st.sidebar.checkbox("Display data", False):
        st.subheader("Show the dataset")
        st.write(df_1)

    def split(df_1):
        y = df_1['does_home_shows_severe_deficit_of_electricity']
        x = df_1[['geography', 'gender', 'financial_situation', 'do_children_have_internet_connection']]
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)

        return x_train, x_test, y_train, y_test

    x_train, x_test, y_train, y_test = split(df_1)

    def plot_metrics(metrics_list):
        if "Confusion Matrix" in metrics_list:
            st.subheader("Confusion Matrix")
            plot_confusion_matrix(model, x_test, y_test,
                                  display_labels=class_names)  # only accepts classifiers, check if your dataset has
            st.pyplot()
        if "ROC Curve" in metrics_list:
            st.subheader("ROC Curve")
            plot_roc_curve(model, x_test, y_test)
            st.pyplot()
        if "Precision-Recall Curve" in metrics_list:
            st.subheader("Precision-Recall Curve")
            plot_precision_recall_curve(model, x_test, y_test)
            st.pyplot()

    st.sidebar.subheader("Choose classifier")
    classifier = st.sidebar.selectbox("Classifier", ("Support Vector Machine (SVM)", "Logistic Regression"))
    if classifier == "Support Vector Machine (SVM)":
        st.sidebar.subheader("Hyperparameters")
        C = st.sidebar.number_input("C (Regularization parameter)", 0.01, 10.0, step=0.01, key="C")
        kernel = st.sidebar.radio("Kernel", ("rbf", "linear"), key="kernel")
        gamma = st.sidebar.radio("Gamma (Kernal coefficient", ("scale", "auto"), key="gamma")
        metrics = st.sidebar.multiselect("What metrics to plot?",
                                         ("Confusion Matrix", "ROC Curve", "Precision-Recall Curve"))
        if st.sidebar.button("Classify", key="classify"):
            st.subheader("Support Vector Machine (SVM) results")
            model = SVC(C=C, kernel=kernel, gamma=gamma)
            model.fit(x_train, y_train)
            accuracy = model.score(x_test, y_test)
            y_pred = model.predict(x_test)
            st.write("Accuracy: ", accuracy.round(2))
            st.write("Precision: ", precision_score(y_test, y_pred, labels=class_names).round(2))
            st.write("Recall: ", recall_score(y_test, y_pred, labels=class_names).round(2))
            plot_metrics(metrics)
    if classifier == 'Logistic Regression':
        st.sidebar.subheader("Model Hyperparameters")
        C = st.sidebar.number_input("C (Regularizaion parameter)", 0.01, 10.0, step=0.01, key='C_LR')
        max_iter = st.sidebar.slider("Maxiumum number of interations", 100, 500, key='max_iter')
        metrics = st.sidebar.multiselect("What metrics to plot?",
                                         ('Confusion Matrix', 'ROC Curve', 'Precision-Recall Curve'))

        if st.sidebar.button("Classfiy", key='classify'):
            st.subheader("Logistic Regression Results")
            model = LogisticRegression(C=C, max_iter=max_iter)
            model.fit(x_train, y_train)
            accuracy = model.score(x_test, y_test)
            y_pred = model.predict(x_test)
            st.write("Accuracy ", accuracy.round(2))
            st.write("Precision: ", precision_score(y_test, y_pred, labels=class_names).round(2))
            st.write("Recall: ", recall_score(y_test, y_pred, labels=class_names).round(2))
            plot_metrics(metrics)

    # source for visualization purposes: https://ruslanmv.com/blog/Web-Application-Classification


if __name__ == '__main__':
    main()