import streamlit as st
import pandas as pd
import requests
from PIL import Image

def get_user_input():
    person_age = st.sidebar.slider('Age', min_value=0, max_value=200, step=1, value=1)
    person_gender = st.sidebar.selectbox('Gender', ['male', 'female'])
    person_education = st.sidebar.selectbox('Eduction', ['Bachelor', 'Associate', 'High School', 'Master', 'Doctorate'])
    person_income=st.sidebar.slider('Income', min_value=0, max_value=100000000)
    person_emp_exp = st.sidebar.slider('Employment experience', min_value=0, max_value=100)
    person_home_ownership = st.sidebar.selectbox('Home ownership', ['RENT', 'OWN', 'MORTGAGE', 'OTHER'])
    loan_amnt = st.sidebar.slider('Loan amount', min_value=0, max_value=100000000)
    loan_intent=st.sidebar.selectbox('Loan intent', ['PERSONAL', 'EDUCATION', 'MEDICAL', 'VENTURE', 'DEBTCONSOLIDATION', 'HOMEIMPROVEMENT'])
    loan_int_rate = st.sidebar.number_input('Interest rate', min_value=0, max_value=25)
    loan_percent_income=st.sidebar.slider('Loan income percentage', min_value=0.0, max_value=1.0, step=0.01)
    cb_person_cred_hist_length=st.sidebar.slider('Credit history length', min_value=0, max_value=50)
    credit_score = st.sidebar.slider('Credit score', min_value=0, max_value=1000)
    previous_loan_defaults_on_file = st.sidebar.selectbox('Previous loan on file', ['No', 'Yes'])

    userdata = {
        'person_age': person_age,
        'person_gender': person_gender,
        'person_education': person_education,
        'person_income': person_income,
        'person_emp_exp': person_emp_exp,
        'person_home_ownership': person_home_ownership,
        'loan_amnt': loan_amnt,
        'loan_intent': loan_intent,
        'loan_int_rate': loan_int_rate,
        'loan_percent_income': loan_percent_income,
        'cb_person_cred_hist_length': cb_person_cred_hist_length,
        'credit_score': credit_score,
        'previous_loan_defaults_on_file': previous_loan_defaults_on_file
    }
    return userdata

class Config:
    PAGE_TITLE = "Predict Loan Application APP"
    SERVER_IP_ADDRESS = "127.0.0.1"
    SERVER_PORT_NUMBER = 8000

server_ip_and_port = f"{Config.SERVER_IP_ADDRESS}:{Config.SERVER_PORT_NUMBER}"

st.set_page_config(
    page_title=Config.PAGE_TITLE
)

st.title(Config.PAGE_TITLE)
image_sidebar = Image.open('images/loan.png')
st.sidebar.image(image_sidebar, use_container_width=True)
st.sidebar.header('Peronal Profile')
st.header("Predict the submitted loan getting approved or not by profiling and loan information")
user_data = get_user_input()

if st.button("Predict"):
    user_data = pd.DataFrame(user_data, index=[0]).to_json()
    get_prediction = requests.get(f"http://{server_ip_and_port}/input/", params={'input': user_data})
    post_prediction = requests.post(f"http://{server_ip_and_port}/predict/")
    
    st.subheader("Predicted result")
    st.write(f"**{post_prediction.json()}**")




