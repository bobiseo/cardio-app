import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import pandas as pd
import numpy as np
import joblib


st.set_page_config(page_title="Cardio Risk Prediction", layout="centered")
st.title("🫀 Cardiovascular Disease Risk Prediction")
st.caption("Enter patient details to estimate cardiovascular disease risk.")


model = joblib.load("saved_models/model_gradient_boosting.pkl")
scaler = joblib.load("saved_models/scaler.pkl")

st.subheader("📝 Patient Information")

age = st.slider("Age (in years)", 18, 100, 50)
gender = st.radio("Gender", ["Male", "Female"])
height = st.slider("Height (cm)", 140, 210, 170)
weight = st.slider("Weight (kg)", 40, 150, 70)
ap_hi = st.slider("Systolic blood pressure (ap_hi)", 90, 200, 120)
ap_lo = st.slider("Diastolic blood pressure (ap_lo)", 60, 140, 80)
cholesterol = st.selectbox("Cholesterol level (1=Normal, 2=Above Normal, 3=Well Above)", [1, 2, 3])
gluc = st.selectbox("Glucose level (1=Normal, 2=Above Normal, 3=Well Above)", [1, 2, 3])
smoke = st.checkbox("Do you smoke?")
alco = st.checkbox("Do you drink alcohol?")
active = st.checkbox("Are you physically active?")


input_data = pd.DataFrame({
    'age': [age],
    'gender': [1 if gender == "Male" else 2],
    'height': [height],
    'weight': [weight],
    'ap_hi': [ap_hi],
    'ap_lo': [ap_lo],
    'cholesterol': [cholesterol],
    'gluc': [gluc],
    'smoke': [int(smoke)],
    'alco': [int(alco)],
    'active': [int(active)]
})


input_data['bmi'] = input_data['weight'] / ((input_data['height'] / 100) ** 2)
input_data.drop(['height', 'weight'], axis=1, inplace=True)


input_data = input_data[[
    'age', 'gender', 'ap_hi', 'ap_lo',
    'cholesterol', 'gluc', 'smoke',
    'alco', 'active', 'bmi'
]]


input_scaled = scaler.transform(input_data)
prediction = model.predict(input_scaled)[0]
probability = model.predict_proba(input_scaled)[0][1]


st.subheader("🧪 Prediction Result")
st.metric(label="Predicted Risk Probability", value=f"{probability:.2f}")

if prediction == 1:
    st.error("⚠️ High Risk of Cardiovascular Disease")
else:
    st.success("✅ Low Risk of Cardiovascular Disease")
