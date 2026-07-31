import streamlit as st
import pickle
import numpy as np


# -----------------------------
# Load Models
# -----------------------------

heart_model = pickle.load(
    open("models/heart_model.pkl", "rb")
)

diabetes_model = pickle.load(
    open("models/diabetes_model.pkl", "rb")
)

parkinson_model = pickle.load(
    open("models/parkinson_model.pkl", "rb")
)



# -----------------------------
# Page Configuration
# -----------------------------

st.set_page_config(
    page_title="Multiple Disease Prediction System",
    page_icon="🩺",
    layout="wide"
)


st.title("🩺 Multiple Disease Prediction System")

st.write(
    "AI-based prediction system for Heart Disease, Diabetes and Parkinson's Disease"
)



# -----------------------------
# Sidebar
# -----------------------------

option = st.sidebar.selectbox(
    "Select Disease",
    [
        "Heart Disease",
        "Diabetes",
        "Parkinson Disease"
    ]
)



# -----------------------------
# Heart Prediction
# -----------------------------

if option == "Heart Disease":

    st.header("❤️ Heart Disease Prediction")


    age = st.number_input("Age")
    sex = st.number_input("Sex (0-Female, 1-Male)")
    cp = st.number_input("Chest Pain Type")
    trestbps = st.number_input("Resting Blood Pressure")
    chol = st.number_input("Cholesterol")
    fbs = st.number_input("Fasting Blood Sugar")
    restecg = st.number_input("Rest ECG")
    thalach = st.number_input("Maximum Heart Rate")
    exang = st.number_input("Exercise Induced Angina")
    oldpeak = st.number_input("Old Peak")
    slope = st.number_input("Slope")
    ca = st.number_input("Major Vessels")
    thal = st.number_input("Thal")


    if st.button("Predict Heart Disease"):

        data = np.array(
            [[
                age, sex, cp, trestbps,
                chol, fbs, restecg,
                thalach, exang, oldpeak,
                slope, ca, thal
            ]]
        )


        prediction = heart_model.predict(data)


        if prediction[0] == 1:
            st.error("⚠️ High Risk of Heart Disease")
        else:
            st.success("✅ Low Risk of Heart Disease")



# -----------------------------
# Diabetes Prediction
# -----------------------------

elif option == "Diabetes":

    st.header("🩸 Diabetes Prediction")


    pregnancies = st.number_input("Pregnancies")
    glucose = st.number_input("Glucose")
    bp = st.number_input("Blood Pressure")
    skin = st.number_input("Skin Thickness")
    insulin = st.number_input("Insulin")
    bmi = st.number_input("BMI")
    diabetes = st.number_input("Diabetes Pedigree Function")
    age = st.number_input("Age")


    if st.button("Predict Diabetes"):


        data = np.array(
            [[
                pregnancies,
                glucose,
                bp,
                skin,
                insulin,
                bmi,
                diabetes,
                age
            ]]
        )


        prediction = diabetes_model.predict(data)


        if prediction[0] == 1:
            st.error("⚠️ Diabetes Detected")
        else:
            st.success("✅ No Diabetes Detected")



# -----------------------------
# Parkinson Prediction
# -----------------------------

else:

    st.header("🧠 Parkinson Disease Prediction")


    features = st.text_area(
        "Enter Parkinson features separated by comma"
    )


    if st.button("Predict Parkinson Disease"):

        try:

            values = [
                float(x)
                for x in features.split(",")
            ]


            data = np.array(
                [values]
            )


            prediction = parkinson_model.predict(data)


            if prediction[0] == 1:
                st.error("⚠️ Parkinson Disease Detected")
            else:
                st.success("✅ No Parkinson Disease Detected")


        except:

            st.warning(
                "Please enter valid numeric values"
            )