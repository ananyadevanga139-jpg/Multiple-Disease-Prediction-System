import streamlit as st
import numpy as np
import joblib


# ==============================
# PAGE CONFIG
# ==============================

st.set_page_config(
    page_title="Multiple Disease Prediction System",
    page_icon="🩺",
    layout="wide"
)


# ==============================
# CUSTOM CSS
# ==============================

st.markdown("""
<style>

body {
    background-color:#f5f7fb;
}

.main-title {
    font-size:42px;
    font-weight:bold;
    text-align:center;
    color:#1f4e79;
}

.sub-title {
    text-align:center;
    font-size:20px;
    color:#555;
}

.card {
    background:white;
    padding:25px;
    border-radius:15px;
    box-shadow:0px 4px 12px rgba(0,0,0,0.1);
    margin:10px;
}

.result {
    background:#e8f5e9;
    padding:20px;
    border-radius:12px;
    color:#1b5e20;
    font-size:20px;
}

.footer {
    text-align:center;
    color:#777;
    margin-top:50px;
}

</style>
""", unsafe_allow_html=True)



# ==============================
# LOAD MODELS
# ==============================

heart_model = joblib.load(
    "models/heart_model.pkl"
)

diabetes_model = joblib.load(
    "models/diabetes_model.pkl"
)

parkinson_model = joblib.load(
    "models/parkinson_model.pkl"
)



# ==============================
# HEADER
# ==============================

st.markdown(
"""
<div class="main-title">
🩺 Multiple Disease Prediction System
</div>

<div class="sub-title">
AI Based Healthcare Prediction using Machine Learning
</div>
""",
unsafe_allow_html=True
)


st.write("")



# ==============================
# SIDEBAR
# ==============================

menu = st.sidebar.selectbox(
    "Navigation",
    [
        "🏠 Home",
        "❤️ Heart Disease",
        "🩸 Diabetes Disease",
        "🧠 Parkinson Disease",
        "📌 About Project"
    ]
)



# ==============================
# HOME PAGE
# ==============================

if menu == "🏠 Home":

    st.markdown("""
    <div class="card">

    ## Welcome 👋

    This application predicts three major diseases using
    Machine Learning algorithms.

    ### Available Predictions:

    ❤️ Heart Disease Prediction

    🩸 Diabetes Prediction

    🧠 Parkinson Disease Prediction


    </div>

    """,
    unsafe_allow_html=True)



    col1,col2,col3 = st.columns(3)


    with col1:
        st.info(
            """
            ❤️ Heart Disease

            Predict cardiovascular disease risk
            based on medical parameters.
            """
        )


    with col2:
        st.success(
            """
            🩸 Diabetes

            Predict diabetes possibility
            using patient information.
            """
        )


    with col3:
        st.warning(
            """
            🧠 Parkinson

            Predict Parkinson disease
            using voice measurements.
            """
        )



# ==============================
# HEART DISEASE
# ==============================


elif menu == "❤️ Heart Disease":


    st.header("❤️ Heart Disease Prediction")


    col1,col2 = st.columns(2)


    with col1:

        age = st.number_input(
            "Age",
            1,
            100
        )

        sex = st.selectbox(
            "Gender",
            [0,1]
        )

        cp = st.number_input(
            "Chest Pain Type"
        )

        trestbps = st.number_input(
            "Blood Pressure"
        )

        chol = st.number_input(
            "Cholesterol"
        )

        fbs = st.number_input(
            "Fasting Blood Sugar"
        )


    with col2:

        restecg = st.number_input(
            "Rest ECG"
        )

        thalach = st.number_input(
            "Maximum Heart Rate"
        )

        exang = st.number_input(
            "Exercise Angina"
        )

        oldpeak = st.number_input(
            "Old Peak"
        )

        slope = st.number_input(
            "Slope"
        )

        ca = st.number_input(
            "Number of vessels"
        )


    if st.button("Predict Heart Disease"):


        data=np.array(
        [
        age,sex,cp,trestbps,
        chol,fbs,restecg,
        thalach,exang,
        oldpeak,slope,ca
        ]
        ).reshape(1,-1)


        prediction = heart_model.predict(data)


        if prediction[0]==1:
            st.error(
            "⚠️ High Risk of Heart Disease"
            )

        else:
            st.success(
            "✅ No Heart Disease Detected"
            )



# ==============================
# DIABETES
# ==============================


elif menu == "🩸 Diabetes Disease":


    st.header("🩸 Diabetes Prediction")


    values=[]


    fields=[
    "Pregnancies",
    "Glucose",
    "Blood Pressure",
    "Skin Thickness",
    "Insulin",
    "BMI",
    "Diabetes Pedigree",
    "Age"
    ]


    for f in fields:
        values.append(
            st.number_input(f)
        )



    if st.button("Predict Diabetes"):


        data=np.array(values).reshape(1,-1)


        result=diabetes_model.predict(data)


        if result[0]==1:
            st.error(
            "⚠️ Diabetes Risk Detected"
            )

        else:
            st.success(
            "✅ No Diabetes Detected"
            )



# ==============================
# PARKINSON
# ==============================


elif menu == "🧠 Parkinson Disease":


    st.header("🧠 Parkinson Disease Prediction")


    st.write(
    "Enter voice measurement values"
    )


    values=[]


    for i in range(22):

        values.append(
            st.number_input(
                f"Feature {i+1}"
            )
        )



    if st.button("Predict Parkinson"):


        data=np.array(values).reshape(1,-1)


        result=parkinson_model.predict(data)


        if result[0]==1:
            st.error(
            "⚠️ Parkinson Disease Detected"
            )

        else:
            st.success(
            "✅ No Parkinson Disease Detected"
            )



# ==============================
# ABOUT
# ==============================


elif menu == "📌 About Project":


    st.header("📌 About Project")


    st.markdown("""
    <div class="card">

    ### Multiple Disease Prediction System

    **Technologies Used**

    - Python
    - Streamlit
    - Machine Learning
    - Scikit-learn
    - Pandas
    - NumPy


    ### Machine Learning Models

    ❤️ Heart Disease - Classification Model

    🩸 Diabetes - Classification Model

    🧠 Parkinson - Classification Model


    ### Developer

    Ananya K

    Information Science and Engineering

    AMC Engineering College


    </div>

    """,
    unsafe_allow_html=True)



# ==============================
# FOOTER
# ==============================


st.markdown(
"""
<div class="footer">

© 2026 Multiple Disease Prediction System  
Developed by Ananya K

</div>
""",
unsafe_allow_html=True
)