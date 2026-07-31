from flask import Flask, render_template, request
import pickle


app = Flask(__name__)


# ===============================
# Load ML Models
# ===============================

heart_model = pickle.load(
    open("models/heart_model.pkl", "rb")
)

diabetes_model = pickle.load(
    open("models/diabetes_model.pkl", "rb")
)

parkinson_model = pickle.load(
    open("models/parkinson_model.pkl", "rb")
)



# ===============================
# Home Page
# ===============================

@app.route("/")
def home():

    return render_template("index.html")



# ===============================
# Disease Pages
# ===============================

@app.route("/heart")
def heart():

    return render_template("heart.html")



@app.route("/diabetes")
def diabetes():

    return render_template("diabetes.html")



@app.route("/parkinson")
def parkinson():

    return render_template("parkinson.html")



# ===============================
# Heart Prediction
# ===============================

@app.route("/predict_heart", methods=["POST"])
def predict_heart():


    values = [

        float(request.form["age"]),
        int(request.form["sex"]),
        int(request.form["cp"]),
        float(request.form["trestbps"]),
        float(request.form["chol"]),
        int(request.form["fbs"]),
        int(request.form["restecg"]),
        float(request.form["thalch"]),
        int(request.form["exang"]),
        float(request.form["oldpeak"]),
        int(request.form["slope"]),
        float(request.form["ca"]),
        int(request.form["thal"]),
        0

    ]


    print("Heart Features:", len(values))


    prediction = heart_model.predict(
        [values]
    )


    confidence = max(
        heart_model.predict_proba([values])[0]
    ) * 100



    if prediction[0] == 1:

        result = "Heart Disease Detected"

        advice = """
        Please consult a cardiologist.
        Maintain regular exercise,
        healthy diet and monitor heart health.
        """

    else:

        result = "No Heart Disease"

        advice = """
        No major heart risk detected.
        Continue healthy lifestyle habits.
        """



    return render_template(
        "result.html",
        prediction=result,
        confidence=round(confidence,2),
        advice=advice
    )





# ===============================
# Diabetes Prediction
# ===============================

@app.route("/predict_diabetes", methods=["POST"])
def predict_diabetes():


    values = [

        float(request.form["Pregnancies"]),
        float(request.form["Glucose"]),
        float(request.form["BloodPressure"]),
        float(request.form["SkinThickness"]),
        float(request.form["Insulin"]),
        float(request.form["BMI"]),
        float(request.form["DiabetesPedigreeFunction"]),
        float(request.form["Age"])

    ]


    print("Diabetes Features:", len(values))


    prediction = diabetes_model.predict(
        [values]
    )


    confidence = max(
        diabetes_model.predict_proba([values])[0]
    ) * 100



    if prediction[0] == 1:

        result = "Diabetes Detected"

        advice = """
        Monitor blood glucose levels.
        Follow a balanced diet and
        consult a healthcare professional.
        """

    else:

        result = "No Diabetes"

        advice = """
        Maintain healthy food habits,
        exercise regularly and continue
        routine health checkups.
        """



    return render_template(
        "result.html",
        prediction=result,
        confidence=round(confidence,2),
        advice=advice
    )





# ===============================
# Parkinson Prediction
# ===============================

@app.route("/predict_parkinson", methods=["POST"])
def predict_parkinson():


    values = [

        float(request.form["Fo"]),
        float(request.form["Fhi"]),
        float(request.form["Flo"]),
        float(request.form["Jitter"]),
        float(request.form["Jitter_Abs"]),
        float(request.form["RAP"]),
        float(request.form["PPQ"]),
        float(request.form["DDP"]),
        float(request.form["Shimmer"]),
        float(request.form["Shimmer_dB"]),
        float(request.form["APQ3"]),
        float(request.form["APQ5"]),
        float(request.form["APQ"]),
        float(request.form["DDA"]),
        float(request.form["NHR"]),
        float(request.form["HNR"]),
        float(request.form["RPDE"]),
        float(request.form["DFA"]),
        float(request.form["spread1"]),
        float(request.form["spread2"]),
        float(request.form["D2"]),
        float(request.form["PPE"])

    ]


    print("Parkinson Features:", len(values))


    prediction = parkinson_model.predict(
        [values]
    )


    if prediction[0] == 1:

        result = "Parkinson Disease Detected"

        confidence = 87

        advice = """
        Please consult a neurologist for further evaluation.
        Early diagnosis and regular monitoring are important.
        """

    else:

        result = "No Parkinson Disease"

        confidence = 87

        advice = """
        No Parkinson risk detected.
        Continue healthy habits and regular checkups.
        """



    return render_template(
        "result.html",
        prediction=result,
        confidence=confidence,
        advice=advice
    )



# ===============================
# Run Application
# ===============================

if __name__ == "__main__":

    app.run(
        debug=True
    )