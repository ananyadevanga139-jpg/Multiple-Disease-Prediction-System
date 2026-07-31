# 🩺 Multiple Disease Prediction System

A Machine Learning-based healthcare application that predicts the possibility of multiple diseases using patient health parameters. The system provides quick predictions for diseases such as Heart Disease, Diabetes, and Parkinson's Disease using trained machine learning models.

## 🚀 Features

* ❤️ Heart Disease Prediction

  * Predicts heart disease risk based on medical parameters.

* 🩸 Diabetes Prediction

  * Analyzes patient health information to predict diabetes risk.

* 🧠 Parkinson's Disease Prediction

  * Uses patient voice-related features to predict Parkinson's disease.

* 🌐 Web Application

  * Interactive interface for entering details and viewing prediction results.

## 🛠️ Technologies Used

* Python
* Streamlit
* Scikit-learn
* Pandas
* NumPy
* Machine Learning
* Matplotlib
* Seaborn
* Joblib

## 📂 Project Structure

```text
Multiple-Disease-Prediction-System/
│
├── streamlit_app.py
├── models/
│   ├── heart_model.pkl
│   ├── diabetes_model.pkl
│   └── parkinson_model.pkl
│
├── dataset/
├── training/
├── analysis/
├── requirements.txt
└── README.md
```

## ⚙️ Installation & Setup

### 1. Clone Repository

```bash
git clone https://github.com/your-username/Multiple-Disease-Prediction-System.git
```

### 2. Navigate to Project Folder

```bash
cd Multiple-Disease-Prediction-System
```

### 3. Create Virtual Environment

```bash
python -m venv venv
```

### 4. Activate Environment

Windows:

```bash
venv\Scripts\activate
```

### 5. Install Dependencies

```bash
pip install -r requirements.txt
```

### 6. Run Application

```bash
streamlit run streamlit_app.py
```

The application will open in your browser:

```text
http://localhost:8501
```

## ☁️ Deployment

The application is deployed using **Streamlit Cloud**.

Deployment Steps:

1. Upload project to GitHub repository.
2. Connect GitHub repository with Streamlit Cloud.
3. Select `streamlit_app.py` as the main application file.
4. Install dependencies from `requirements.txt`.
5. Deploy the application online.

## 📌 Machine Learning Models

### Heart Disease Model

* Algorithm: Machine Learning Classification Model
* Model File:

```text
heart_model.pkl
```

### Diabetes Prediction Model

* Algorithm: Machine Learning Classification Model
* Model File:

```text
diabetes_model.pkl
```

### Parkinson's Disease Model

* Algorithm: Machine Learning Classification Model
* Model File:

```text
parkinson_model.pkl
```

## 🎯 Future Enhancements

* Add more disease prediction modules.
* Improve model accuracy using advanced algorithms.
* Add user authentication.
* Store prediction history.
* Integrate healthcare APIs.

## 👩‍💻 Developed By

**Ananya K**
