# 🩺 Multiple Disease Prediction System

An AI-based healthcare prediction web application that uses Machine Learning models to predict the possibility of **Heart Disease, Diabetes, and Parkinson Disease**.

The application provides an interactive user interface built with **Streamlit** and uses trained ML classification models for real-time predictions.

---

# 🚀 Live Demo

Streamlit App:

https://your-streamlit-link.streamlit.app/

---

# 📌 Features

### ❤️ Heart Disease Prediction
- Predicts heart disease risk using patient medical parameters.
- Uses a Machine Learning classification model.

### 🩸 Diabetes Prediction
- Predicts diabetes possibility based on health-related features.

### 🧠 Parkinson Disease Prediction
- Uses voice measurement features to predict Parkinson disease.

### 🎨 User Interface
- Clean Streamlit dashboard
- Sidebar navigation
- Multiple prediction modules
- Interactive input forms
- Real-time results

---

# 🛠️ Technologies Used

## Programming Language
- Python

## Framework
- Streamlit

## Machine Learning
- Scikit-learn
- Classification Algorithms

## Data Processing
- Pandas
- NumPy

## Model Saving
- Joblib

---

# 📂 Project Structure


Multiple-Disease-Prediction-System

│
├── streamlit_app.py # Main Streamlit application
│
├── models/
│ ├── heart_model.pkl # Heart disease model
│ ├── diabetes_model.pkl # Diabetes model
│ └── parkinson_model.pkl # Parkinson model
│
├── dataset/
│ ├── heart.csv
│ ├── diabetes.csv
│ └── parkinson.csv
│
├── training/
│ └── Model training files
│
├── analysis/
│ └── Exploratory Data Analysis
│
├── requirements.txt
│
└── README.md


---

# ⚙️ Installation and Setup

Clone the repository:

```bash
git clone https://github.com/ananyadevanga139-jpg/Multiple-Disease-Prediction-System.git

Go inside the project:

cd Multiple-Disease-Prediction-System

Create virtual environment:

python -m venv venv

Activate environment:

Windows:

venv\Scripts\activate

Install dependencies:

pip install -r requirements.txt
▶️ Run Application

Start Streamlit:

streamlit run streamlit_app.py

The application will open in your browser:

http://localhost:8501
📊 Machine Learning Workflow
Data Collection
Data Preprocessing
Exploratory Data Analysis
Model Training
Model Evaluation
Model Saving using Joblib
Streamlit Deployment
📈 Models
Disease	Model
Heart Disease	Machine Learning Classification Model
Diabetes	Machine Learning Classification Model
Parkinson Disease	Machine Learning Classification Model
🎯 Future Enhancements
Add deep learning models
Add patient history storage
Add doctor recommendation system
Improve UI design
Add cloud database integration
👩‍💻 Developer

Ananya K
