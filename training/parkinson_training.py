import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report


# Load dataset

data = pd.read_csv(
    "dataset/parkinson.csv"
)


print("Parkinson Dataset Loaded")


# Remove unnecessary column if exists

if "name" in data.columns:
    data.drop(
        "name",
        axis=1,
        inplace=True
    )


# Split input and output

X = data.drop(
    "status",
    axis=1
)

y = data["status"]



# Handle missing values

imputer = SimpleImputer(
    strategy="mean"
)


X = pd.DataFrame(
    imputer.fit_transform(X),
    columns=X.columns
)



# Split data

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)



# Create SVM model

model = SVC(
    kernel="linear"
)



# Train

model.fit(
    X_train,
    y_train
)



# Prediction

prediction = model.predict(
    X_test
)



# Accuracy

accuracy = accuracy_score(
    y_test,
    prediction
)


print(
    "Parkinson Model Accuracy:",
    accuracy
)


print(
    classification_report(
        y_test,
        prediction
    )
)



# Save model

pickle.dump(
    model,
    open(
        "models/parkinson_model.pkl",
        "wb"
    )
)


print(
    "Parkinson Model Saved Successfully"
)