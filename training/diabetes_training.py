import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.impute import SimpleImputer


# Load dataset

data = pd.read_csv(
    "dataset/diabetes.csv"
)


print("Diabetes Dataset Loaded")


# Separate input and output

X = data.drop(
    "Outcome",
    axis=1
)

y = data["Outcome"]



# Handle missing values

imputer = SimpleImputer(
    strategy="mean"
)


X = pd.DataFrame(
    imputer.fit_transform(X),
    columns=X.columns
)



# Split dataset

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)



# Create model

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)



# Train

model.fit(
    X_train,
    y_train
)



# Predict

y_pred = model.predict(
    X_test
)



# Accuracy

accuracy = accuracy_score(
    y_test,
    y_pred
)


print(
    "Diabetes Model Accuracy:",
    accuracy
)


print(
    classification_report(
        y_test,
        y_pred
    )
)



# Save model

pickle.dump(
    model,
    open(
        "models/diabetes_model.pkl",
        "wb"
    )
)


print(
    "Diabetes Model Saved Successfully"
)