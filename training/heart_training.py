import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report


# Load dataset
data = pd.read_csv(
    "dataset/heart.csv"
)

print("Dataset Loaded")


# Convert target
# 0 = Healthy
# 1,2,3,4 = Disease

data["num"] = data["num"].apply(
    lambda x: 0 if x == 0 else 1
)


# Remove ID column

data.drop(
    "id",
    axis=1,
    inplace=True
)


# Separate features and target

X = data.drop(
    "num",
    axis=1
)

y = data["num"]


# Encode categorical data

encoder = LabelEncoder()

for col in X.select_dtypes(include=["object","str"]).columns:
    X[col] = encoder.fit_transform(
        X[col].astype(str)
    )


# Handle missing values

imputer = SimpleImputer(
    strategy="most_frequent"
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


# Train model

model.fit(
    X_train,
    y_train
)


# Test model

prediction = model.predict(
    X_test
)


accuracy = accuracy_score(
    y_test,
    prediction
)


print(
    "Heart Disease Model Accuracy:",
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
        "models/heart_model.pkl",
        "wb"
    )
)


print(
    "Heart Model Saved Successfully"
)