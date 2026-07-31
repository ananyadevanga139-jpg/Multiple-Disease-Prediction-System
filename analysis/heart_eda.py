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


# Split input and output

X = data.drop(
    "num",
    axis=1
)

y = data["num"]



# Encode text columns

encoder = LabelEncoder()


for column in X.select_dtypes(include=["object", "str"]).columns:

    X[column] = encoder.fit_transform(
        X[column].astype(str)
    )


# Handle missing values

imputer = SimpleImputer(
    strategy="most_frequent"
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



# Create Model

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)



# Train

model.fit(
    X_train,
    y_train
)



# Prediction

y_pred = model.predict(
    X_test
)



# Accuracy

accuracy = accuracy_score(
    y_test,
    y_pred
)


print(
    "Model Accuracy:",
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
        "models/heart_model.pkl",
        "wb"
    )
)


print(
    "Heart Model Saved Successfully"
)