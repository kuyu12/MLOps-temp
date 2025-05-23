import pandas as pd
from sklearn import datasets
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

import mlflow
from mlflow.models import infer_signature


mlflow.set_tracking_uri(uri="http://127.0.0.1:5001")

mlflow.set_experiment("MLflow autolog")

import mlflow

mlflow.autolog()

with mlflow.start_run():
    # Load the Iris dataset
    X, y = datasets.load_iris(return_X_y=True)

    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Define the model hyperparameters
    params = {"solver": "lbfgs", "max_iter": 1000, "multi_class": "auto", "random_state": 8888}

    # Train the model
    lr = LogisticRegression(**params)
    lr.fit(X_train, y_train)

    # Predict on the test set
    y_pred = lr.predict(X_test)

    # Calculate accuracy as a target loss metric
    accuracy = accuracy_score(y_test, y_pred)
