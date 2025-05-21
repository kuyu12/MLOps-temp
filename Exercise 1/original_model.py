"""
Original model training script without MLflow
This is a typical data science script that trains a classification model
"""

import pandas as pd
import numpy as np
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import joblib
import os
from datetime import datetime

# Configuration
RANDOM_STATE = 42
TEST_SIZE = 0.2
N_ESTIMATORS = 100
MAX_DEPTH = 10
MIN_SAMPLES_SPLIT = 5
MIN_SAMPLES_LEAF = 2


def load_and_preprocess_data():
    """Load wine dataset and preprocess it"""
    print("Loading wine dataset...")
    data = load_wine()

    # Create DataFrame
    df = pd.DataFrame(data.data, columns=data.feature_names)
    df['target'] = data.target

    # Check for missing values
    print(f"Missing values: {df.isnull().sum().sum()}")

    # Separate features and target
    X = df.drop('target', axis=1)
    y = df['target']

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print(f"Training samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")
    print(f"Number of features: {X.shape[1]}")
    print(f"Number of classes: {len(np.unique(y))}")

    return X_train_scaled, X_test_scaled, y_train, y_test, scaler, data.feature_names


def train_model(X_train, y_train):
    """Train Random Forest classifier"""
    print("\nTraining Random Forest model...")

    model = RandomForestClassifier(
        n_estimators=N_ESTIMATORS,
        max_depth=MAX_DEPTH,
        min_samples_split=MIN_SAMPLES_SPLIT,
        min_samples_leaf=MIN_SAMPLES_LEAF,
        random_state=RANDOM_STATE,
        n_jobs=-1
    )

    # Train the model
    model.fit(X_train, y_train)

    # Get feature importance
    feature_importance = model.feature_importances_

    return model, feature_importance


def evaluate_model(model, X_test, y_test):
    """Evaluate the trained model"""
    print("\nEvaluating model...")

    # Make predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)

    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    conf_matrix = confusion_matrix(y_test, y_pred)

    # Print results
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")
    print(f"\nConfusion Matrix:\n{conf_matrix}")

    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1
    }

    return metrics, y_pred, y_pred_proba

def main():
    """Main training pipeline"""
    print("Wine Quality Classification Model Training")
    print("=" * 40)

    # Load and preprocess data
    X_train, X_test, y_train, y_test, scaler, feature_names = load_and_preprocess_data()

    # Train model
    model, feature_importance = train_model(X_train, y_train)

    # Evaluate model
    metrics, predictions, probabilities = evaluate_model(model, X_test, y_test)
    print("\nTraining completed successfully!")

    # Quick test prediction
    print("\nTest prediction on first sample:")
    sample = X_test[0].reshape(1, -1)
    prediction = model.predict(sample)[0]
    probability = model.predict_proba(sample)[0]
    print(f"Predicted class: {prediction}")
    print(f"Probabilities: {probability}")


if __name__ == "__main__":
    main()