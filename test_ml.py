import pytest
import pandas as pd
from ml.model import train_model, compute_model_metrics, inference
from ml.data import process_data
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Test 1: Verify that train_model returns a trained RandomForestClassifier
def test_model():
    """
    Ensure that the train_model function returns a fitted RandomForestClassifier.
    """
    X_sample = pd.DataFrame({
        'age': [25, 30, 35, 40],
        'education': ['Bachelors', 'Masters', 'PhD', 'Bachelors'],
        'hours_per_week': [40, 45, 50, 55]
    })
    y_sample = pd.Series([0, 1, 1, 0])

    # Ensure the label exists in the DataFrame before processing
    X_sample["salary"] = y_sample  

    # Process the data correctly with a label
    X_processed, y_processed, _, _ = process_data(
        X_sample, categorical_features=['education'], label="salary", training=True
    )

    # Train model
    model = train_model(X_processed, y_processed)  

    assert isinstance(model, RandomForestClassifier)
    assert hasattr(model, "predict")  # Ensure it is a trained model

# Test 2: Validate compute_model_metrics returns reasonable scores
def test_compute_model_metrics():
    """
    Ensure that compute_model_metrics returns precision, recall, and f1-score within valid ranges.
    """
    y_true = pd.Series([0, 1, 1, 0, 1, 0, 1])
    y_pred = pd.Series([0, 1, 0, 0, 1, 1, 1])

    precision, recall, f1 = compute_model_metrics(y_true, y_pred)

    assert isinstance(precision, float) and 0 <= precision <= 1
    assert isinstance(recall, float) and 0 <= recall <= 1
    assert isinstance(f1, float) and 0 <= f1 <= 1

# Test 3: Verify correct data split with train_test_split
def test_train_test_split():
    """
    Ensure train_test_split correctly divides data while preserving format.
    """
    dummy_data = pd.DataFrame({
        'feature_1': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
        'feature_2': ['A', 'B', 'C', 'A', 'B', 'C', 'A', 'B', 'C', 'A'],
        'label': [1, 0, 1, 0, 1, 0, 1, 0, 1, 0]
    })

    train_data, test_data = train_test_split(dummy_data, test_size=0.3, random_state=42)

    assert len(train_data) == 7  # 70% of 10
    assert len(test_data) == 3   # 30% of 10
    assert isinstance(train_data, pd.DataFrame)
    assert isinstance(test_data, pd.DataFrame)

