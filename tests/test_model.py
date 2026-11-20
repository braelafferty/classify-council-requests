"""Model tests: checks the processed data and the saved model."""

from pathlib import Path

import joblib
import pandas as pd
import pytest
from sklearn.metrics import accuracy_score

# Build paths from this file's location so the tests work from any folder
ROOT = Path(__file__).resolve().parent.parent

cleaned = pd.read_csv(ROOT / "data" / "processed" / "cleaned_requests.csv")
train_data = pd.read_csv(ROOT / "data" / "processed" / "train.csv")
test_data = pd.read_csv(ROOT / "data" / "processed" / "test.csv")

model = joblib.load(ROOT / "model" / "model.joblib")
vectoriser = joblib.load(ROOT / "model" / "vectoriser.joblib")


# Test 1: the cleaned data has no duplicates or missing text
def test_cleaned_data_has_no_duplicates_or_missing_text():
    assert len(cleaned) == 60
    assert cleaned["cleaned_text"].duplicated().sum() == 0
    assert cleaned["cleaned_text"].isnull().sum() == 0


# Test 2: the data is split 80/20 (48 training rows and 12 test rows)
def test_split_is_80_20():
    assert len(train_data) == 48
    assert len(test_data) == 12


# Test 3: the saved model gets the 0.75 accuracy reported for the test set
def test_saved_model_accuracy():
    X_test_tfidf = vectoriser.transform(test_data["cleaned_text"])
    predictions = model.predict(X_test_tfidf)
    accuracy = accuracy_score(test_data["category"], predictions)
    print(f"\nAccuracy on test set: {accuracy}")
    assert accuracy == 0.75


# Test 4: every confidence score from predict_proba is between 0 and 1
def test_confidence_is_between_0_and_1():
    X_test_tfidf = vectoriser.transform(test_data["cleaned_text"])
    confidence = model.predict_proba(X_test_tfidf).max(axis=1)
    print(f"\nConfidence range: {confidence.min():.3f} to {confidence.max():.3f}")
    assert (confidence >= 0).all()
    assert (confidence <= 1).all()


# Run with: python test_model.py  (or: pytest -v -s  from the project folder)
if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])