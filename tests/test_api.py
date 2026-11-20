"""API tests: checks that /health, /categories and /predict work."""

import os
import sys
from pathlib import Path

import pytest

# app.py loads the model from "../model", so run it from inside the api folder
API_FOLDER = Path(__file__).resolve().parent.parent / "api"
sys.path.insert(0, str(API_FOLDER))
os.chdir(API_FOLDER)

from app import app, REVIEW_THRESHOLD

# Flask's test client sends requests to the API without starting the server
client = app.test_client()


# Test 1: /health shows the API is running
def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.get_json() == {"status": "ok"}


# Test 2: /categories lists the four categories
def test_categories():
    response = client.get("/categories")
    assert response.status_code == 200
    assert response.get_json() == {"categories": ["road", "safety", "waste", "water"]}


# Test 3: /predict gives the right category, a confidence score and a needs_review flag for five valid requests
def test_predict_valid_requests():
    examples = [
        ("Large pothole on Colombo Street damaging cars", "road"),
        ("Streetlight broken outside the park, feels unsafe at night", "safety"),
        ("Missed rubbish collection outside my house", "waste"),
        ("Burst water pipe flooding the footpath", "water"),
        ("Rubbish bins overflowing at the bus stop", "waste"),
    ]
    for text, expected in examples:
        response = client.post("/predict", json={"request_text": text})
        result = response.get_json()
        print(f"\n{text} -> {result['predicted_category']} "
              f"(confidence {result['confidence']:.3f}, needs_review {result['needs_review']})")
        assert response.status_code == 200
        assert result["predicted_category"] == expected
        assert 0 <= result["confidence"] <= 1
        assert result["needs_review"] == (result["confidence"] < REVIEW_THRESHOLD)


# Test 4: an empty request_text returns a 400 error
def test_predict_empty_text():
    response = client.post("/predict", json={"request_text": ""})
    assert response.status_code == 400
    assert "error" in response.get_json()


# Test 5: a missing request_text field returns a 400 error
def test_predict_missing_field():
    response = client.post("/predict", json={})
    assert response.status_code == 400
    assert "error" in response.get_json()


# Run with: python test_api.py  (or: pytest -v -s  from the project folder)
if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
