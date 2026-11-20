
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, classification_report
import joblib
from pathlib import Path


train_data = pd.read_csv("../data/processed/train.csv")
test_data = pd.read_csv("../data/processed/test.csv")


X_train = train_data["cleaned_text"]
y_train = train_data["category"]

X_test = test_data["cleaned_text"]
y_test = test_data["category"]


vectoriser = TfidfVectorizer(ngram_range=(1, 1))
X_train_tfidf = vectoriser.fit_transform(X_train)
X_test_tfidf = vectoriser.transform(X_test)


model = CalibratedClassifierCV(LinearSVC(C=0.1, random_state=42), cv=3)
model.fit(X_train_tfidf, y_train)

predictions = model.predict(X_test_tfidf)
accuracy = accuracy_score(y_test, predictions)

print("\nModel: Calibrated LinearSVC (C=0.1)")
print("TF-IDF: Unigrams")
print("\nAccuracy:", accuracy)
print("\nClassification report:")
print(classification_report(y_test, predictions, zero_division=0))

probabilities = model.predict_proba(X_test_tfidf)

confidence = probabilities.max(axis=1)

results = pd.DataFrame({
    "request_text": test_data["request_text"],
    "true_label": y_test,
    "predicted_label": predictions,
    "confidence": confidence
})

print("\nFive example predictions:")
print(results.head(5).to_string(index=False))

print("\nMean confidence on test set:", round(confidence.mean(), 3))

Path("../model").mkdir(parents=True, exist_ok=True)
Path("../outputs").mkdir(parents=True, exist_ok=True)

# Save model
joblib.dump(model, "../model/model.joblib")
joblib.dump(vectoriser, "../model/vectoriser.joblib")

results.to_csv("../outputs/test_predictions.csv", index=False)

results.head(5).to_csv(
    "../outputs/five_example_predictions.csv",
    index=False
)

print("\nPredictions saved to ../outputs/*")
