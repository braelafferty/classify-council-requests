# Flask API 

from pathlib import Path

from flask import Flask, request, send_from_directory
import joblib
app = Flask(__name__)

# Find the model folder from where this file is, so the API runs from any folder
MODEL_FOLDER = Path(__file__).parent.parent / "model"

# Load model and vectoriser 
model = joblib.load(MODEL_FOLDER / "model.joblib")
vectoriser = joblib.load(MODEL_FOLDER / "vectoriser.joblib")

# Predictions below this confidence are flagged for a staff member to check.
# On the test set this caught 2 of the 3 wrong predictions.
REVIEW_THRESHOLD = 0.50

# /Health route to show the app is working
@app.route("/health", methods=["GET"])
def health():
    return {"status": "ok"}

# / Categories endpoint to show all the endpoints 
@app.route("/categories", methods=["GET"])
def categories():
    return {"categories": model.classes_.tolist()}


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(silent=True)
    # If no data is sent
    if data is None:
        return {"error": "Please send valid JSON."}, 400
    if not isinstance(data, dict):
        return {
            "error": 'Send a JSON object like {"request_text": "missed rubbish collection"}.'
        }, 400
    text = data.get("request_text")
    # If request_text is empty error catch 
    if not isinstance(text, str) or text.strip() == "":
        return {"error": "request_text must contain some text."}, 400
    

    text = " ".join(text.lower().split())
    # Put the cleaned text to numbers TF-IDF
    features = vectoriser.transform([text])
    
    # Uses the model and use predict / predict_proba based on the numbers 
    predictions = model.predict(features)
    probabilities = model.predict_proba(features)
    confidence = probabilities.max(axis=1)
    # True if the model is unsure, so a staff member should check the category
    needs_review = bool(confidence[0] < REVIEW_THRESHOLD)
    # Returns back to user 
    return {
        "request_text": text,
        "predicted_category": str(predictions[0]),
        "confidence": float(confidence[0]),
        "needs_review": needs_review
    }


# Swagger page at /docs so the endpoints can be tested in the browser.
# The page loads Swagger UI from the internet and reads the endpoint list from openapi.json.
SWAGGER_PAGE = """<!DOCTYPE html>
<html>
<head>
  <title>Council Service Request Classifier API</title>
  <link rel="stylesheet" href="https://unpkg.com/swagger-ui-dist@5/swagger-ui.css">
</head>
<body>
  <div id="swagger-ui"></div>
  <script src="https://unpkg.com/swagger-ui-dist@5/swagger-ui-bundle.js"></script>
  <script>
    SwaggerUIBundle({ url: "/openapi.json", dom_id: "#swagger-ui" });
  </script>
</body>
</html>"""


@app.route("/docs", methods=["GET"])
def docs():
    return SWAGGER_PAGE


# openapi.json describes each endpoint and gives example inputs for Swagger
@app.route("/openapi.json", methods=["GET"])
def openapi_spec():
    return send_from_directory(Path(__file__).parent, "openapi.json")


if __name__ == "__main__":
    app.run(debug=True)