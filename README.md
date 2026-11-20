# Council Service Request Classifier

BCIT601 Introduction to AI and Machine Learning – Assessment 2, Final Submission
Brae Lafferty

A prototype that reads a council service request written by a resident, predicts its category
(**road**, **safety**, **waste** or **water**) and returns the prediction with a confidence score
through a Flask API. It is **decision support**: uncertain predictions are flagged for review and
a staff member confirms the final category.

## Results

Held-out test set (12 requests, 3 per category):

| Model | Accuracy | Macro F1 | Road recall | Mean confidence |
|---|---|---|---|---|
| Logistic Regression (baseline) | 0.75 | 0.73 | 0.33 | 0.335 |
| Multinomial Naive Bayes | 0.67 | 0.57 | 0.00 | 0.343 |
| **Calibrated LinearSVC (final)** | **0.75** | **0.75** | **0.67** | **0.517** |

The final model doubles road recall and gives more spread-out confidence scores. The test set has
only 12 requests, so one request changes accuracy by about 8 percentage points and the dataset
(60 unique requests) is the main limit.

## Repository structure

```
data/
  raw/                     BCIT601_starter_service_requests_dataset.csv (80 records)
  processed/               cleaned_requests.csv (60), train.csv (48), test.csv (12)
notebooks_or_scripts/
  prepare_data.py          load, clean, remove duplicates, stratified 80/20 split
  train_model.py           TF-IDF + calibrated LinearSVC; saves the model with joblib
  evaluate_models.py       trains and tests all three models the same way and prints the results
model/                     model.joblib, vectoriser.joblib
api/
  app.py                   Flask API: /health, /categories, /predict, plus Swagger docs at /docs
  openapi.json             endpoint descriptions and example inputs used by the Swagger page
tests/
  test_api.py              5 API tests
  test_model.py            4 data and model tests
outputs/                   predictions, evaluation log and test results
requirements.txt
README.md
```

## Setup

Python 3.13 was used.

```
python -m venv .venv
.venv\Scripts\activate          # Windows  (macOS/Linux: source .venv/bin/activate)
pip install -r requirements.txt
```

## Run the pipeline

The data and training scripts use paths relative to their own folder, so run them from inside it.
The model files are already in the repository, so this step is only needed to rebuild them.

```
cd notebooks_or_scripts
python prepare_data.py       # writes data/processed/*.csv
python train_model.py        # writes model/*.joblib and outputs/test_predictions.csv
python evaluate_models.py    # prints accuracy, classification report and example predictions for each model
cd ..
```

## Run the API

```
python api/app.py            # serves http://127.0.0.1:5000
```

The easiest way to test it is the Swagger page at **http://127.0.0.1:5000/docs**. Pick an endpoint, click
**Try it out**, choose one of the example inputs (valid, te reo Māori, off-topic, empty or missing text) and
click **Execute** to see the response. The page loads Swagger UI from the internet, so the browser needs a connection.

You can also use curl 
in a second terminal (on Windows PowerShell type `curl.exe`, not `curl`):

```
curl http://127.0.0.1:5000/health
curl http://127.0.0.1:5000/categories
curl -X POST http://127.0.0.1:5000/predict -H "Content-Type: application/json" -d "{\"request_text\": \"Missed rubbish collection outside my house\"}"
```

### Endpoints

| Endpoint | Method | Returns |
|---|---|---|
| `/health` | GET | `{"status": "ok"}` |
| `/categories` | GET | `{"categories": ["road", "safety", "waste", "water"]}` |
| `/predict` | POST | the cleaned text, `predicted_category`, `confidence` and `needs_review` |
| `/docs` | GET | Swagger page for trying the endpoints in the browser |
| `/openapi.json` | GET | the API description that the Swagger page reads |

Example `/predict` response:

```json
{
  "request_text": "missed rubbish collection outside my house",
  "predicted_category": "waste",
  "confidence": 0.6805377702687984,
  "needs_review": false
}
```

`confidence` is the highest class probability from `predict_proba`. `needs_review` is `true` when
confidence is below 0.50, the threshold that caught 2 of the 3 wrong predictions on the test set.

### Errors

Errors return a JSON `error` message with status 400.

| Problem | Message |
|---|---|
| Body is not valid JSON, or has no `Content-Type: application/json` header | Please send valid JSON. |
| JSON is not an object (for example a list) | Send a JSON object like {"request_text": "missed rubbish collection"}. |
| `request_text` is missing, not a string, empty or only spaces | request_text must contain some text. |

## Tests

```
pytest -v -s
```

9 tests: 5 API tests (`/health`, `/categories`, five valid requests to `/predict`, an empty
`request_text` and a missing `request_text`) and 4 data and model tests (no duplicates or missing
text, 80/20 split, saved model accuracy of 0.75, confidence between 0 and 1). Each file can also be
run on its own from the `tests` folder with `python test_api.py` or `python test_model.py`.
The latest run is saved in `outputs/test_results.txt`.

## Ethics and limitations

- The starter data is synthetic, balanced and contains no te reo Māori. A te reo Māori request
  ("Kua pakaru te paipa wai i te huarahi i Ōtautahi") is misclassified with low confidence (0.29),
  so it is flagged for review.
- Requests outside the four categories still receive a category ("What time does the library open
  on Sunday?" is predicted as safety at 0.37); low confidence flags them.
- Free text can contain personal information. The API does not store request text.
- Before any real use: retrain on de-identified council data gathered under a Māori data
  governance agreement with mana whenua (te Tiriti o Waitangi, Te Mana Raraunga and CARE
  principles), report performance per category and for te reo Māori, and keep staff review.

The final report covers these points, the full evaluation and the testing in detail.
