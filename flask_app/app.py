from flask import Flask, render_template, request
import mlflow
import pickle
import os
import pandas as pd
from prometheus_client import Counter, Histogram, generate_latest, CollectorRegistry, CONTENT_TYPE_LATEST
import time
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import string
import re
import dagshub
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score , roc_auc_score

import warnings
warnings.simplefilter("ignore", UserWarning)
warnings.filterwarnings("ignore")

def lemmatization(text):
    """Lemmatize the text."""
    lemmatizer = WordNetLemmatizer()
    text = text.split()
    text = [lemmatizer.lemmatize(word) for word in text]
    return " ".join(text)

def remove_stop_words(text):
    """Remove stop words from the text."""
    stop_words = set(stopwords.words("english"))
    text = [word for word in str(text).split() if word not in stop_words]
    return " ".join(text)

def removing_numbers(text):
    """Remove numbers from the text."""
    text = ''.join([char for char in text if not char.isdigit()])
    return text

def lower_case(text):
    """Convert text to lower case."""
    text = text.split()
    text = [word.lower() for word in text]
    return " ".join(text)

def removing_punctuations(text):
    """Remove punctuations from the text."""
    text = re.sub('[%s]' % re.escape(string.punctuation), ' ', text)
    text = text.replace('؛', "")
    text = re.sub('\s+', ' ', text).strip()
    return text

def removing_urls(text):
    """Remove URLs from the text."""
    url_pattern = re.compile(r'https?://\S+|www\.\S+')
    return url_pattern.sub(r'', text)

def remove_small_sentences(df):
    """Remove sentences with less than 3 words."""
    for i in range(len(df)):
        if len(df.text.iloc[i].split()) < 3:
            df.text.iloc[i] = np.nan

def normalize_text(text):
    text = lower_case(text)
    text = remove_stop_words(text)
    text = removing_numbers(text)
    text = removing_punctuations(text)
    text = removing_urls(text)
    text = lemmatization(text)

    return text

# Below code block is for local use
# -------------------------------------------------------------------------------------
#mlflow.set_tracking_uri('https://dagshub.com/rahul-nayak01/ML-OPs-Model-deployment-Mini-project-.mlflow')
#dagshub.init(repo_owner='rahul-nayak01', repo_name='ML-OPs-Model-deployment-Mini-project-', mlflow=True)
# -------------------------------------------------------------------------------------

# Below code block is for production use
# -------------------------------------------------------------------------------------
# Set up DagsHub credentials for MLflow tracking
dagshub_token = os.getenv("CAPSTONE_TEST")
if not dagshub_token:
    raise EnvironmentError("CAPSTONE_TEST environment variable is not set")

os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_token
os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token

dagshub_url = "https://dagshub.com"
repo_owner = "rahul-nayak01"
repo_name = "ML-OPs-Model-deployment-Mini-project-"
# Set up MLflow tracking URI
mlflow.set_tracking_uri(f'{dagshub_url}/{repo_owner}/{repo_name}.mlflow')
# -------------------------------------------------------------------------------------


# Initialize Flask app
app = Flask(__name__)

latest_input = None
latest_prediction = None
batch_true = []
batch_pred = []
BATCH_SIZE = 10

# from prometheus_client import CollectorRegistry

# Create a custom registry
registry = CollectorRegistry()

# Define your custom metrics using this registry
REQUEST_COUNT = Counter(
    "app_request_count", "Total number of requests to the app", ["method", "endpoint"], registry=registry
)
REQUEST_LATENCY = Histogram(
    "app_request_latency_seconds", "Latency of requests in seconds", ["endpoint"], registry=registry
)
PREDICTION_COUNT = Counter(
    "model_prediction_count", "Count of predictions for each class", ["prediction"], registry=registry
)

# ------------------------------------------------------------------------------------------
# Model and vectorizer setup
model_name = "my_model"
def get_latest_model_version(model_name):
    client = mlflow.MlflowClient()
    latest_version = client.get_latest_versions(model_name, stages=["Production"])
    if not latest_version:
        latest_version = client.get_latest_versions(model_name, stages=["None"])
    return latest_version[0].version if latest_version else None

model_version = get_latest_model_version(model_name)
model_uri = f'models:/{model_name}/{model_version}'
print(f"Fetching model from: {model_uri}")
model = mlflow.pyfunc.load_model(model_uri)
vectorizer = pickle.load(open('models/vectorizer.pkl', 'rb'))

# Routes
@app.route("/")
def home():
    REQUEST_COUNT.labels(method="GET", endpoint="/").inc()
    start_time = time.time()
    response = render_template("index.html", result=None)
    REQUEST_LATENCY.labels(endpoint="/").observe(time.time() - start_time)
    return response

@app.route("/predict", methods=["POST"])
def predict():
    global latest_input, latest_prediction, batch_true, batch_pred

    REQUEST_COUNT.labels(method="POST", endpoint="/predict").inc()
    start_time = time.time()

    # Inputs
    text = request.form["text"]
    true_label = int(request.form["true_label"])

    cleaned = normalize_text(text)

    # Vectorize
    features = vectorizer.transform([cleaned])
    features_df = pd.DataFrame(features.toarray(), columns=[str(i) for i in range(features.shape[1])])

    # Predict
    pred = int(model.predict(features_df)[0])

    # Store latest
    latest_input = text
    latest_prediction = pred

    # Add to batch
    batch_true.append(true_label)
    batch_pred.append(pred)

    aggregated_metrics = None

    # When batch completes
    if len(batch_pred) == BATCH_SIZE:
        aggregated_metrics = {
            "accuracy": round(accuracy_score(batch_true, batch_pred), 3),
            "precision": round(precision_score(batch_true, batch_pred, zero_division=0), 3),
            "recall": round(recall_score(batch_true, batch_pred, zero_division=0), 3),
            "f1_score": round(f1_score(batch_true, batch_pred, zero_division=0), 3)
        }

        # Fetch production metrics
        client = mlflow.MlflowClient()
        prod_version_info = client.get_latest_versions(model_name, stages=["Production"])
        
        if prod_version_info:
            prod_run_id = prod_version_info[0].run_id
            production_metrics = client.get_run(prod_run_id).data.metrics
        else:
            production_metrics = {}

        # Reset batch
        batch_true = []
        batch_pred = []

        return render_template(
            "index.html",
            result=pred,
            metrics=aggregated_metrics,
            production_metrics=production_metrics
        )
    
    return render_template("index.html", result=pred)


@app.route("/metrics", methods=["GET"])
def metrics():
    """Expose only custom Prometheus metrics."""
    return generate_latest(registry), 200, {"Content-Type": CONTENT_TYPE_LATEST}

@app.route("/dashboard", methods=["GET"])
def dashboard():
    """Dashboard to compare unseen input evaluation with production model metrics."""
    REQUEST_COUNT.labels(method="GET", endpoint="/dashboard").inc()
    start_time = time.time()

    client = mlflow.MlflowClient()

    # Fetch production model metrics
    prod_version_info = client.get_latest_versions(model_name, stages=["Production"])
    if prod_version_info:
        prod_version = prod_version_info[0].version
        prod_run_id = prod_version_info[0].run_id
        prod_metrics = client.get_run(prod_run_id).data.metrics
    else:
        prod_metrics = {"message": "No production model found."}

    # Evaluate model on unseen data (only if user gave input)
    unseen_metrics = {}
    if latest_input and latest_prediction is not None:
        try:
            # Transform the same input using vectorizer
            X_unseen = vectorizer.transform([normalize_text(latest_input)])
            y_pred = model.predict(X_unseen)

            # Simulate a true label assumption
            # In real-time systems, you'd log and later compare with ground truth
            y_true = [latest_prediction]  # placeholder — you can replace this later

            unseen_metrics = {
                "accuracy": float(accuracy_score(y_true, y_pred)),
                "precision": float(precision_score(y_true, y_pred, average='binary', zero_division=0)),
                "recall": float(recall_score(y_true, y_pred, average='binary', zero_division=0)),
                "f1_score": float(f1_score(y_true, y_pred, average='binary', zero_division=0))
                #"auc": float(roc_auc_score(y_true, y_pred, average='binary'))
            }
        except Exception as e:
            unseen_metrics = {"error": str(e)}

    REQUEST_LATENCY.labels(endpoint="/dashboard").observe(time.time() - start_time)

    return render_template(
        "dashboard.html",
        latest_input=latest_input,
        latest_prediction=latest_prediction,
        unseen_metrics=unseen_metrics,
        prod_metrics=prod_metrics,
        model_version=model_version,
    )


if __name__ == "__main__":
    # app.run(debug=True) # for local use
    app.run(debug=True, host="0.0.0.0", port=5000)  # Accessible from outside Docker