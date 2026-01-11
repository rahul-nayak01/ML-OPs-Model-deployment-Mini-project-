import unittest
import mlflow
import os
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pickle


class TestModelLoading(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
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
        mlflow.set_tracking_uri(f"{dagshub_url}/{repo_owner}/{repo_name}.mlflow")

        # Load the new model (Staging)
        cls.new_model_name = "my_model"
        cls.new_model_version = cls.get_latest_model_version(cls.new_model_name, "Staging")
        if not cls.new_model_version:
            raise ValueError("No staged model found for testing")

        cls.new_model_uri = f"models:/{cls.new_model_name}/{cls.new_model_version}"
        cls.new_model = mlflow.pyfunc.load_model(cls.new_model_uri)

        # Try loading the production model (if available)
        cls.prod_model_version = cls.get_latest_model_version(cls.new_model_name, "Production")
        cls.prod_model = (
            mlflow.pyfunc.load_model(f"models:/{cls.new_model_name}/{cls.prod_model_version}")
            if cls.prod_model_version
            else None
        )

        # Load vectorizer
        cls.vectorizer = pickle.load(open("models/vectorizer.pkl", "rb"))

        # Load holdout test data
        cls.holdout_data = pd.read_csv("data/processed/test_bow.csv")

    @staticmethod
    def get_latest_model_version(model_name, stage):
        client = mlflow.MlflowClient()
        versions = client.get_latest_versions(model_name, stages=[stage])
        return versions[0].version if versions else None

    def test_model_loaded_properly(self):
        self.assertIsNotNone(self.new_model)

    def test_model_signature(self):
        input_text = "hi how are you"
        input_data = self.vectorizer.transform([input_text])
        input_df = pd.DataFrame(input_data.toarray(), columns=[str(i) for i in range(input_data.shape[1])])
        prediction = self.new_model.predict(input_df)

        self.assertEqual(input_df.shape[1], len(self.vectorizer.get_feature_names_out()))
        self.assertEqual(len(prediction), input_df.shape[0])
        self.assertEqual(len(prediction.shape), 1)

    def test_model_performance_and_compare(self):
        X_holdout = self.holdout_data.iloc[:, :-1]
        y_holdout = self.holdout_data.iloc[:, -1]

        # Evaluate new model
        y_pred_new = self.new_model.predict(X_holdout)
        metrics_new = {
            "accuracy": accuracy_score(y_holdout, y_pred_new),
            "precision": precision_score(y_holdout, y_pred_new),
            "recall": recall_score(y_holdout, y_pred_new),
            "f1": f1_score(y_holdout, y_pred_new)
        }

        print("\n New (Staging) model metrics:")
        for k, v in metrics_new.items():
            print(f"{k.capitalize()}: {v:.4f}")

        # Compare with production model (if exists)
        if self.prod_model:
            y_pred_old = self.prod_model.predict(X_holdout)
            metrics_old = {
                "accuracy": accuracy_score(y_holdout, y_pred_old),
                "precision": precision_score(y_holdout, y_pred_old),
                "recall": recall_score(y_holdout, y_pred_old),
                "f1": f1_score(y_holdout, y_pred_old)
            }

            print("\n Production model metrics:")
            for k, v in metrics_old.items():
                print(f"{k.capitalize()}: {v:.4f}")

            #  The core gating logic: fail test if new model underperforms
            for metric in ["accuracy", "precision"]:
                self.assertGreaterEqual(
                    metrics_new[metric],
                    metrics_old[metric],
                    f"❌ New model {metric}={metrics_new[metric]:.4f} "
                    f"is worse than production {metric}={metrics_old[metric]:.4f}"
                )
        else:
            print("\n No Production model found for comparison. Skipping comparison tests.")


if __name__ == "__main__":
    unittest.main()