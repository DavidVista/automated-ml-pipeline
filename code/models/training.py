from abc import ABC, abstractmethod
from sklearn.base import BaseEstimator
from sklearn.metrics import accuracy_score, precision_score, recall_score, precision_recall_curve, roc_auc_score
import numpy as np
import pickle
import os
import matplotlib.pyplot as plt
import tempfile

# MlFlow-related imports
import mlflow
from mlflow.models import infer_signature
import mlflow.sklearn


class BaseModel(ABC):

    def __init__(self, name: str = "model"):
        self.name = name
        self.mlflow_experiment_name = "credit-risk-classification"

    @abstractmethod
    def train(self, X: np.array, y: np.array):
        pass

    @abstractmethod
    def evaluate(self, **kwargs):
        pass

    @abstractmethod
    def dump(self, models_dir: str = '/opt/airflow/models'):
        pass

    def setup_mlflow(self):
        """Setup MLflow tracking"""
        MLFLOW_URL = os.environ.get("MLFLOW_URL", "http://mlflow-server:5000")
        mlflow.set_tracking_uri(MLFLOW_URL)
        mlflow.set_experiment(self.mlflow_experiment_name)


class SklearnModel(BaseModel):

    def __init__(self, name: str, model: BaseEstimator, **kwargs):
        super().__init__(name)
        self.model = model(**kwargs)
        self.setup_mlflow()

    def train(self, X: np.array, y: np.array):
        with mlflow.start_run(run_name=f"{self.name}-training") as run:
            self.model.fit(X, y)

            # Log parameters
            mlflow.log_params(self.model.get_params())

            # Log training data info
            mlflow.log_metric("training_samples", X.shape[0])
            mlflow.log_metric("training_features", X.shape[1])

            return run.info.run_id

    def dump(self, run_id: str, models_dir: str = '/opt/airflow/models'):
        with mlflow.start_run(run_id=run_id, nested=True):
            os.makedirs(models_dir, exist_ok=True)
            model_path = os.path.join(models_dir, f'{self.name}.pkl')

            with open(model_path, 'wb') as model_file:
                pickle.dump(self.model, model_file)

            # Log model artifact
            mlflow.log_artifact(model_path, "models")

            # Also log as MLflow model
            signature = infer_signature(self.X_train, self.model.predict(self.X_train))
            mlflow.sklearn.log_model(
                self.model,
                f"{self.name}-mlflow-model",
                signature=signature,
                registered_model_name=self.name
            )


class GradientBoostingModel(SklearnModel):

    def __init__(self, name: str, model: BaseEstimator, X_train: np.array, **kwargs):
        super().__init__(name, model, **kwargs)
        self.X_train = X_train
        self.optimal_threshold = None

    def find_optimal_threshold(self, X_val: np.array, y_val: np.array, run_id: str):
        """Find optimal threshold on validation set"""
        with mlflow.start_run(run_id=run_id, nested=True):
            y_val_proba = self.model.predict_proba(X_val)
            precision, recall, thresholds = precision_recall_curve(y_val, y_val_proba[:, 1])

            f1_scores = 2 * (precision * recall) / (precision + recall + 1e-7)
            optimal_idx = np.argmax(f1_scores[:-1])  # Exclude last element

            self.optimal_threshold = thresholds[optimal_idx]
            mlflow.log_metric("optimal_threshold", self.optimal_threshold)

            return self.optimal_threshold

    def evaluate(self, X_test: np.array, y_test: np.array, run_id: str, use_optimal_threshold: bool = True):
        """
        Model evaluation with MLflow logging
        """
        with mlflow.start_run(run_id=run_id, nested=True):
            if use_optimal_threshold and self.optimal_threshold is None:
                raise ValueError(
                    "Optimal threshold not set. Call find_optimal_threshold() first or set use_optimal_threshold=False"
                )

            y_pred_proba = self.model.predict_proba(X_test)

            # Use optimal threshold if available, otherwise use default 0.5
            threshold = self.optimal_threshold if use_optimal_threshold else 0.5

            precision, recall, thresholds = precision_recall_curve(y_test, y_pred_proba[:, 1])

            # Create and save precision-recall curve
            plt.figure(figsize=(10, 6))
            plt.plot(thresholds, precision[:-1], color='red', label='precision')
            plt.plot(thresholds, recall[:-1], color='blue', label='recall')
            plt.xlabel('Threshold')
            plt.ylabel('Score')
            plt.title('Precision-Recall Curve')
            plt.grid()
            plt.legend()

            # Save plot to temporary file and log to MLflow
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_file:
                plt.savefig(tmp_file.name, dpi=300, bbox_inches='tight')
                mlflow.log_artifact(tmp_file.name, "plots")
                tmp_path = tmp_file.name
            plt.close()

            # Remove temp file
            os.unlink(tmp_path)

            y_pred = (y_pred_proba[:, 1] > threshold).astype(np.int32)

            # Calculate metrics
            metrics = {
                "accuracy": accuracy_score(y_test, y_pred),
                "precision": precision_score(y_test, y_pred, zero_division=0),
                "recall": recall_score(y_test, y_pred, zero_division=0),
                "roc_auc": roc_auc_score(y_test, y_pred_proba[:, 1])
            }

            # Log metrics to MLflow
            for metric_name, metric_value in metrics.items():
                mlflow.log_metric(metric_name, metric_value)

            # Log confusion matrix
            from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
            cm = confusion_matrix(y_test, y_pred)
            disp = ConfusionMatrixDisplay(confusion_matrix=cm)

            plt.figure(figsize=(8, 6))
            disp.plot(cmap='Blues')
            plt.title('Confusion Matrix')

            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_file:
                plt.savefig(tmp_file.name, dpi=300, bbox_inches='tight')
                mlflow.log_artifact(tmp_file.name, "plots")
                tmp_path_cm = tmp_file.name
            plt.close()
            os.unlink(tmp_path_cm)

            return metrics
