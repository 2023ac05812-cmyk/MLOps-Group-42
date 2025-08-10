# src/train.py
import os
import mlflow
import mlflow.sklearn
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

MLFLOW_URI = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
mlflow.set_tracking_uri(MLFLOW_URI)

def load_data():
    iris = load_iris(as_frame=True)
    X = iris.data
    y = iris.target
    return train_test_split(X, y, test_size=0.2, random_state=42)

def train_and_log():
    X_train, X_test, y_train, y_test = load_data()

    models = {
        "logreg": LogisticRegression(max_iter=200),
        "rf": RandomForestClassifier(n_estimators=100, random_state=42)
    }
    best = {"name": None, "acc": 0, "run_id": None, "model_uri": None}

    for name, model in models.items():
        with mlflow.start_run(run_name=name) as run:
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            acc = accuracy_score(y_test, preds)

            mlflow.log_param("model", name)
            mlflow.log_metric("accuracy", acc)
            # save model artifact
            mlflow.sklearn.log_model(model, "model")
            run_id = run.info.run_id
            model_uri = f"runs:/{run_id}/model"

            if acc > best["acc"]:
                best.update({"name": name, "acc": acc, "run_id": run_id, "model_uri": model_uri})

            print(f"Run {name}: acc={acc:.4f} run_id={run_id}")

    print("Best:", best)
    # register best model (requires MLflow server with registry support)
    try:
        result = mlflow.register_model(best["model_uri"], "IrisModel")
        print("Registered model:", result.name, result.version)
    except Exception as e:
        print("Model registration failed (ok in local mlruns):", e)

    # also save locally as fallback
    joblib.dump(models[best["name"]], "models/best_model.pkl")
    print("Saved models/best_model.pkl")

if __name__ == "__main__":
    os.makedirs("models", exist_ok=True)
    train_and_log()
