"""
Step 3 - Model building with experiment tracking (MLflow).

* loads the train / test splits produced by prep.py
* defines a preprocessing + XGBoost pipeline and a hyper-parameter grid
* tunes the model with GridSearchCV (5-fold stratified CV)
* logs every tuned parameter combination, the best parameters and the
  evaluation metrics to MLflow
* evaluates the best model on train and test data
* saves the best model (and a small metrics file) into
  tourism_project/deployment/ so the workflow can commit it to the repo
  and the Streamlit app can load it.
"""
import contextlib
import io
import json
import os
import sys
import urllib.request
import warnings
from datetime import datetime, timezone
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
import xgboost as xgb
from sklearn.compose import make_column_transformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (accuracy_score, classification_report, f1_score,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

warnings.filterwarnings("ignore")

# ----------------------------------------------------------------------------
# Paths & constants
# ----------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[2]
SPLIT_DIR = ROOT                                    # where prep.py wrote the CSVs
DEPLOY_DIR = ROOT / "tourism_project" / "deployment"
MODEL_PATH = DEPLOY_DIR / "best_tourism_model_v1.joblib"
METRICS_PATH = DEPLOY_DIR / "model_metrics.json"

EXPERIMENT_NAME = "tourism-wellness-package"
RANDOM_STATE = 42
CV_FOLDS = 5
SCORING = "f1"                      # balances precision and recall (19% positives)
CLASSIFICATION_THRESHOLD = 0.50

NUMERIC_FEATURES = [
    "Age", "CityTier", "DurationOfPitch", "NumberOfPersonVisiting",
    "NumberOfFollowups", "PreferredPropertyStar", "NumberOfTrips", "Passport",
    "PitchSatisfactionScore", "OwnCar", "NumberOfChildrenVisiting", "MonthlyIncome",
]
CATEGORICAL_FEATURES = [
    "TypeofContact", "Occupation", "Gender", "ProductPitched",
    "MaritalStatus", "Designation",
]

PARAM_GRID = {
    "xgbclassifier__n_estimators": [100, 200],
    "xgbclassifier__max_depth": [3, 5, 7],
    "xgbclassifier__learning_rate": [0.05, 0.10],
    "xgbclassifier__subsample": [0.8, 1.0],
    "xgbclassifier__colsample_bytree": [0.7, 1.0],
}


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------
def setup_mlflow() -> str:
    """Use the MLflow server started by the workflow; fall back to a local folder."""
    uri = os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000")
    try:
        urllib.request.urlopen(uri, timeout=3)
        mlflow.set_tracking_uri(uri)
    except Exception:
        uri = (ROOT / "mlruns").as_uri()
        print(f"[info] MLflow server not reachable - logging to local folder {uri}")
        mlflow.set_tracking_uri(uri)
    mlflow.set_experiment(EXPERIMENT_NAME)
    return mlflow.get_tracking_uri()


def load_splits():
    needed = ["Xtrain.csv", "Xtest.csv", "ytrain.csv", "ytest.csv"]
    missing = [f for f in needed if not (SPLIT_DIR / f).exists()]
    if missing:
        sys.exit(f"[ERROR] Missing split files {missing} in {SPLIT_DIR}. Run prep.py first.")
    Xtrain = pd.read_csv(SPLIT_DIR / "Xtrain.csv")
    Xtest = pd.read_csv(SPLIT_DIR / "Xtest.csv")
    ytrain = pd.read_csv(SPLIT_DIR / "ytrain.csv").squeeze("columns")
    ytest = pd.read_csv(SPLIT_DIR / "ytest.csv").squeeze("columns")
    return Xtrain, Xtest, ytrain, ytest


def build_pipeline(scale_pos_weight: float):
    preprocessor = make_column_transformer(
        (make_pipeline(SimpleImputer(strategy="median"), StandardScaler()), NUMERIC_FEATURES),
        (make_pipeline(SimpleImputer(strategy="most_frequent"),
                       OneHotEncoder(handle_unknown="ignore")), CATEGORICAL_FEATURES),
    )
    model = xgb.XGBClassifier(
        scale_pos_weight=scale_pos_weight,   # compensates for the class imbalance
        eval_metric="logloss",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    return make_pipeline(preprocessor, model)


def evaluate(model, X, y) -> dict:
    proba = model.predict_proba(X)[:, 1]
    pred = (proba >= CLASSIFICATION_THRESHOLD).astype(int)
    return {
        "accuracy": float(accuracy_score(y, pred)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred)),
        "f1": float(f1_score(y, pred)),
        "roc_auc": float(roc_auc_score(y, proba)),
    }


def clean_param_names(params: dict) -> dict:
    return {k.split("__", 1)[1]: v for k, v in params.items()}


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def main() -> None:
    Xtrain, Xtest, ytrain, ytest = load_splits()
    print(f"Train: {Xtrain.shape} | Test: {Xtest.shape}")

    scale_pos_weight = float((ytrain == 0).sum() / (ytrain == 1).sum())
    print(f"scale_pos_weight (neg/pos) = {scale_pos_weight:.2f}")

    tracking_uri = setup_mlflow()
    print(f"MLflow tracking URI: {tracking_uri}")

    pipeline = build_pipeline(scale_pos_weight)
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    grid = GridSearchCV(pipeline, PARAM_GRID, scoring=SCORING, cv=cv, n_jobs=-1, refit=True)

    n_candidates = 1
    for values in PARAM_GRID.values():
        n_candidates *= len(values)

    with mlflow.start_run(run_name="xgboost-grid-search"):
        mlflow.log_params({
            "model": "XGBClassifier", "cv_folds": CV_FOLDS, "scoring": SCORING,
            "n_candidates": n_candidates, "scale_pos_weight": round(scale_pos_weight, 3),
            "classification_threshold": CLASSIFICATION_THRESHOLD,
            "random_state": RANDOM_STATE,
        })

        print(f"\nTuning {n_candidates} parameter combinations x {CV_FOLDS} folds ...")
        grid.fit(Xtrain, ytrain)

        # ---- log every tuned parameter combination as a nested run ----------
        results = grid.cv_results_
        with contextlib.redirect_stdout(io.StringIO()):  # silence MLflow's "View run ..." lines
            for i, params in enumerate(results["params"]):
                with mlflow.start_run(run_name=f"candidate-{i:03d}", nested=True):
                    mlflow.log_params(clean_param_names(params))
                    mlflow.log_metrics({
                        f"mean_cv_{SCORING}": float(results["mean_test_score"][i]),
                        f"std_cv_{SCORING}": float(results["std_test_score"][i]),
                        "rank": int(results["rank_test_score"][i]),
                    })
        print(f"Logged {len(results['params'])} tuned parameter combinations to MLflow (nested runs).")

        # ---- best model ------------------------------------------------------
        best_model = grid.best_estimator_
        best_params = clean_param_names(grid.best_params_)
        mlflow.log_params({f"best_{k}": v for k, v in best_params.items()})
        mlflow.log_metric(f"best_cv_{SCORING}", float(grid.best_score_))

        train_metrics = evaluate(best_model, Xtrain, ytrain)
        test_metrics = evaluate(best_model, Xtest, ytest)
        mlflow.log_metrics({f"train_{k}": v for k, v in train_metrics.items()})
        mlflow.log_metrics({f"test_{k}": v for k, v in test_metrics.items()})

        test_pred = (best_model.predict_proba(Xtest)[:, 1] >= CLASSIFICATION_THRESHOLD).astype(int)
        train_pred = (best_model.predict_proba(Xtrain)[:, 1] >= CLASSIFICATION_THRESHOLD).astype(int)
        report_text = (
            "TRAIN\n" + classification_report(ytrain, train_pred) +
            "\nTEST\n" + classification_report(ytest, test_pred)
        )
        mlflow.log_text(report_text, "classification_report.txt")

        try:  # logging the model artifact is a nice-to-have, never fail the pipeline for it
            mlflow.sklearn.log_model(best_model, name="model", input_example=Xtrain.head(5))
        except Exception as exc:  # noqa: BLE001
            print(f"[warn] Could not log model artifact to MLflow: {exc}")

    # ---- console report -----------------------------------------------------
    top = (pd.DataFrame(results["params"]).rename(columns=lambda c: c.split("__", 1)[1])
           .assign(mean_cv_score=results["mean_test_score"], rank=results["rank_test_score"])
           .sort_values("rank").head(5))
    print("\nTop 5 parameter combinations (by mean CV %s):" % SCORING)
    print(top.to_string(index=False))

    print("\nBest parameters:", best_params)
    print(f"Best mean CV {SCORING}: {grid.best_score_:.4f}")
    print("\n--- TRAIN classification report ---")
    print(classification_report(ytrain, train_pred))
    print("--- TEST classification report ---")
    print(classification_report(ytest, test_pred))

    # ---- save model + metrics so the workflow can commit them ---------------
    DEPLOY_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_model, MODEL_PATH, compress=3)

    metrics = {
        "model": "XGBClassifier",
        "trained_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
        "classification_threshold": CLASSIFICATION_THRESHOLD,
        "best_params": best_params,
        f"best_cv_{SCORING}": float(grid.best_score_),
        "train_metrics": train_metrics,
        "test_metrics": test_metrics,
        "n_train": int(len(Xtrain)),
        "n_test": int(len(Xtest)),
    }
    METRICS_PATH.write_text(json.dumps(metrics, indent=2))

    print(f"\nSaved model   -> {MODEL_PATH.relative_to(ROOT)}")
    print(f"Saved metrics -> {METRICS_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
