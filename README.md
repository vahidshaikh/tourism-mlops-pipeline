# Wellness Tourism Package – MLOps Pipeline

End-to-end MLOps project for **"Visit with Us"**: predict whether a customer will purchase the
newly introduced **Wellness Tourism Package**, so the sales team can prioritise whom to contact.

Everything runs on **GitHub Actions** (automated ML pipeline) and **Streamlit Community Cloud** (live app).

## Repository structure

```
.
├── .github/workflows/pipeline.yml          # CI/CD pipeline (GitHub Actions)
└── tourism_project/
    ├── data/tourism.csv                    # raw dataset (registered + validated by the pipeline)
    ├── model_building/
    │   ├── data_register.py                # 1. validate expected columns, print dataset summary
    │   ├── prep.py                         # 2. clean data, train/test split -> Xtrain/Xtest/ytrain/ytest.csv
    │   └── train.py                        # 3. tune XGBoost, track with MLflow, save best model
    ├── deployment/
    │   ├── app.py                          # Streamlit app
    │   ├── requirements.txt                # app dependencies
    │   ├── best_tourism_model_v1.joblib    # trained model (committed by the pipeline)
    │   └── model_metrics.json              # metrics of that model (committed by the pipeline)
    └── requirements.txt                    # pipeline dependencies
```

## Pipeline (`.github/workflows/pipeline.yml`)

Triggered on every push to `main` (or manually via *Run workflow*):

| Job | What it does | Output |
|---|---|---|
| `register-dataset` | validates `tourism.csv` (`data_register.py`) | `registered-data` artifact |
| `data-prep` | cleans the data and creates the train/test splits (`prep.py`) | `data-splits` artifact |
| `model-training` | grid-search tuning of an XGBoost pipeline with MLflow tracking (`train.py`), then **commits the best model back to `main`** | `mlflow-experiments` artifact + model in `tourism_project/deployment/` |

The model commit message contains `[skip ci]`, so it does not trigger another pipeline run.
Streamlit Community Cloud redeploys automatically when the new model lands on `main`.

## Data preparation decisions

* dropped the exported row-index column and `CustomerID` (no predictive value)
* fixed the `Fe Male` → `Female` typo in `Gender`
* removed exact duplicate customer profiles (avoids the same profile appearing in both train and test)
* stratified 80/20 split on `ProdTaken` (about 19 % buyers)

## Model

XGBoost classifier inside a scikit-learn pipeline (imputation, scaling, one-hot encoding). Tuned with
5-fold stratified `GridSearchCV` (F1); `scale_pos_weight` handles the class imbalance. Every tried parameter
combination, the best parameters and the train/test metrics are logged to MLflow.

## Run locally

```bash
pip install -r tourism_project/requirements.txt
python tourism_project/model_building/data_register.py
python tourism_project/model_building/prep.py
python tourism_project/model_building/train.py      # writes the model to tourism_project/deployment/

pip install -r tourism_project/deployment/requirements.txt
streamlit run tourism_project/deployment/app.py
```
