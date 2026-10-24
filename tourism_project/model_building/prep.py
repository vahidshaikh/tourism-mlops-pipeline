"""
Step 2 - Data preparation.

* loads tourism.csv directly from the repository data folder
* removes unnecessary columns and cleans the data
* splits it into train / test sets and saves them as CSV files in the
  repository root (Xtrain.csv, Xtest.csv, ytrain.csv, ytest.csv)

The GitHub Actions workflow uploads those four files as an artifact so the
next job (model training) can download them.
"""
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT / "tourism_project" / "data" / "tourism.csv"
OUT_DIR = ROOT  # the workflow expects the split files in the repo root

TARGET = "ProdTaken"
TEST_SIZE = 0.20
RANDOM_STATE = 42


def clean(df: pd.DataFrame) -> pd.DataFrame:
    n_start = len(df)

    # 1. Drop unnecessary columns: the unnamed row-index column that was
    #    exported with the CSV and the CustomerID identifier (no predictive value).
    drop_cols = [c for c in df.columns if c.startswith("Unnamed")] + ["CustomerID"]
    df = df.drop(columns=drop_cols, errors="ignore")
    print(f"Dropped columns            : {drop_cols}")

    # 2. Trim stray whitespace in text columns.
    for col in df.select_dtypes(include=["object", "string"]).columns:
        df[col] = df[col].astype(str).str.strip()

    # 3. Fix inconsistent category labels ('Fe Male' is a typo for 'Female').
    df["Gender"] = df["Gender"].replace({"Fe Male": "Female"})

    # 4. Rows without a target value cannot be used for supervised learning.
    df = df.dropna(subset=[TARGET])

    # 5. Remove exact duplicate records (after the ID is removed) so the same
    #    customer profile cannot end up in both the train and the test set.
    n_dupes = int(df.duplicated().sum())
    df = df.drop_duplicates().reset_index(drop=True)
    print(f"Duplicate rows removed     : {n_dupes}")
    print(f"Rows before -> after       : {n_start} -> {len(df)}")

    # Remaining missing values (if any) are imputed inside the model pipeline
    # (train.py) so that exactly the same logic is applied at prediction time.
    remaining = int(df.isnull().sum().sum())
    print(f"Missing values left        : {remaining} (handled by the model pipeline)")
    return df


def main() -> None:
    df = pd.read_csv(DATA_PATH)
    print(f"Loaded {DATA_PATH.relative_to(ROOT)} with shape {df.shape}")

    df = clean(df)

    X = df.drop(columns=[TARGET])
    y = df[TARGET].astype(int)

    Xtrain, Xtest, ytrain, ytest = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    Xtrain.to_csv(OUT_DIR / "Xtrain.csv", index=False)
    Xtest.to_csv(OUT_DIR / "Xtest.csv", index=False)
    ytrain.to_csv(OUT_DIR / "ytrain.csv", index=False)
    ytest.to_csv(OUT_DIR / "ytest.csv", index=False)

    print(f"\nXtrain: {Xtrain.shape} | Xtest: {Xtest.shape}")
    print(f"Target rate  train: {ytrain.mean():.3f} | test: {ytest.mean():.3f}")
    print("Saved Xtrain.csv, Xtest.csv, ytrain.csv, ytest.csv to", OUT_DIR)


if __name__ == "__main__":
    main()
