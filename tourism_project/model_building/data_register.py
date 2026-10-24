"""
Step 1 - Data registration.

Reads tourism.csv from the repository's data folder, validates that the
expected columns are present (and sane), and prints a short summary.
The script exits with a non-zero status when validation fails, which makes
the GitHub Actions job (and therefore the whole pipeline) fail early.
"""
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT / "tourism_project" / "data" / "tourism.csv"

EXPECTED_COLUMNS = [
    "CustomerID", "ProdTaken", "Age", "TypeofContact", "CityTier",
    "DurationOfPitch", "Occupation", "Gender", "NumberOfPersonVisiting",
    "NumberOfFollowups", "ProductPitched", "PreferredPropertyStar",
    "MaritalStatus", "NumberOfTrips", "Passport", "PitchSatisfactionScore",
    "OwnCar", "NumberOfChildrenVisiting", "Designation", "MonthlyIncome",
]
TARGET = "ProdTaken"


def validate(df: pd.DataFrame) -> list:
    """Return a list of validation errors (empty list == dataset is valid)."""
    errors = []

    if df.empty:
        errors.append("Dataset is empty.")

    missing_cols = [c for c in EXPECTED_COLUMNS if c not in df.columns]
    if missing_cols:
        errors.append(f"Missing expected columns: {missing_cols}")

    if TARGET in df.columns:
        bad_target = set(df[TARGET].dropna().unique()) - {0, 1}
        if bad_target:
            errors.append(f"{TARGET} must be 0/1 but found values: {sorted(bad_target)}")
        if df[TARGET].isna().any():
            errors.append(f"{TARGET} contains missing values.")

    if "CustomerID" in df.columns and df["CustomerID"].duplicated().any():
        errors.append("CustomerID is not unique.")

    return errors


def main() -> None:
    if not DATA_PATH.exists():
        sys.exit(f"[ERROR] Dataset not found at: {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)

    print("=" * 60)
    print("DATASET REGISTRATION SUMMARY")
    print("=" * 60)
    print(f"Location          : {DATA_PATH.relative_to(ROOT)}")
    print(f"Rows x Columns    : {df.shape[0]} x {df.shape[1]}")

    extra_cols = [c for c in df.columns if c not in EXPECTED_COLUMNS]
    print(f"Extra columns     : {extra_cols if extra_cols else 'none'}")

    print("\nColumn types:")
    print(df.dtypes.to_string())

    print("\nMissing values per column (only columns with gaps):")
    nulls = df.isnull().sum()
    print(nulls[nulls > 0].to_string() if nulls.any() else "none")

    if TARGET in df.columns:
        print(f"\nTarget distribution ({TARGET}):")
        print(df[TARGET].value_counts(normalize=True).round(3).to_string())

    errors = validate(df)
    print("\nValidation:", "PASSED" if not errors else "FAILED")
    if errors:
        for err in errors:
            print(f"  - {err}")
        sys.exit(1)

    print("Dataset registered successfully.")


if __name__ == "__main__":
    main()
