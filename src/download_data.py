"""Download the Heart Disease UCI dataset (Cleveland) and save as CSV.

Source: https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/
"""
from __future__ import annotations

import io
import sys
from pathlib import Path

import pandas as pd
import requests

URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/"
    "heart-disease/processed.cleveland.data"
)

COLUMNS = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
    "thalach", "exang", "oldpeak", "slope", "ca", "thal", "num",
]

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
RAW_PATH = DATA_DIR / "heart_disease_raw.csv"
CLEAN_PATH = DATA_DIR / "heart_disease_clean.csv"


def download() -> pd.DataFrame:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Downloading dataset from {URL}")
    r = requests.get(URL, timeout=30)
    r.raise_for_status()
    df = pd.read_csv(io.StringIO(r.text), header=None, names=COLUMNS, na_values="?")
    df.to_csv(RAW_PATH, index=False)
    print(f"Saved raw dataset -> {RAW_PATH} (shape={df.shape})")
    return df


def clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # Impute missing with median (only ca, thal typically have NA)
    for col in df.columns:
        if df[col].isna().any():
            df[col] = df[col].fillna(df[col].median())
    # Binarize target: 0 = no disease, 1 = disease
    df["target"] = (df["num"] > 0).astype(int)
    df = df.drop(columns=["num"])
    df.to_csv(CLEAN_PATH, index=False)
    print(f"Saved cleaned dataset -> {CLEAN_PATH} (shape={df.shape})")
    return df


def main() -> int:
    try:
        df = download()
    except Exception as e:  # network failure fallback
        print(f"Download failed: {e}", file=sys.stderr)
        if RAW_PATH.exists():
            print("Using existing raw file.")
            df = pd.read_csv(RAW_PATH)
        else:
            return 1
    clean(df)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
