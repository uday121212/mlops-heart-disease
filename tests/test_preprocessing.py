import pandas as pd
from sklearn.linear_model import LogisticRegression

from src.preprocessing import (
    CATEGORICAL_FEATURES,
    FEATURE_COLUMNS,
    NUMERIC_FEATURES,
    build_pipeline,
    build_preprocessor,
)


SAMPLE_ROW = {
    "age": 63, "sex": 1, "cp": 3, "trestbps": 145, "chol": 233, "fbs": 1,
    "restecg": 0, "thalach": 150, "exang": 0, "oldpeak": 2.3, "slope": 0,
    "ca": 0, "thal": 1,
}


def test_feature_lists_disjoint_and_complete():
    assert set(NUMERIC_FEATURES).isdisjoint(CATEGORICAL_FEATURES)
    assert FEATURE_COLUMNS == NUMERIC_FEATURES + CATEGORICAL_FEATURES
    assert set(SAMPLE_ROW.keys()) == set(FEATURE_COLUMNS)


def test_preprocessor_transforms_dataframe():
    df = pd.DataFrame([SAMPLE_ROW, SAMPLE_ROW])
    pre = build_preprocessor()
    arr = pre.fit_transform(df)
    assert arr.shape[0] == 2
    assert arr.shape[1] >= len(NUMERIC_FEATURES)


def test_pipeline_fits_and_predicts():
    df = pd.DataFrame([SAMPLE_ROW] * 20)
    y = [0, 1] * 10
    pipe = build_pipeline(LogisticRegression(max_iter=200))
    pipe.fit(df, y)
    preds = pipe.predict(df)
    assert len(preds) == 20
    assert set(preds).issubset({0, 1})
