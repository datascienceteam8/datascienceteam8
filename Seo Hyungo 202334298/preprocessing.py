from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

"""
Clean, transform and split the Dataset.

Parameters:
csv_path : str, default "healthcare-dataset-stroke-data.csv"
    Location of the Dataset CSV file.
test_size : float, default 0.20
    Fraction reserved for testing (remainder used for training).
random_state : int, default 42
    Reproducibility seed for the train/test split.

Returns:
X_train, X_test : ndarray
    Model‑ready feature matrices.
y_train, y_test : ndarray
    Corresponding label vectors.
pipeline : sklearn.pipeline.Pipeline
    The fitted preprocessing pipeline (re‑use on new data).
"""
def preprocess_stroke_data(
    csv_path: str = "healthcare-dataset-stroke-data.csv",
    test_size: float = 0.20,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Pipeline]:
    df = pd.read_csv(csv_path)
    # Drop id cloumn(non‑informative)
    df = df.drop(columns=["id"])

    # Impute missing values for BMI
    df["bmi"] = df["bmi"].fillna(df["bmi"].median())

    # Strip categorical data
    categorical_cols = [
        "gender",
        "ever_married",
        "work_type",
        "Residence_type",
        "smoking_status",
    ]
    df[categorical_cols] = df[categorical_cols].apply(lambda s: s.str.strip())

    # Clip BMI outliers
    lower, upper = df["bmi"].quantile([0.01, 0.99])
    df["bmi"] = df["bmi"].clip(lower, upper)

    # Group age
    #  0 - 17: Child
    # 18 - 59: Adult
    # 60 -   : Senior
    df["age_group"] = pd.cut(
        df["age"],
        bins=[0, 18, 60, np.inf],
        labels=["Child", "Adult", "Senior"],
        right=False,
    )

    # Binary encoding (map to integers)
    bin_maps = {
        "gender": {"Male": 0, "Female": 1, "Other": 2},
        "ever_married": {"No": 0, "Yes": 1},
        "Residence_type": {"Rural": 0, "Urban": 1},
    }
    for col, mapping in bin_maps.items():
        df[col] = df[col].map(mapping).astype(int)

    # Define target & feature groups
    y = df.pop("stroke").values
    numeric_feats = ["age", "avg_glucose_level", "bmi"]
    categorical_feats = ["work_type", "smoking_status", "age_group"]

    # Define transformers
    numeric_tf = StandardScaler()
    categorical_tf = OneHotEncoder(handle_unknown="ignore", sparse_output=False)

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_tf, numeric_feats),
            ("cat", categorical_tf, categorical_feats),
        ],
        remainder="passthrough",  # keep the newly encoded binary ints
    )

    pipeline = Pipeline([("preprocessor", preprocessor)])

    # Stratified train/test split and fit
    X_train, X_test, y_train, y_test = train_test_split(
        df,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    pipeline.fit(X_train)

    # Transform
    X_train_proc = pipeline.transform(X_train)
    X_test_proc = pipeline.transform(X_test)

    return X_train_proc, X_test_proc, y_train, y_test, pipeline


if __name__ == "__main__":
    Xtr, Xte, ytr, yte, pp = preprocess_stroke_data()
    print(
        f"Training matrix shape: {Xtr.shape}\n"
        f"Test matrix shape:      {Xte.shape}\n"
    )
