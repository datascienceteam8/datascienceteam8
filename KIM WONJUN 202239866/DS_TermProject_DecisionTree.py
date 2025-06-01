from typing import Tuple
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.tree import DecisionTreeClassifier
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

CSV_PATH = "C:/Users/admin/Downloads/archive (3)/healthcare-dataset-stroke-data.csv"

def preprocess_stroke_data(csv_path: str = CSV_PATH) -> Tuple[np.ndarray, np.ndarray, Pipeline]:
    print("▶️ 데이터 전처리 시작...")
    df = pd.read_csv(csv_path)
    df = df.drop(columns=["id"])
    df["bmi"] = df["bmi"].fillna(df["bmi"].median())

    categorical_cols = ["gender", "ever_married", "work_type", "Residence_type", "smoking_status"]
    for col in categorical_cols:
        if df[col].dtype == 'object':
            df[col] = df[col].str.strip()

    lower, upper = df["bmi"].quantile([0.01, 0.99])
    df["bmi"] = df["bmi"].clip(lower, upper)

    df["age_group"] = pd.cut(df["age"], bins=[0, 18, 60, np.inf],
                             labels=["Child", "Adult", "Senior"], right=False).astype(str)

    bin_maps = {
        "ever_married": {"No": 0, "Yes": 1},
        "Residence_type": {"Rural": 0, "Urban": 1},
    }
    for col, mapping in bin_maps.items():
        df[col] = df[col].map(mapping).fillna(-1).astype(int)

    y = df.pop("stroke").values

    numeric_feats = ["age", "avg_glucose_level", "bmi"]
    categorical_feats = ["gender", "ever_married", "work_type", "Residence_type",
                         "smoking_status", "age_group", "hypertension", "heart_disease"]

    numeric_tf = StandardScaler()
    categorical_tf = OneHotEncoder(handle_unknown="ignore", sparse_output=False)

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_tf, numeric_feats),
            ("cat", categorical_tf, categorical_feats),
        ],
        remainder="passthrough"
    )

    preprocess_pipeline = Pipeline([("preprocessor", preprocessor)])
    X_processed = preprocess_pipeline.fit_transform(df)

    print("✅ 데이터 전처리 완료! 데이터 형태:", X_processed.shape)
    return X_processed, y, preprocess_pipeline

def evaluate_decision_tree_cv(X, y):
    print("\n▶️ Decision Tree 모델 정의...")
    dt_model = DecisionTreeClassifier(criterion="entropy", max_depth=3, random_state=42)
    print("✅ Decision Tree 모델 정의 완료.")

    rskf = RepeatedStratifiedKFold(n_splits=10, n_repeats=10, random_state=42)

    accs, precisions, recalls, f1s, aucs = [], [], [], [], []

    print("▶️ 10x10 반복된 Stratified K-Fold 평가 시작...")
    for fold, (train_idx, val_idx) in enumerate(rskf.split(X, y), 1):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        dt_model.fit(X_train, y_train)
        y_pred = dt_model.predict(X_val)
        y_proba = dt_model.predict_proba(X_val)[:, 1]

        accs.append(accuracy_score(y_val, y_pred))
        precisions.append(precision_score(y_val, y_pred))
        recalls.append(recall_score(y_val, y_pred))
        f1s.append(f1_score(y_val, y_pred))
        aucs.append(roc_auc_score(y_val, y_proba))

        print(f"✅ Fold {fold:02d} - ACC: {accs[-1]:.4f}, Precision: {precisions[-1]:.4f}, Recall: {recalls[-1]:.4f}, F1: {f1s[-1]:.4f}, AUC: {aucs[-1]:.4f}")

    print("\n📊 🔟x🔟 평균 성능:")
    print(f"✅ Accuracy : {np.mean(accs):.4f}")
    print(f"✅ Precision: {np.mean(precisions):.4f}")
    print(f"✅ Recall   : {np.mean(recalls):.4f}")
    print(f"✅ F1 Score : {np.mean(f1s):.4f}")
    print(f"✅ ROC AUC  : {np.mean(aucs):.4f}")

    return {
        "accuracy": np.mean(accs),
        "precision": np.mean(precisions),
        "recall": np.mean(recalls),
        "f1": np.mean(f1s),
        "roc_auc": np.mean(aucs)
    }


if __name__ == "__main__":
    X, y, preprocessor_pipeline = preprocess_stroke_data()

    print("\n▶️ SMOTE 오버샘플링 적용 중...")
    smote = SMOTE(random_state=42)
    X_res, y_res = smote.fit_resample(X, y)

    print(f"Before SMOTE: 0 = {np.sum(y == 0)}, 1 = {np.sum(y == 1)}")
    print(f"After SMOTE: 0 = {np.sum(y_res == 0)}, 1 = {np.sum(y_res == 1)}")

    dt_scores = evaluate_decision_tree_cv(X_res, y_res)
