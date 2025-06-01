# DS_TermProject_DecisionTree
from typing import Tuple
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from matplotlib import pyplot as plt
from sklearn.tree import plot_tree
from sklearn.tree import export_text
from sklearn.tree import _tree

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
    csv_path: str = r"C:\Users\ans52\OneDrive\바탕 화면\가천\3-1\데이터과학\term project\healthcare-dataset-stroke-data.csv",
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

# Supervised learning
# Predict stroke using age, avg_glucose_level, bmi, hypertension, heart_disease, ...
def decision_tree_classification():

    # Preprocess data
    Xtr, Xte, ytr, yte, pp = preprocess_stroke_data()

    # Transformed feature names from preprocessor pipeline
    feature_names = pp.named_steps["preprocessor"].get_feature_names_out()

    # Decision tree with entropy as criterion
    model = DecisionTreeClassifier(criterion = "entropy", # Criteria for division
                                   max_depth = 3, # Maximum Depth of Tree
                                   random_state = 42) # Reproduce Result
    model.fit(Xtr, ytr) # Model learning
    y_predict = model.predict(Xte) # Test data prediction

    # Readable dictionary of decoded transformed feature names
    decodedDict = {
        'num__age': 'age',
        'num__avg_glucose_level': 'avg_glucose_level',
        'num__bmi': 'bmi',
        'cat__work_type_Govt_job': 'work_type: Govt_job',
        'cat__work_type_Never_worked': 'work_type: Never_worked',
        'cat__work_type_Private': 'work_type: Private',
        'cat__work_type_Self-employed': 'work_type: Self-employed',
        'cat__work_type_children': 'work_type: children',
        'cat__smoking_status_Unknown': 'smoking_status: Unknown',
        'cat__smoking_status_formerly smoked': 'smoking_status: formerly smoked',
        'cat__smoking_status_never smoked': 'smoking_status: never smoked',
        'cat__smoking_status_smokes': 'smoking_status: smokes',
        'cat__age_group_Child': 'age_group: Child',
        'cat__age_group_Adult': 'age_group: Adult',
        'cat__age_group_Senior': 'age_group: Senior',
        'remainder__gender': 'gender', # 0 = Male, 1 = Female, 2 = Other
        'remainder__hypertension': 'hypertension', # 0 = No, 1 = Yes
        'remainder__heart_disease': 'heart_disease', # 0 = No, 1 = Yes
        'remainder__ever_married': 'ever_married', # 0 = No, 1 = Yes
        'remainder__Residence_type': 'Residence_type' # 0 = Rural, 1 = Urban
    }

    # Decoded feature names for readability
    decoded_names = []

    for i in feature_names:
        if i in decodedDict:
            decoded_names.append(decodedDict[i])
        else:
            decoded_names.append(i)

    # Restore normalized numerical feature for reverse normalization
    scaler = pp.named_steps["preprocessor"].named_transformers_["num"]
    num_features = ["age", "avg_glucose_level", "bmi"]
    
    tree = model.tree_ # Learned decision tree object

    # Convert scaled values back to original values
    def inverse_scale(feature, value):
        if feature.startswith("num__"):
            real_feature = feature.split("__")[1]
            idx = num_features.index(real_feature)
            return value * scaler.scale_[idx] + scaler.mean_[idx] # Reverse normalization
        else:
            return value # Just use if not numerical feature

    # Recursive function to print decision tree
    def tree_to_text(node, depth):
        indent = "  " * depth # Indent by current depth
        if tree.feature[node] != _tree.TREE_UNDEFINED: # Check current node is split node
            name = decoded_names[tree.feature[node]] # Feature name for split
            threshold = tree.threshold[node] # Scaled value for split

            # Convert scaled values back to original values
            threshold_value = inverse_scale(feature_names[tree.feature[node]], threshold)

            # Left child node when less than scaled value
            print(f"{indent} {name} <= {threshold_value}:  # normalized: {threshold}")

            # Recursive call to left child node (depth = depth + 1)
            tree_to_text(tree.children_left[node], depth + 1)
            
            # Right child node when more than scaled value
            print(f"{indent} {name} > {threshold_value}")

            # Recursive call to right child node (depth = depth + 1)
            tree_to_text(tree.children_right[node], depth + 1)

        else: # Current node is leaf node
            class_id = tree.value[node].argmax() # Most class at current node
            print(f"{indent} class {class_id}")

    # Split model based on feature in tree structure
    tree_to_text(0, 0) # Start at (node = 0, depth = 0)
    print()

    # Export tree in text format
    # Split model based on feature in tree structure
    normalized_tree = export_text(model, feature_names = decoded_names)
    print(normalized_tree)

    print(f"Depth: {model.get_depth()}")
    print(f"Accuracy: {model.score(Xte, yte)}")

    # Visualization Decision Tree
    # Split model based on feature in tree structure
    plt.figure(figsize  = (20, 10))
    plot_tree(model, 
              feature_names = decoded_names,
              class_names = ["0", "1"], # 0: stroke X, 1: stroke O
              filled = True)
    plt.title("Decision Tree Classification (stroke prediction)")

    # Feature importance that which feature strongly influences stroke
    importances = model.feature_importances_
    plt.figure(figsize = (10, 6))
    plt.barh(decoded_names, importances)
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.title("Feature Importance for Stroke Prediction")
    plt.show()

    return model

if __name__ == "__main__":
    Xtr, Xte, ytr, yte, pp = preprocess_stroke_data()
    decision_tree_classification()