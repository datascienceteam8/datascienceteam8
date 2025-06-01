# DS_TermProject_KMeans
from typing import Tuple
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from kneed import KneeLocator # pip install kneed
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns

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

# Find best number of cluster
def elbow_method():

    # Preprocess data
    Xtr, Xte, ytr, yte, pp = preprocess_stroke_data()

    # Data point in the cluster from centroid of the cluster
    wcss = [] # within-cluster sum of squares

    # Change number of clusters by number of features (except id, stroke)
    for i in range(1, 11):
        model = KMeans(n_clusters = i, random_state = 42) # KMeans
        model.fit(Xtr) # Model learning
        wcss.append(model.inertia_) # inertia_ = WCSS
    
    # Auto finding number of clusters at convex
    # Decreasing WCSS when increasing number of clusters
    clusterNum = KneeLocator(range(1, 11), wcss, curve = "convex", direction = "decreasing")
    print(f"number of clusters (Elbow point): {clusterNum.elbow}")
    
    # Visualization elbow method
    plt.figure(figsize = (8, 5))
    plt.plot(range(1, 11), wcss)
    plt.title("Elbow method")
    plt.xlabel("Number of clusters")
    plt.ylabel("WCSS")
    plt.xticks(range(1, 11))
    plt.grid(True)
    plt.show()

    return clusterNum

# Unsupervised learning
# Grouping based on similar health information without using stroke feature
# Analysis of high-risk cluster for stroke feature
def kmeans_clustering():

    # Preprocess data
    Xtr, Xte, ytr, yte, pp = preprocess_stroke_data()

    # Auto find best cluster number
    clusterNum = elbow_method()

    # Transformed feature names from preprocessor pipeline
    feature_names = pp.named_steps["preprocessor"].get_feature_names_out()

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
    
    # Readable value of integer
    label_decoding_map = {
        'gender': {0: 'Male', 1: 'Female', 2: 'Other'},
        'ever_married': {0: 'No', 1: 'Yes'},
        'Residence_type': {0: 'Rural', 1: 'Urban'},
        'hypertension': {0: 'No', 1: 'Yes'},
        'heart_disease': {0: 'No', 1: 'Yes'},
    }

    # Decoded feature names for readability
    decoded_names = []

    for i in feature_names:
        if i in decodedDict:
            decoded_names.append(decodedDict[i])
        else:
            decoded_names.append(i)
    
    # Restore normalized numerical feature
    scaler = pp.named_steps["preprocessor"].named_transformers_["num"]
    num_features = ["age", "avg_glucose_level", "bmi"]

    # Convert scaled values back to original values
    denormalized_num_features = []

    for j in num_features:
        name = "num__" + j
        denormalized_num_features.append(name)

    # Optimizing cluster by iterative centroid recalculation
    for k in range(100, 301, 100):
        model = KMeans(n_clusters = clusterNum.elbow, max_iter = k) # KMeans
        model.fit(Xtr) # Model learning
        labels = model.labels_ # Data point in which cluster

        # DataFrame of clustering result
        df_clustering = pd.DataFrame(Xtr, columns = feature_names)

        # Convert scaled values back to original values
        df_clustering[denormalized_num_features] = scaler.inverse_transform(df_clustering[denormalized_num_features])
        
        # Decoded feature names for readability
        df_clustering.columns = decoded_names

        # Cluster number array
        df_clustering["cluster"] = labels

        # Stroke or not by training data
        df_clustering["stroke"] = ytr

        print(f"\n Iteration = {k}")
        print("\n < Number of data per cluster >\n", df_clustering["cluster"].value_counts())

        # Mean of numerical feature by cluster
        print("\n< Mean value of numerical feature per cluster >\n", df_clustering.groupby("cluster")[["age", "avg_glucose_level", "bmi"]].mean())
        
        print("\n< Mode value of categorical feature per cluster >")
        
        binary_col = ["gender", "ever_married", "Residence_type", "hypertension", "heart_disease"]
        for col in binary_col:
            if col in df_clustering.columns:
                mode_per_cluster = []

                # Repeat label(cluster number) in df_clustering
                for cluster in df_clustering["cluster"].unique():

                    # Data in cluster of current label
                    cluster_data = df_clustering[df_clustering["cluster"] == cluster]

                    # Mode of feature in the cluster
                    mode = cluster_data[col].mode()

                    if not mode.empty: # Mode O
                        mode_per_cluster.append(mode.iloc[0]) # First mode
                    else: # Mode X
                        mode_per_cluster.append("N/A") # Can not compute
                
            if col in label_decoding_map:
                # Integer to readable value
                mode_per_cluster = pd.Series(mode_per_cluster).map(label_decoding_map[col])
            print(f"{col}:") # Current feature
            print(mode_per_cluster)
            print()
          
        onehot_att = {
            "work_type": [], "smoking_status": [], "age_group": []
        }

        # Check feature name in df_clustering
        # Save feature in each list
        for att in df_clustering.columns:
            if att.startswith("work_type:"):
                onehot_att["work_type"].append(att)
            elif att.startswith("smoking_status:"):
                onehot_att["smoking_status"].append(att)
            elif att.startswith("age_group:"):
                onehot_att["age_group"].append(att)
        
        for group_name, cols in onehot_att.items():
            print(f"{group_name}:") # work_type, smoking_status, age_group
            mode_labels = (
                df_clustering[cols] # onehot_att
                .groupby(df_clustering["cluster"]).mean() # Mean of onehot feature by cluster
                .idxmax(axis=1) # Feature name in cluster with most mean
            )
            print(mode_labels)
            print()
        
        # Mean of stroke by cluster * 100
        print("< Stroke ratio per cluster (%) >\n", (df_clustering.groupby("cluster")["stroke"].mean() * 100))

        # PCA to 2D for clustering visualization
        pca = PCA(n_components = 2)
        principalComponents = pca.fit_transform(Xtr) # Model learning

        # PCA in DataFrame
        df_clustering["PC1"] = principalComponents[:, 0] # First PC
        df_clustering["PC2"] = principalComponents[:, 1] # Second PC

        # N-dimension centroid tok 2-d centroid
        centroids_2d = pca.transform(model.cluster_centers_)
        print("\n< Centroid coordinate >\n", centroids_2d)

        # Visualization KMeans
        plt.figure(figsize = (8, 6))
        color = ["red", "green", "blue", "yellow"]

        # hue = "cluster" : Color per cluster
        sns.scatterplot(x = "PC1", y = "PC2", hue = "cluster", data = df_clustering, palette = color, s = 50)
        
        # Cluster centroids
        plt.scatter(centroids_2d[:, 0], centroids_2d[:, 1], s = 200, c = "black")
        
        plt.title(f"KMeans Clustering (k = {clusterNum.elbow}, max_iter = {k})")
        plt.xlabel("Principal Component 1")
        plt.ylabel("Principal Component 2")
        plt.legend(title = "Cluster")
        plt.grid(True)
        plt.show()

    return model, df_clustering

if __name__ == "__main__":
    Xtr, Xte, ytr, yte, pp = preprocess_stroke_data()
    kmeans_clustering()