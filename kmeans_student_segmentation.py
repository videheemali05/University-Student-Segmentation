"""
University Student Segmentation Using K-Means Clustering
PBL 2 - Programming for Data Science

Setup:
1. Put student_dataset.csv in the same folder as this script.
2. Install dependencies: pip install pandas numpy matplotlib scikit-learn
3. Run: python kmeans_student_segmentation.py
"""

from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

BASE_DIR = Path(__file__).resolve().parent
DATASET_PATH = BASE_DIR / "student_dataset.csv"
SAMPLE_SIZE = 500
RANDOM_STATE = 42
K = 2
FEATURES = ["Attendance", "CGPA"]


def load_dataset(path):
    """Load the CSV and validate required columns and row count."""
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {path}\n"
            "Place student_dataset.csv beside this Python file."
        )

    df = pd.read_csv(path)
    missing_columns = [col for col in FEATURES if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")
    if len(df) < SAMPLE_SIZE:
        raise ValueError(f"At least {SAMPLE_SIZE} rows are required.")
    return df


def inspect_dataset(df):
    """Display basic dataset information and data-quality checks."""
    print("\n" + "=" * 55)
    print("DATASET INFORMATION")
    print("=" * 55)
    print(f"Rows: {df.shape[0]}")
    print(f"Columns: {df.shape[1]}")
    print("Column names:", df.columns.tolist())
    print("\nFirst five rows:")
    print(df.head())
    print("\nMissing values:")
    print(df.isnull().sum())
    print("\nDuplicate rows:", df.duplicated().sum())


def prepare_sample(df):
    """Select 500 reproducible records with valid selected features."""
    data = df.copy()
    for col in FEATURES:
        data[col] = pd.to_numeric(data[col], errors="coerce")
    data = data.dropna(subset=FEATURES)

    if len(data) < SAMPLE_SIZE:
        raise ValueError(
            f"Only {len(data)} rows have valid Attendance and CGPA values; "
            f"{SAMPLE_SIZE} are required."
        )

    sample = data.sample(n=SAMPLE_SIZE, random_state=RANDOM_STATE).copy()
    print("\nSelected features:", FEATURES)
    print("Sample size:", len(sample))
    print("\nSample feature statistics:")
    print(sample[FEATURES].describe().round(2))
    return sample


def standardize_features(sample):
    """Standardize Attendance and CGPA before clustering."""
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(sample[FEATURES])
    print("\nFeatures standardized successfully.")
    return scaler, X_scaled


def run_kmeans(sample, X_scaled):
    """Fit K-Means and append cluster labels (1 and 2) to the sample."""
    model = KMeans(n_clusters=K, random_state=RANDOM_STATE, n_init=10)
    labels = model.fit_predict(X_scaled)
    sample["Cluster"] = labels + 1
    print("K-Means clustering completed.")
    return model, labels, sample


def summarize_results(sample, model, scaler):
    """Print cluster sizes, feature averages, and centroids."""
    print("\n" + "=" * 55)
    print("CLUSTER SIZES")
    print("=" * 55)
    counts = sample["Cluster"].value_counts().sort_index()
    for cluster, count in counts.items():
        print(f"Cluster {cluster}: {count} students")

    summary = sample.groupby("Cluster")[FEATURES].mean().round(2)
    print("\nAVERAGE ATTENDANCE AND CGPA BY CLUSTER")
    print(summary)

    # Convert centroids from standardized values to original units.
    centroid_values = scaler.inverse_transform(model.cluster_centers_)
    centroids = pd.DataFrame(centroid_values, columns=FEATURES)
    centroids.index = [f"Cluster {i}" for i in range(1, K + 1)]
    print("\nK-MEANS CENTROIDS (ORIGINAL UNITS)")
    print(centroids.round(2))
    return summary, centroids


def save_cluster_plot(sample, labels, centroids):
    """Save and display the cluster scatter plot."""
    plt.figure(figsize=(9, 6))
    plt.scatter(
        sample["Attendance"], sample["CGPA"],
        c=labels, alpha=0.65, cmap="viridis"
    )
    plt.scatter(
        centroids["Attendance"], centroids["CGPA"],
        marker="X", s=250, color="red", edgecolor="black",
        label="K-Means centroids"
    )
    plt.xlabel("Attendance")
    plt.ylabel("CGPA")
    plt.title("K-Means Clustering of Students (K = 2)")
    plt.legend()
    plt.grid(True, alpha=0.25)
    plt.tight_layout()

    plot_path = BASE_DIR / "kmeans_student_clusters.png"
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    print(f"Cluster plot saved to: {plot_path}")
    plt.show()
    plt.close()


def save_results(sample):
    """Save cluster assignments to a CSV file."""
    output_path = BASE_DIR / "University_Analysis_KMeans_500_Results.csv"
    sample.to_csv(output_path, index=False)
    print(f"Clustered results saved to: {output_path}")


def main():
    print("=" * 55)
    print("UNIVERSITY STUDENT SEGMENTATION USING K-MEANS")
    print("=" * 55)

    df = load_dataset(DATASET_PATH)
    inspect_dataset(df)
    sample = prepare_sample(df)
    scaler, X_scaled = standardize_features(sample)
    model, labels, sample = run_kmeans(sample, X_scaled)
    summary, centroids = summarize_results(sample, model, scaler)
    save_cluster_plot(sample, labels, centroids)
    save_results(sample)

    print("\nPROJECT COMPLETED SUCCESSFULLY")
    print(f"Students clustered: {len(sample)}")
    print(f"Features used: {', '.join(FEATURES)}")
    print(f"Number of clusters: {K}")
    print("\nFinal cluster summary:")
    print(summary)


if __name__ == "__main__":
    main()
