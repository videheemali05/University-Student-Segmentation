
# University Student Segmentation Using K-Means

## Project Overview
This project uses K-Means clustering to group university students based on similarities in their attendance and CGPA.

## Objectives
- Apply an unsupervised machine learning algorithm.
- Identify groups with similar attendance and CGPA patterns.
- Analyze cluster sizes and average values.
- Visualize the resulting clusters.

## Dataset
The project uses a university student dataset and selects a sample of 500 records.

Features used:
- Attendance
- CGPA

The dataset is not included in this repository because student information may be private.

## Methodology
1. Load and inspect the dataset.
2. Select 500 records using a fixed random seed.
3. Standardize Attendance and CGPA.
4. Apply K-Means clustering with two clusters.
5. Summarize the results and generate a scatter plot.

## Technologies Used
- Python
- Pandas
- NumPy
- Matplotlib
- Scikit-learn

## Installation
Install the required libraries:

```bash
python -m pip install -r requirements.txt
```

## Dataset Setup
Place an authorized CSV dataset named `student_dataset.csv` in the same folder as the Python script. It must contain the `Attendance` and `CGPA` columns and enough valid records.

## How to Run
Execute:

```bash
python kmeans_student_segmentation.py
```

## Output
- `kmeans_student_clusters.png` — scatter plot of the clusters.
- `University_Analysis_KMeans_500_Results.csv` — cluster results.

## Interpretation
The clusters show similarities in the selected features. They should not be treated as definitive measures of student ability.

## Academic Project
PBL — Python for Data Science
