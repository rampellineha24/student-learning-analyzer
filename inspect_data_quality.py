import pandas as pd
import os

data_folder = "data/raw"

files = [
    "assessments.csv",
    "courses.csv",
    "studentAssessment.csv",
    "studentInfo.csv",
    "studentRegistration.csv",
    "studentVle.csv",
    "vle.csv"
]

for file in files:
    path = os.path.join(data_folder, file)

    print("\n" + "=" * 60)
    print("FILE:", file)

    total_rows = 0
    missing_counts = None

    for chunk in pd.read_csv(path, na_values=["?"], chunksize=500000):
        total_rows += len(chunk)
        counts = chunk.isnull().sum()

        if missing_counts is None:
            missing_counts = counts
        else:
            missing_counts += counts

    print("TOTAL ROWS:", total_rows)
    print("MISSING VALUES:")
    print(missing_counts[missing_counts > 0])