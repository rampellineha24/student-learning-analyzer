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

    df = pd.read_csv(path)

    print("\n" + "=" * 60)
    print("FILE:", file)
    print("ROWS AND COLUMNS:", df.shape)

    print("\nCOLUMN NAMES:")
    print(df.columns.tolist())

    print("\nFIRST 5 ROWS:")
    print(df.head())

    print("\nMISSING VALUES:")
    print(df.isnull().sum())