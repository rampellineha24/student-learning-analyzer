import pandas as pd
import os

FILE = "data/processed/student_features.csv"
RAW = "data/raw"

print("=" * 55)
print("PREPARED DATASET AUDIT")
print("=" * 55)

# STEP 1: Load prepared dataset

df = pd.read_csv(FILE)

student_key = [
    "code_module",
    "code_presentation",
    "id_student"
]

print("\n1. DATASET INFORMATION")

print("Rows:", len(df))
print("Columns:", len(df.columns))

print("\nDuplicate student-course records:")

print(df.duplicated(subset=student_key).sum())


# STEP 2: Check missing values

print("\n2. MISSING VALUES")

print(
    df.isnull().sum()[
        df.isnull().sum() > 0
    ]
)


# STEP 3: Check numerical features

print("\n3. NUMERICAL FEATURE SUMMARY")

features = [
    "total_clicks",
    "active_days",
    "average_score",
    "submitted_assessments",
    "late_submissions",
    "missing_scores"
]

print(df[features].describe())


# STEP 4: Check LMS activity

print("\n4. LMS ACTIVITY CHECK")

print(
    "Students with zero clicks:",
    (df["total_clicks"] == 0).sum()
)

print(
    "Students with zero active days:",
    (df["active_days"] == 0).sum()
)


# STEP 5: Check assessment records

print("\n5. ASSESSMENT RECORD CHECK")

print(
    "Students without assessment records:",
    df["submitted_assessments"].isna().sum()
    + (df["submitted_assessments"] == 0).sum()
)

print(
    "Students with missing scores:",
    (df["missing_scores"] > 0).sum()
)


# STEP 6: Check missing assessment deadlines

print("\n6. ASSESSMENT DEADLINE CHECK")

assessments = pd.read_csv(
    os.path.join(RAW, "assessments.csv"),
    na_values=["?"]
)

student_assessments = pd.read_csv(
    os.path.join(RAW, "studentAssessment.csv"),
    na_values=["?"]
)

merged = student_assessments.merge(
    assessments,
    on="id_assessment",
    how="left",
    validate="many_to_one"
)

print(
    "Total student assessment records:",
    len(merged)
)

print(
    "Records with missing deadlines:",
    merged["date"].isna().sum()
)

print(
    "Records with available deadlines:",
    merged["date"].notna().sum()
)

print(
    "Records with missing scores:",
    merged["score"].isna().sum()
)


# STEP 7: Check late submission values

print("\n7. LATE SUBMISSION CHECK")

known_deadline = merged["date"].notna()

late = (
    known_deadline
    & (merged["date_submitted"] > merged["date"])
)

print("Late submissions:", late.sum())

print("Records with unknown deadlines:",
      (~known_deadline).sum())


# STEP 8: Final check

print("\n8. FINAL RESULT DISTRIBUTION")

print(df["final_result"].value_counts())

print("\nDATASET AUDIT COMPLETED!")