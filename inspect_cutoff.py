import pandas as pd
import os
import numpy as np

RAW = "data/raw"

student_key = [
    "code_module",
    "code_presentation",
    "id_student"
]

print("=" * 55)
print("CORRECTED EARLY-WARNING CUTOFF VALIDATION")
print("=" * 55)

# STEP 1: Course-specific cutoffs

courses = pd.read_csv(
    os.path.join(RAW, "courses.csv")
)

courses["cutoff_day"] = np.floor(
    courses["module_presentation_length"] * 0.5
).astype(int)

print("\n1. COURSE CUTOFFS")
print("Course presentations:", len(courses))
print("Minimum cutoff:", courses["cutoff_day"].min())
print("Maximum cutoff:", courses["cutoff_day"].max())


# STEP 2: Process LMS activity

print("\n2. LMS ACTIVITY")

daily_parts = []

pre_course_records = 0
pre_course_clicks = 0

for chunk in pd.read_csv(
    os.path.join(RAW, "studentVle.csv"),
    chunksize=500000
):

    chunk = chunk.merge(
        courses[
            student_key[:2] + ["cutoff_day"]
        ],
        on=student_key[:2],
        how="left",
        validate="many_to_one"
    )

    pre_course = chunk[chunk["date"] < 0]

    pre_course_records += len(pre_course)
    pre_course_clicks += pre_course["sum_click"].sum()

    early = chunk[
        (chunk["date"] >= 0)
        & (chunk["date"] <= chunk["cutoff_day"])
    ]

    daily = (
        early.groupby(student_key + ["date"])
        .agg(
            daily_clicks=("sum_click", "sum")
        )
        .reset_index()
    )

    daily_parts.append(daily)

daily_data = pd.concat(
    daily_parts,
    ignore_index=True
)

daily_data = (
    daily_data.groupby(student_key + ["date"])
    .agg(
        daily_clicks=("daily_clicks", "sum")
    )
    .reset_index()
)

early_features = (
    daily_data.groupby(student_key)
    .agg(
        early_clicks=("daily_clicks", "sum"),
        early_active_days=("date", "nunique")
    )
    .reset_index()
)

print("Pre-course activity records:", pre_course_records)
print("Pre-course clicks:", pre_course_clicks)

print(
    "Student-course records with early activity:",
    len(early_features)
)

print(
    "Student-course records without early activity:",
    32593 - len(early_features)
)

print("\nEarly activity summary:")
print(
    early_features[
        ["early_clicks", "early_active_days"]
    ].describe()
)


# STEP 3: Early assessment submissions

print("\n3. EARLY ASSESSMENTS")

assessments = pd.read_csv(
    os.path.join(RAW, "assessments.csv"),
    na_values=["?"]
)

student_assessments = pd.read_csv(
    os.path.join(RAW, "studentAssessment.csv"),
    na_values=["?"]
)

assessment_data = student_assessments.merge(
    assessments,
    on="id_assessment",
    how="left",
    validate="many_to_one"
)

assessment_data = assessment_data.merge(
    courses[
        student_key[:2] + ["cutoff_day"]
    ],
    on=student_key[:2],
    how="left",
    validate="many_to_one"
)

early = assessment_data[
    (assessment_data["date_submitted"] >= 0)
    & (
        assessment_data["date_submitted"]
        <= assessment_data["cutoff_day"]
    )
].copy()

early["late_status"] = np.where(
    early["date"].isna(),
    "Unknown deadline",
    np.where(
        early["date_submitted"] > early["date"],
        "Late",
        "On time"
    )
)

assessment_features = (
    early.groupby(student_key)
    .agg(
        early_submissions=("id_assessment", "count"),
        early_valid_scores=("score", "count"),
        early_missing_scores=("score", lambda x: x.isna().sum())
    )
    .reset_index()
)

print("Early submission records:", len(early))

print(
    "Student-course records with early submissions:",
    len(assessment_features)
)

print("\nEarly submission status:")
print(early["late_status"].value_counts())

print("\nEarly assessment summary:")
print(assessment_features.describe())


# STEP 4: Verify student-course coverage

print("\n4. COVERAGE CHECK")

students = pd.read_csv(
    os.path.join(RAW, "studentInfo.csv"),
    na_values=["?"]
)

print("Total student-course records:", len(students))

print(
    "Unique student-course combinations:",
    students[student_key].drop_duplicates().shape[0]
)

print(
    "Unique students:",
    students["id_student"].nunique()
)

print("\nVALIDATION COMPLETED!")