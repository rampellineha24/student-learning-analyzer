
import pandas as pd
import os

RAW = "data/raw"
OUTPUT = "data/processed"

os.makedirs(OUTPUT, exist_ok=True)

student_key = [
    "code_module",
    "code_presentation",
    "id_student"
]

print("STEP 1: Loading student information...")

students = pd.read_csv(
    os.path.join(RAW, "studentInfo.csv"),
    na_values=["?"]
)

print("Students loaded:", len(students))


print("\nSTEP 2: Processing LMS activity...")

daily_activity = []

for chunk in pd.read_csv(
    os.path.join(RAW, "studentVle.csv"),
    chunksize=500000
):
    daily = (
        chunk.groupby(student_key + ["date"])
        .agg(daily_clicks=("sum_click", "sum"))
        .reset_index()
    )

    daily_activity.append(daily)

daily_data = pd.concat(
    daily_activity,
    ignore_index=True
)

vle_features = (
    daily_data.groupby(student_key)
    .agg(
        total_clicks=("daily_clicks", "sum"),
        active_days=("date", "nunique")
    )
    .reset_index()
)

print("LMS activity processed.")


print("\nSTEP 3: Processing assessment information...")

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

assessment_data["late_submission"] = (
    assessment_data["date"].notna()
    & (
        assessment_data["date_submitted"]
        > assessment_data["date"]
    )
)

assessment_features = (
    assessment_data.groupby(student_key)
    .agg(
        average_score=("score", "mean"),
        submitted_assessments=("id_assessment", "count"),
        late_submissions=("late_submission", "sum"),
        missing_scores=("score", lambda x: x.isna().sum())
    )
    .reset_index()
)

print("Assessment information processed.")


print("\nSTEP 4: Combining student information...")

final_data = students.merge(
    vle_features,
    on=student_key,
    how="left",
    validate="one_to_one"
)

final_data = final_data.merge(
    assessment_features,
    on=student_key,
    how="left",
    validate="one_to_one"
)

# Students without recorded LMS activity
# receive zero activity counts.
activity_columns = [
    "total_clicks",
    "active_days"
]

final_data[activity_columns] = (
    final_data[activity_columns].fillna(0)
)

# Students without assessment records
# receive zero submission counts.
count_columns = [
    "submitted_assessments",
    "late_submissions",
    "missing_scores"
]

final_data[count_columns] = (
    final_data[count_columns].fillna(0)
)

output_file = os.path.join(
    OUTPUT,
    "student_features.csv"
)

final_data.to_csv(output_file, index=False)


print("\nSTEP 5: Dataset preparation completed!")

print("Final dataset shape:", final_data.shape)

print("\nSelected columns:")
print(final_data.columns.tolist())

print("\nFirst five records:")
print(final_data.head())

print("\nFinal result distribution:")
print(final_data["final_result"].value_counts())

print("\nSaved successfully to:", output_file)
