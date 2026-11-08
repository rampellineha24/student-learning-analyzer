
import pandas as pd
from pathlib import Path

# ============================================================
# CANDIDATE MACHINE LEARNING DATASET PREPARATION
# ============================================================

print("=" * 70)
print("STUDENT LEARNING SUPPORT SYSTEM")
print("CANDIDATE MODELLING DATASET PREPARATION")
print("=" * 70)

# 1. Load original dataset

input_path = Path(
    "data/processed/final_early_warning_dataset.csv"
)

df = pd.read_csv(input_path)

print("\n1. ORIGINAL DATASET")
print("-" * 70)
print("Shape:", df.shape)

# 2. Define target and grouping identifier

target_column = "final_result"
group_column = "id_student"

# 3. Select baseline model features

feature_columns = [
    "code_module",
    "code_presentation",
    "cutoff_day",
    "num_of_prev_attempts",
    "studied_credits",
    "early_total_clicks",
    "early_active_days",
    "early_on_time_submissions",
    "early_late_submissions"
]

# 4. Validate required columns

required_columns = (
    feature_columns + [group_column, target_column]
)

missing_columns = [
    column for column in required_columns
    if column not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Missing required columns: {missing_columns}"
    )

# 5. Create candidate modelling dataset

selected_columns = (
    [group_column] + feature_columns + [target_column]
)

model_df = df[selected_columns].copy()

# 6. Check missing values

print("\n2. MISSING VALUE CHECK")
print("-" * 70)

print(model_df.isnull().sum())

if model_df.isnull().sum().sum() > 0:
    raise ValueError(
        "Missing values detected. Investigate before proceeding."
    )

# 7. Check duplicate records

print("\n3. DUPLICATE RECORD CHECK")
print("-" * 70)

duplicates = model_df.duplicated(
    subset=[group_column, "code_module", "code_presentation"]
).sum()

print("Duplicate student-course records:", duplicates)

if duplicates > 0:
    raise ValueError("Duplicate student-course records detected.")

# 8. Check target classes

print("\n4. TARGET CLASS DISTRIBUTION")
print("-" * 70)

print(model_df[target_column].value_counts())

expected_classes = {
    "Pass",
    "Fail",
    "Distinction",
    "Withdrawn"
}

actual_classes = set(model_df[target_column].unique())

if actual_classes != expected_classes:
    raise ValueError(
        f"Unexpected target classes: {actual_classes}"
    )

# 9. Student grouping information

print("\n5. STUDENT GROUPING INFORMATION")
print("-" * 70)

student_counts = model_df.groupby(group_column).size()

print("Unique students:", student_counts.size)

print(
    "Students with multiple records:",
    (student_counts > 1).sum()
)

print(
    "Maximum records per student:",
    student_counts.max()
)

# 10. Display selected features

print("\n6. SELECTED MODEL FEATURES")
print("-" * 70)

for index, feature in enumerate(feature_columns, start=1):
    print(f"{index}. {feature}")

print("\nGrouping identifier:", group_column)
print("Prediction target:", target_column)

# 11. Save candidate dataset

output_path = Path(
    "data/processed/candidate_model_dataset.csv"
)

output_path.parent.mkdir(
    parents=True,
    exist_ok=True
)

model_df.to_csv(output_path, index=False)

# 12. Save selection report

report_path = Path(
    "reports/eda/11_feature_selection_plan.txt"
)

report = []

report.append("CANDIDATE FEATURE SELECTION REPORT")
report.append("=" * 70)

report.append(f"\nOriginal dataset shape: {df.shape}")
report.append(f"Candidate dataset shape: {model_df.shape}")

report.append("\nSELECTED MODEL FEATURES:")
report.extend(
    [f"{i}. {feature}" for i, feature
     in enumerate(feature_columns, start=1)]
)

report.append("\nGROUPING IDENTIFIER:")
report.append(group_column)

report.append("\nPREDICTION TARGET:")
report.append(target_column)

report.append("\nEXCLUDED FEATURES:")

excluded_features = [
    "module_presentation_length",
    "early_submitted_assessments",
    "gender",
    "region",
    "highest_education",
    "imd_band",
    "age_band",
    "disability"
]

report.extend(excluded_features)

report.append(
    "\nDemographic variables are reserved for separate "
    "fairness evaluation."
)

report.append(
    "\nStudent identifiers are retained only for grouped "
    "data splitting and are not model inputs."
)

report.append(
    "\nThis is a candidate baseline dataset. "
    "Further preprocessing and validation are required."
)

with open(report_path, "w", encoding="utf-8") as file:
    file.write("\n".join(report))

# 13. Final confirmation

print("\n" + "=" * 70)
print("CANDIDATE DATASET PREPARATION COMPLETED")
print("=" * 70)

print("Candidate dataset shape:", model_df.shape)
print("Number of model features:", len(feature_columns))

print("\nDataset saved at:")
print(output_path)

print("\nFeature selection report saved at:")
print(report_path)

print("\nOriginal dataset remains unchanged.")
