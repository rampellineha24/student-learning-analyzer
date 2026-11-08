
import pandas as pd
import numpy as np
from pathlib import Path

# ============================================================
# FEATURE RELATIONSHIP AND DATA LEAKAGE AUDIT
# ============================================================

print("=" * 70)
print("FEATURE RELATIONSHIP AUDIT")
print("STUDENT LEARNING SUPPORT SYSTEM")
print("=" * 70)

df = pd.read_csv(
    "data/processed/final_early_warning_dataset.csv"
)

report = []

def section(title):
    report.append("\n" + "=" * 70)
    report.append(title)
    report.append("=" * 70)

def show(message=""):
    print(message)
    report.append(str(message))


# ------------------------------------------------------------
# 1. DATASET OVERVIEW
# ------------------------------------------------------------

section("1. DATASET OVERVIEW")

show(f"Total records: {len(df)}")
show(f"Total columns: {len(df.columns)}")


# ------------------------------------------------------------
# 2. STUDENT RECORD DISTRIBUTION
# ------------------------------------------------------------

section("2. STUDENT RECORD DISTRIBUTION")

student_records = df.groupby("id_student").size()

show(f"Unique students: {df['id_student'].nunique()}")
show(f"Students with one record: {(student_records == 1).sum()}")
show(f"Students with multiple records: {(student_records > 1).sum()}")
show(f"Maximum records for one student: {student_records.max()}")

show("\nRecords per student distribution:")
show(student_records.value_counts().sort_index().to_string())


# ------------------------------------------------------------
# 3. COURSE LENGTH AND CUTOFF RELATIONSHIP
# ------------------------------------------------------------

section("3. CUTOFF DAY AND COURSE LENGTH")

correlation = df[
    ["cutoff_day", "module_presentation_length"]
].corr().iloc[0, 1]

difference = (
    df["module_presentation_length"] - df["cutoff_day"]
)

show(f"Correlation: {correlation:.4f}")
show(f"Unique cutoff days: {df['cutoff_day'].nunique()}")
show(
    f"Unique course lengths: "
    f"{df['module_presentation_length'].nunique()}"
)
show(f"Minimum difference: {difference.min()}")
show(f"Maximum difference: {difference.max()}")
show(f"Unique differences: {difference.nunique()}")

if correlation == 1.0:
    show(
        "\nObservation: Both features have a perfect positive "
        "linear correlation. Investigate redundancy before selection."
    )


# ------------------------------------------------------------
# 4. ASSESSMENT SUBMISSION RELATIONSHIP
# ------------------------------------------------------------

section("4. ASSESSMENT SUBMISSION RELATIONSHIP")

calculated_total = (
    df["early_on_time_submissions"]
    + df["early_late_submissions"]
)

difference = (
    df["early_submitted_assessments"] - calculated_total
)

show(
    "Records where total submissions equal "
    "on-time plus late submissions: "
    f"{(difference == 0).sum()}"
)

show(
    "Records where the values do not match: "
    f"{(difference != 0).sum()}"
)

show(f"Minimum difference: {difference.min()}")
show(f"Maximum difference: {difference.max()}")

show("\nCorrelation matrix:")
submission_columns = [
    "early_submitted_assessments",
    "early_on_time_submissions",
    "early_late_submissions"
]

show(
    df[submission_columns].corr().round(3).to_string()
)


# ------------------------------------------------------------
# 5. IMD BAND SPECIAL VALUE CHECK
# ------------------------------------------------------------

section("5. IMD BAND SPECIAL VALUE CHECK")

show("IMD band value counts:")
show(df["imd_band"].value_counts(dropna=False).to_string())

unknown_count = (
    df["imd_band"].astype(str).str.strip() == "?"
).sum()

show(f"\nNumber of '?' values: {unknown_count}")
show(f"Percentage of dataset: {unknown_count / len(df) * 100:.2f}%")


# ------------------------------------------------------------
# 6. CATEGORICAL FEATURE DISTRIBUTIONS
# ------------------------------------------------------------

section("6. CATEGORICAL FEATURE DISTRIBUTIONS")

categorical_columns = [
    "code_module",
    "code_presentation",
    "gender",
    "region",
    "highest_education",
    "age_band",
    "disability"
]

for column in categorical_columns:

    if column in df.columns:

        show(f"\n{column}:")
        show(df[column].value_counts().to_string())


# ------------------------------------------------------------
# 7. FINAL RESULT BY SELECTED GROUPS
# ------------------------------------------------------------

section("7. OUTCOME DISTRIBUTION ACROSS GROUPS")

group_columns = [
    "gender",
    "age_band",
    "highest_education",
    "disability"
]

for column in group_columns:

    if column in df.columns:

        show(f"\nFinal result distribution by {column}:")

        table = pd.crosstab(
            df[column],
            df["final_result"],
            normalize="index"
        ) * 100

        show(table.round(2).to_string())


# ------------------------------------------------------------
# 8. FINAL AUDIT NOTES
# ------------------------------------------------------------

section("8. AUDIT NOTES")

show(
    "1. Student identifiers are required for grouped "
    "train-test splitting, not as model inputs."
)

show(
    "2. Highly correlated features should be investigated "
    "before deciding which features to retain."
)

show(
    "3. Submission counts may contain mathematical redundancy."
)

show(
    "4. Special '?' values in IMD band require appropriate "
    "missing-category treatment."
)

show(
    "5. Demographic features require careful consideration "
    "and fairness evaluation before model inclusion."
)

show(
    "6. Correlation and group differences do not establish causation."
)


# ------------------------------------------------------------
# 9. SAVE REPORT
# ------------------------------------------------------------

output_path = Path(
    "reports/eda/10_feature_relationship_audit.txt"
)

output_path.parent.mkdir(parents=True, exist_ok=True)

with open(output_path, "w", encoding="utf-8") as file:
    file.write("\n".join(report))

print("\n" + "=" * 70)
print("FEATURE RELATIONSHIP AUDIT COMPLETED")
print("=" * 70)

print("Report saved at:", output_path)
