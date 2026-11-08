
import pandas as pd
import numpy as np
from pathlib import Path

# ======================================================
# 1. PROJECT PATHS
# ======================================================

BASE_DIR = Path(__file__).resolve().parent
RAW_DIR = BASE_DIR / "data" / "raw"

KEYS = [
    "code_module",
    "code_presentation",
    "id_student"
]

print("=" * 60)
print("ASSESSMENT COUNT VERIFICATION")
print("=" * 60)


# ======================================================
# 2. LOAD COURSE AND REGISTRATION DATA
# ======================================================

courses = pd.read_csv(RAW_DIR / "courses.csv")

courses["cutoff_day"] = np.floor(
    courses["module_presentation_length"] * 0.5
).astype(int)

registration = pd.read_csv(
    RAW_DIR / "studentRegistration.csv"
)

registration["date_registration"] = pd.to_numeric(
    registration["date_registration"],
    errors="coerce"
)

registration["date_unregistration"] = pd.to_numeric(
    registration["date_unregistration"],
    errors="coerce"
)

registration = registration.merge(
    courses[
        [
            "code_module",
            "code_presentation",
            "cutoff_day"
        ]
    ],
    on=["code_module", "code_presentation"],
    how="left",
    validate="many_to_one"
)

registration["eligibility_status"] = np.select(
    [
        registration["date_registration"].isna(),

        registration["date_registration"]
        > registration["cutoff_day"],

        registration["date_unregistration"].notna()
        & (
            registration["date_unregistration"]
            <= registration["cutoff_day"]
        )
    ],
    [
        "Unknown registration",
        "Registered after cutoff",
        "Withdrawn before or at cutoff"
    ],
    default="Eligible at cutoff"
)

print("\n1. ELIGIBILITY COUNTS")

print(
    registration["eligibility_status"].value_counts()
)


# ======================================================
# 3. LOAD ASSESSMENT DATA
# ======================================================

assessments = pd.read_csv(
    RAW_DIR / "assessments.csv"
)

student_assessments = pd.read_csv(
    RAW_DIR / "studentAssessment.csv"
)

assessment_data = student_assessments.merge(
    assessments[
        [
            "id_assessment",
            "code_module",
            "code_presentation",
            "date"
        ]
    ],
    on="id_assessment",
    how="left",
    validate="many_to_one"
)

assessment_data["date_submitted"] = pd.to_numeric(
    assessment_data["date_submitted"],
    errors="coerce"
)

assessment_data["date"] = pd.to_numeric(
    assessment_data["date"],
    errors="coerce"
)


# ======================================================
# 4. ATTACH ELIGIBILITY INFORMATION
# ======================================================

assessment_data = assessment_data.merge(
    registration[
        KEYS + ["cutoff_day", "eligibility_status"]
    ],
    on=KEYS,
    how="left",
    validate="many_to_one"
)

print("\n2. ASSESSMENT RECORD CHECK")

print("Total assessment records:", len(assessment_data))

print(
    "Missing eligibility matches:",
    assessment_data["eligibility_status"].isna().sum()
)


# ======================================================
# 5. FILTER EARLY SUBMISSIONS
# ======================================================

early_assessments = assessment_data[
    (assessment_data["date_submitted"] >= 0)
    & (
        assessment_data["date_submitted"]
        <= assessment_data["cutoff_day"]
    )
].copy()

print("\n3. EARLY ASSESSMENT SUBMISSIONS")

print(
    "Total early submissions across all records:",
    len(early_assessments)
)


# ======================================================
# 6. BREAKDOWN BY ELIGIBILITY
# ======================================================

print("\n4. EARLY SUBMISSIONS BY ELIGIBILITY STATUS")

print(
    early_assessments["eligibility_status"]
    .value_counts(dropna=False)
)

eligible_early = early_assessments[
    early_assessments["eligibility_status"]
    == "Eligible at cutoff"
]

print("\n5. ELIGIBLE COHORT COMPARISON")

print(
    "Early submissions in eligible cohort:",
    len(eligible_early)
)

print(
    "Early submissions outside eligible cohort:",
    len(early_assessments) - len(eligible_early)
)

print(
    "Earlier reported total:",
    95432
)

print(
    "Earlier total minus eligible total:",
    95432 - len(eligible_early)
)


# ======================================================
# 7. FINAL VERIFICATION
# ======================================================

print("\n6. VERIFICATION RESULTS")

if len(early_assessments) == 95432:
    print("Full early assessment count: MATCHED")
else:
    print(
        "Full early assessment count: DIFFERENT",
        "| Calculated:",
        len(early_assessments)
    )

if len(eligible_early) == 90433:
    print("Eligible early assessment count: MATCHED")
else:
    print(
        "Eligible early assessment count: DIFFERENT",
        "| Calculated:",
        len(eligible_early)
    )

print("\n" + "=" * 60)
print("ASSESSMENT VERIFICATION COMPLETED")
print("=" * 60)
