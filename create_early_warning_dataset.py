
import pandas as pd
import numpy as np
from pathlib import Path
import shutil

# ==========================================================
# 1. PROJECT PATHS AND SETTINGS
# ==========================================================

BASE_DIR = Path(__file__).resolve().parent
RAW_DIR = BASE_DIR / "data" / "raw"
PROCESSED_DIR = BASE_DIR / "data" / "processed"

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_FILE = PROCESSED_DIR / "final_early_warning_dataset.csv"

TEMP_DIR = PROCESSED_DIR / "temp_early_vle"
TEMP_DIR.mkdir(parents=True, exist_ok=True)

KEYS = ["code_module", "code_presentation", "id_student"]

EXPECTED_ELIGIBLE = 24592
CHUNK_SIZE = 500000

print("=" * 65)
print("FINAL EARLY-WARNING DATASET CREATION")
print("=" * 65)


# ==========================================================
# 2. LOAD COURSE INFORMATION
# ==========================================================

print("\n1. LOADING COURSE INFORMATION")

courses = pd.read_csv(RAW_DIR / "courses.csv")

courses["cutoff_day"] = np.floor(
    courses["module_presentation_length"] * 0.5
).astype(int)

print("Course presentations:", len(courses))
print("Cutoff range:",
      courses["cutoff_day"].min(),
      "to",
      courses["cutoff_day"].max())


# ==========================================================
# 3. DETERMINE STUDENT ELIGIBILITY
# ==========================================================

print("\n2. CHECKING STUDENT ELIGIBILITY")

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

# Merge course duration and cutoff into registration.
registration = registration.merge(
    courses[
        [
            "code_module",
            "code_presentation",
            "module_presentation_length",
            "cutoff_day"
        ]
    ],
    on=["code_module", "code_presentation"],
    how="left",
    validate="many_to_one"
)

# Determine eligibility at the 50% cutoff.
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

eligible = registration[
    registration["eligibility_status"] == "Eligible at cutoff"
].copy()

print("Eligible records:", len(eligible))
print("Unique eligible students:", eligible["id_student"].nunique())

if len(eligible) != EXPECTED_ELIGIBLE:
    raise ValueError(
        f"Expected {EXPECTED_ELIGIBLE} eligible records, "
        f"but found {len(eligible)}."
    )

# Keep course length and cutoff for later feature validation.
eligible_keys = eligible[
    KEYS + [
        "cutoff_day",
        "module_presentation_length"
    ]
].copy()

print("Eligibility verification: PASSED")


# ==========================================================
# 4. LOAD STUDENT BACKGROUND INFORMATION
# ==========================================================

print("\n3. LOADING STUDENT BACKGROUND INFORMATION")

student_info = pd.read_csv(
    RAW_DIR / "studentInfo.csv"
)

background_columns = [
    "id_student",
    "code_module",
    "code_presentation",
    "gender",
    "region",
    "highest_education",
    "imd_band",
    "age_band",
    "num_of_prev_attempts",
    "studied_credits",
    "disability",
    "final_result"
]

background = student_info[background_columns].copy()

dataset = eligible_keys.merge(
    background,
    on=KEYS,
    how="left",
    validate="one_to_one",
    indicator=True
)

if not (dataset["_merge"] == "both").all():
    raise ValueError(
        "Some eligible records have no matching student information."
    )

dataset.drop(columns=["_merge"], inplace=True)

print("Background records merged:", len(dataset))


# ==========================================================
# 5. PROCESS EARLY LMS ACTIVITY
# ==========================================================

print("\n4. PROCESSING EARLY LMS ACTIVITY")

student_vle_file = RAW_DIR / "studentVle.csv"

cutoff_lookup = eligible_keys[
    KEYS + ["cutoff_day"]
]

chunk_number = 0
total_early_lms_rows = 0

# Process the large LMS file in smaller chunks.
for chunk in pd.read_csv(
    student_vle_file,
    chunksize=CHUNK_SIZE
):

    chunk = chunk.merge(
        cutoff_lookup,
        on=KEYS,
        how="inner",
        validate="many_to_one"
    )

    chunk["date"] = pd.to_numeric(
        chunk["date"],
        errors="coerce"
    )

    # Keep activity only between day 0 and the
    # individual course's 50% cutoff.
    chunk = chunk[
        (chunk["date"] >= 0)
        & (chunk["date"] <= chunk["cutoff_day"])
    ]

    if not chunk.empty:

        total_early_lms_rows += len(chunk)

        # Aggregate clicks by student-course-day.
        daily_activity = (
            chunk.groupby(
                KEYS + ["date"],
                as_index=False
            )["sum_click"].sum()
        )

        chunk_number += 1

        daily_activity.to_csv(
            TEMP_DIR / f"vle_part_{chunk_number}.csv",
            index=False
        )

    print(
        f"LMS chunks processed: {chunk_number} | "
        f"Early records: {total_early_lms_rows}"
    )

print("\nTotal early LMS interaction records:",
      total_early_lms_rows)


# Combine daily activity from all chunks.
daily_files = sorted(
    TEMP_DIR.glob("vle_part_*.csv")
)

if daily_files:

    daily_parts = [
        pd.read_csv(file)
        for file in daily_files
    ]

    daily = pd.concat(
        daily_parts,
        ignore_index=True
    )

    # Combine duplicate student-course-day entries.
    daily = (
        daily.groupby(
            KEYS + ["date"],
            as_index=False
        )["sum_click"].sum()
    )

    # Create LMS features.
    lms_features = (
        daily.groupby(KEYS)
        .agg(
            early_total_clicks=("sum_click", "sum"),
            early_active_days=("date", "nunique")
        )
        .reset_index()
    )

else:

    lms_features = pd.DataFrame(
        columns=KEYS + [
            "early_total_clicks",
            "early_active_days"
        ]
    )

dataset = dataset.merge(
    lms_features,
    on=KEYS,
    how="left",
    validate="one_to_one"
)

dataset["early_total_clicks"] = (
    dataset["early_total_clicks"]
    .fillna(0)
    .astype(int)
)

dataset["early_active_days"] = (
    dataset["early_active_days"]
    .fillna(0)
    .astype(int)
)

print("Early LMS features created successfully.")


# ==========================================================
# 6. PROCESS EARLY ASSESSMENT SUBMISSIONS
# ==========================================================

print("\n5. PROCESSING EARLY ASSESSMENT SUBMISSIONS")

assessments = pd.read_csv(
    RAW_DIR / "assessments.csv"
)

student_assessments = pd.read_csv(
    RAW_DIR / "studentAssessment.csv"
)

# Attach assessment deadlines.
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

# Retain only eligible student-course records.
assessment_data = assessment_data.merge(
    eligible_keys[KEYS + ["cutoff_day"]],
    on=KEYS,
    how="inner",
    validate="many_to_one"
)

# Keep submissions made between day 0 and cutoff.
early_assessments = assessment_data[
    (assessment_data["date_submitted"] >= 0)
    & (
        assessment_data["date_submitted"]
        <= assessment_data["cutoff_day"]
    )
].copy()

# Identify late and on-time submissions.
early_assessments["is_late"] = (
    early_assessments["date"].notna()
    & (early_assessments["date"] >= 0)
    & (
        early_assessments["date_submitted"]
        > early_assessments["date"]
    )
)

early_assessments["is_on_time"] = (
    early_assessments["date"].notna()
    & (early_assessments["date"] >= 0)
    & (
        early_assessments["date_submitted"]
        <= early_assessments["date"]
    )
)

# Aggregate submission behaviour.
assessment_features = (
    early_assessments.groupby(KEYS)
    .agg(
        early_submitted_assessments=("id_assessment", "count"),
        early_late_submissions=("is_late", "sum"),
        early_on_time_submissions=("is_on_time", "sum")
    )
    .reset_index()
)

dataset = dataset.merge(
    assessment_features,
    on=KEYS,
    how="left",
    validate="one_to_one"
)

assessment_columns = [
    "early_submitted_assessments",
    "early_late_submissions",
    "early_on_time_submissions"
]

for column in assessment_columns:
    dataset[column] = (
        dataset[column].fillna(0).astype(int)
    )

print("Early assessment submission records:",
      len(early_assessments))

print("Assessment features created successfully.")


# ==========================================================
# 7. FINAL DATASET PREPARATION
# ==========================================================

print("\n6. FINAL DATASET PREPARATION")

# Detect duplicates instead of silently deleting them.
duplicate_count = dataset.duplicated(
    subset=KEYS
).sum()

if duplicate_count > 0:
    raise ValueError(
        f"Duplicate student-course records detected: {duplicate_count}"
    )

dataset = dataset.sort_values(
    KEYS
).reset_index(drop=True)

print("\nFinal dataset columns:")

for column in dataset.columns:
    print("-", column)


# ==========================================================
# 8. DATASET VALIDATION
# ==========================================================

print("\n7. DATASET VALIDATION")

print("Rows:", len(dataset))
print("Columns:", len(dataset.columns))

print(
    "Duplicate student-course records:",
    dataset.duplicated(subset=KEYS).sum()
)

print("\nMissing values:")
print(dataset.isnull().sum())

print("\nFinal result distribution:")
print(dataset["final_result"].value_counts(dropna=False))

print("\nEarly behaviour summary:")
print(
    dataset[
        [
            "early_total_clicks",
            "early_active_days",
            "early_submitted_assessments",
            "early_late_submissions",
            "early_on_time_submissions"
        ]
    ].describe()
)

# Validate target and record counts.
assert len(dataset) == EXPECTED_ELIGIBLE

assert dataset["final_result"].notna().all()

assert dataset.duplicated(subset=KEYS).sum() == 0

# Validate assessment counts.
assert (
    dataset["early_late_submissions"]
    <= dataset["early_submitted_assessments"]
).all()

assert (
    dataset["early_on_time_submissions"]
    <= dataset["early_submitted_assessments"]
).all()

assert (
    dataset["early_late_submissions"]
    + dataset["early_on_time_submissions"]
    <= dataset["early_submitted_assessments"]
).all()

# Validate LMS feature ranges.
assert (dataset["early_total_clicks"] >= 0).all()
assert (dataset["early_active_days"] >= 0).all()

assert (
    dataset["early_active_days"]
    <= dataset["cutoff_day"] + 1
).all()

# Validate cutoff metadata.
assert dataset["cutoff_day"].notna().all()

print("\nBasic validation: PASSED")


# ==========================================================
# 9. SAVE FINAL DATASET
# ==========================================================

print("\n8. SAVING FINAL DATASET")

dataset.to_csv(
    OUTPUT_FILE,
    index=False
)

print("Saved successfully:")
print(OUTPUT_FILE)

print("Final dataset shape:", dataset.shape)

# Remove temporary aggregation files after successful save.
shutil.rmtree(TEMP_DIR)

print("\n" + "=" * 65)
print("EARLY-WARNING DATASET CREATION COMPLETED!")
print("=" * 65)
