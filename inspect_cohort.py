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
print("STUDENT ELIGIBILITY AT 50% COURSE CUTOFF")
print("=" * 55)


# STEP 1: Load required datasets

students = pd.read_csv(
    os.path.join(RAW, "studentInfo.csv"),
    na_values=["?"]
)

registration = pd.read_csv(
    os.path.join(RAW, "studentRegistration.csv"),
    na_values=["?"]
)

courses = pd.read_csv(
    os.path.join(RAW, "courses.csv")
)

# STEP 2: Calculate course-specific cutoff

courses["cutoff_day"] = np.floor(
    courses["module_presentation_length"] * 0.5
).astype(int)

# STEP 3: Combine student and registration data

df = students.merge(
    registration,
    on=student_key,
    how="left",
    validate="one_to_one"
)

df = df.merge(
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

print("\n1. COMBINED DATASET")

print("Total student-course records:", len(df))

print(
    "Missing registration dates:",
    df["date_registration"].isna().sum()
)


# STEP 4: Determine eligibility

print("\n2. STUDENT ELIGIBILITY")

conditions = [
    df["date_registration"].isna(),

    df["date_registration"] > df["cutoff_day"],

    (
        df["date_unregistration"].notna()
        & (
            df["date_unregistration"]
            <= df["cutoff_day"]
        )
    )
]

choices = [
    "Unknown registration",
    "Registered after cutoff",
    "Withdrawn before or at cutoff"
]

df["eligibility_status"] = np.select(
    conditions,
    choices,
    default="Eligible at cutoff"
)

print(
    df["eligibility_status"].value_counts()
)


# STEP 5: Analyse eligible students

print("\n3. ELIGIBLE STUDENT RECORDS")

eligible = df[
    df["eligibility_status"] == "Eligible at cutoff"
].copy()

print("Eligible records:", len(eligible))

print(
    "Unique eligible students:",
    eligible["id_student"].nunique()
)


# STEP 6: Final result distribution

print("\n4. FINAL RESULTS AMONG ELIGIBLE STUDENTS")

print(
    eligible["final_result"].value_counts()
)


# STEP 7: Preliminary risk outcome groups

print("\n5. PRELIMINARY OUTCOME GROUPS")

eligible["outcome_group"] = eligible[
    "final_result"
].map({
    "Pass": "Completed without failure",
    "Distinction": "Completed without failure",
    "Fail": "Academic failure",
    "Withdrawn": "Withdrawal"
})

print(
    eligible["outcome_group"].value_counts(
        dropna=False
    )
)


# STEP 8: Final verification

print("\n6. FINAL VERIFICATION")

print(
    "Total records accounted for:",
    df["eligibility_status"].value_counts().sum()
)

print(
    "Eligibility records match original:",
    len(df) == len(students)
)

print("\nCOHORT ANALYSIS COMPLETED!")