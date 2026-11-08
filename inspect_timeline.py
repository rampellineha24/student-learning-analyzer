import pandas as pd
import os

RAW = "data/raw"

print("=" * 55)
print("COURSE TIMELINE ANALYSIS")
print("=" * 55)


# STEP 1: Course duration

print("\n1. COURSE DURATION")

courses = pd.read_csv(
    os.path.join(RAW, "courses.csv")
)

print(courses["module_presentation_length"].describe())

print("\nCourse presentations:")
print(courses.head(10))


# STEP 2: LMS activity dates

print("\n2. LMS ACTIVITY TIMELINE")

date_counts = {
    "before_start": 0,
    "day_zero": 0,
    "after_start": 0
}

minimum_date = None
maximum_date = None

for chunk in pd.read_csv(
    os.path.join(RAW, "studentVle.csv"),
    chunksize=500000
):

    dates = chunk["date"]

    if minimum_date is None:
        minimum_date = dates.min()
        maximum_date = dates.max()
    else:
        minimum_date = min(minimum_date, dates.min())
        maximum_date = max(maximum_date, dates.max())

    date_counts["before_start"] += (dates < 0).sum()
    date_counts["day_zero"] += (dates == 0).sum()
    date_counts["after_start"] += (dates > 0).sum()

print("Earliest LMS activity date:", minimum_date)
print("Latest LMS activity date:", maximum_date)

print("Activity before course start:", date_counts["before_start"])
print("Activity on course start:", date_counts["day_zero"])
print("Activity after course start:", date_counts["after_start"])


# STEP 3: Assessment timeline

print("\n3. ASSESSMENT TIMELINE")

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

print("\nAssessment deadlines:")
print(assessments["date"].describe())

print("\nAssessment deadlines by type:")
print(
    assessments.groupby("assessment_type")["date"]
    .agg(["count", "min", "max"])
)

print("\nStudent submission dates:")
print(student_assessments["date_submitted"].describe())

print(
    "\nMissing submission dates:",
    student_assessments["date_submitted"].isna().sum()
)

print(
    "\nMissing assessment deadlines:",
    assessment_data["date"].isna().sum()
)


# STEP 4: Registration timeline

print("\n4. STUDENT REGISTRATION TIMELINE")

registration = pd.read_csv(
    os.path.join(RAW, "studentRegistration.csv"),
    na_values=["?"]
)

print("\nRegistration dates:")
print(registration["date_registration"].describe())

print("\nUnregistration dates:")
print(registration["date_unregistration"].describe())


print("\n" + "=" * 55)
print("TIMELINE ANALYSIS COMPLETED!")
print("=" * 55)