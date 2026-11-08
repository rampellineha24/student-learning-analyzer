
import pandas as pd

path = "data/raw/"

students = pd.read_csv(path + "studentInfo.csv", na_values=["?"])
registration = pd.read_csv(path + "studentRegistration.csv", na_values=["?"])
assessments = pd.read_csv(path + "assessments.csv", na_values=["?"])
student_assessment = pd.read_csv(
    path + "studentAssessment.csv", na_values=["?"]
)

print("\n========== STUDENT INFORMATION ==========")

print("Final result distribution:")
print(students["final_result"].value_counts())

student_key = ["code_module", "code_presentation", "id_student"]

print("\nDuplicate student records:")
print(students.duplicated(subset=student_key).sum())

print("\n========== REGISTRATION ==========")

print("Missing registration dates:")
print(registration["date_registration"].isnull().sum())

print("\nWithdrawal comparison:")
print(pd.crosstab(
    registration["date_unregistration"].isnull(),
    students["final_result"]
))

print("\n========== ASSESSMENTS ==========")

print("Assessment types:")
print(assessments["assessment_type"].value_counts())

print("\nMissing assessment deadlines by type:")
print(
    assessments.groupby("assessment_type")["date"]
    .apply(lambda x: x.isnull().sum())
)

print("\n========== STUDENT ASSESSMENT ==========")

print("Missing scores:")
print(student_assessment["score"].isnull().sum())

print("\nScore range:")
print(student_assessment["score"].min())
print(student_assessment["score"].max())

print("\nDuplicate assessment records:")
print(
    student_assessment.duplicated(
        subset=["id_assessment", "id_student"]
    ).sum()
)

print("\nDATA RELATIONSHIP CHECK COMPLETED!")
