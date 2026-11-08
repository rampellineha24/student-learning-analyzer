
import pandas as pd
import os

print("=" * 65)
print("STUDENT LEARNING SUPPORT SYSTEM - EDA SUMMARY")
print("=" * 65)

# Load processed dataset
data = pd.read_csv(
    "data/processed/final_early_warning_dataset.csv"
)

print("\nDataset loaded successfully:", data.shape)

# Create reports directory
os.makedirs("reports/eda", exist_ok=True)

# 1. Final result distribution
result_summary = data["final_result"].value_counts().to_frame(
    name="Student Count"
)

result_summary["Percentage"] = (
    data["final_result"].value_counts(normalize=True) * 100
)

# 2. LMS activity summary
lms_summary = data.groupby("final_result").agg(
    Mean_Clicks=("early_total_clicks", "mean"),
    Median_Clicks=("early_total_clicks", "median"),
    Mean_Active_Days=("early_active_days", "mean"),
    Median_Active_Days=("early_active_days", "median")
)

# 3. Assessment submission summary
assessment_summary = data.groupby("final_result").agg(
    Mean_Submissions=("early_submitted_assessments", "mean"),
    Median_Submissions=("early_submitted_assessments", "median"),
    Mean_On_Time=("early_on_time_submissions", "mean"),
    Mean_Late=("early_late_submissions", "mean")
)

# 4. Outlier summary using IQR
features = [
    "early_total_clicks",
    "early_active_days",
    "early_submitted_assessments"
]

outlier_results = []

for feature in features:

    Q1 = data[feature].quantile(0.25)
    Q3 = data[feature].quantile(0.75)
    IQR = Q3 - Q1

    lower_limit = Q1 - 1.5 * IQR
    upper_limit = Q3 + 1.5 * IQR

    outlier_count = (
        (data[feature] < lower_limit) |
        (data[feature] > upper_limit)
    ).sum()

    outlier_percentage = (
        outlier_count / len(data)
    ) * 100

    outlier_results.append({
        "Feature": feature,
        "Outlier Count": outlier_count,
        "Outlier Percentage": round(
            outlier_percentage, 2
        )
    })

outlier_summary = pd.DataFrame(outlier_results)

# Display summaries
print("\n1. FINAL RESULT DISTRIBUTION")
print(result_summary.round(2))

print("\n2. EARLY LMS ACTIVITY SUMMARY")
print(lms_summary.round(2))

print("\n3. ASSESSMENT SUBMISSION SUMMARY")
print(assessment_summary.round(2))

print("\n4. OUTLIER SUMMARY")
print(outlier_summary.to_string(index=False))

# Save all summaries into one text report
report_path = "reports/eda/08_eda_findings_summary.txt"

with open(report_path, "w", encoding="utf-8") as file:

    file.write("STUDENT LEARNING SUPPORT SYSTEM\n")
    file.write("EXPLORATORY DATA ANALYSIS SUMMARY\n")
    file.write("=" * 60 + "\n\n")

    file.write(f"Dataset shape: {data.shape}\n\n")

    file.write("1. FINAL RESULT DISTRIBUTION\n")
    file.write(result_summary.round(2).to_string())
    file.write("\n\n")

    file.write("2. EARLY LMS ACTIVITY SUMMARY\n")
    file.write(lms_summary.round(2).to_string())
    file.write("\n\n")

    file.write("3. ASSESSMENT SUBMISSION SUMMARY\n")
    file.write(assessment_summary.round(2).to_string())
    file.write("\n\n")

    file.write("4. OUTLIER SUMMARY\n")
    file.write(outlier_summary.to_string(index=False))
    file.write("\n\n")

    file.write("EDA SUMMARY COMPLETED\n")

print("\nSummary report saved successfully at:")
print(report_path)

print("\nALL EDA SUMMARY ANALYSES COMPLETED!")
