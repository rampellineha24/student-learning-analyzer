
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

print("=" * 60)
print("EARLY ASSESSMENT SUBMISSION TIMING ANALYSIS")
print("=" * 60)

# Load the processed dataset
data = pd.read_csv(
    "data/processed/final_early_warning_dataset.csv"
)

print("\nDataset loaded successfully:", data.shape)

# Calculate on-time and late submission statistics
print("\nON-TIME SUBMISSIONS BY FINAL RESULT")

on_time_summary = data.groupby("final_result")[
    "early_on_time_submissions"
].agg(["count", "mean", "median", "min", "max"])

print(on_time_summary.round(2))

print("\nLATE SUBMISSIONS BY FINAL RESULT")

late_summary = data.groupby("final_result")[
    "early_late_submissions"
].agg(["count", "mean", "median", "min", "max"])

print(late_summary.round(2))

# Create output directory
os.makedirs("reports/eda", exist_ok=True)

# Prepare data for visualisation
result_order = [
    "Distinction",
    "Pass",
    "Withdrawn",
    "Fail"
]

timing_data = data.melt(
    id_vars=["final_result"],
    value_vars=[
        "early_on_time_submissions",
        "early_late_submissions"
    ],
    var_name="submission_type",
    value_name="submission_count"
)

# Rename categories for readability
timing_data["submission_type"] = timing_data[
    "submission_type"
].map({
    "early_on_time_submissions": "On-Time",
    "early_late_submissions": "Late"
})

# Create comparison graph
plt.figure(figsize=(11, 6))

sns.barplot(
    data=timing_data,
    x="final_result",
    y="submission_count",
    hue="submission_type",
    order=result_order,
    errorbar=None
)

plt.title(
    "Average Early On-Time vs Late Submissions by Final Result",
    fontsize=14
)

plt.xlabel("Final Result")
plt.ylabel("Average Number of Submissions")
plt.legend(title="Submission Type")

plt.tight_layout()

# Save graph
output_path = (
    "reports/eda/05_early_submission_timing_by_result.png"
)

plt.savefig(output_path, dpi=300)
plt.show()

print("\nGraph saved successfully at:")
print(output_path)

print("\nEARLY SUBMISSION TIMING ANALYSIS COMPLETED!")
