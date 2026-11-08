
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

print("=" * 60)
print("EARLY ASSESSMENT SUBMISSION ANALYSIS")
print("=" * 60)

# Load the processed dataset
data = pd.read_csv(
    "data/processed/final_early_warning_dataset.csv"
)

print("\nDataset loaded successfully:", data.shape)

# Calculate assessment submission statistics
print("\nEARLY ASSESSMENT SUBMISSIONS BY FINAL RESULT")

summary = data.groupby("final_result")[
    "early_submitted_assessments"
].agg(
    ["count", "mean", "median", "min", "max"]
)

print(summary.round(2))

# Create output directory
os.makedirs("reports/eda", exist_ok=True)

# Set result order
result_order = [
    "Distinction",
    "Pass",
    "Withdrawn",
    "Fail"
]

# Create box plot
plt.figure(figsize=(10, 6))

sns.boxplot(
    data=data,
    x="final_result",
    y="early_submitted_assessments",
    order=result_order
)

plt.title(
    "Early Assessment Submissions by Final Result",
    fontsize=14
)

plt.xlabel("Final Result")
plt.ylabel("Number of Early Assessment Submissions")

plt.tight_layout()

# Save graph
output_path = (
    "reports/eda/04_assessment_submissions_by_result.png"
)

plt.savefig(output_path, dpi=300)
plt.show()

print("\nGraph saved successfully at:")
print(output_path)

print("\nEARLY ASSESSMENT SUBMISSION ANALYSIS COMPLETED!")
