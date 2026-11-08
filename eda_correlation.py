
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

print("=" * 60)
print("CORRELATION HEATMAP ANALYSIS")
print("=" * 60)

# Load processed dataset
data = pd.read_csv(
    "data/processed/final_early_warning_dataset.csv"
)

print("\nDataset loaded successfully:", data.shape)

# Select numerical features
numerical_features = [
    "cutoff_day",
    "module_presentation_length",
    "num_of_prev_attempts",
    "studied_credits",
    "early_total_clicks",
    "early_active_days",
    "early_submitted_assessments",
    "early_late_submissions",
    "early_on_time_submissions"
]

# Create numerical feature dataset
numerical_data = data[numerical_features]

# Calculate correlation matrix
correlation_matrix = numerical_data.corr()

print("\nCORRELATION MATRIX")
print(correlation_matrix.round(2))

# Create output directory
os.makedirs("reports/eda", exist_ok=True)

# Generate heatmap
plt.figure(figsize=(12, 9))

sns.heatmap(
    correlation_matrix,
    annot=True,
    cmap="coolwarm",
    fmt=".2f",
    linewidths=0.5,
    square=True
)

plt.title(
    "Correlation Heatmap of Numerical Early-Warning Features",
    fontsize=14
)

plt.tight_layout()

# Save graph
output_path = "reports/eda/06_correlation_heatmap.png"

plt.savefig(output_path, dpi=300)
plt.show()

print("\nGraph saved successfully at:")
print(output_path)

print("\nCORRELATION HEATMAP ANALYSIS COMPLETED!")
