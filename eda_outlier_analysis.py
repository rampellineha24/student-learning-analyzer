
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

print("=" * 60)
print("OUTLIER ANALYSIS")
print("=" * 60)

# Load processed dataset
data = pd.read_csv(
    "data/processed/final_early_warning_dataset.csv"
)

print("\nDataset loaded successfully:", data.shape)

# Features selected for outlier analysis
features = [
    "early_total_clicks",
    "early_active_days",
    "early_submitted_assessments"
]

# Calculate outlier statistics using IQR
print("\nOUTLIER SUMMARY")

outlier_summary = []

for feature in features:

    Q1 = data[feature].quantile(0.25)
    Q3 = data[feature].quantile(0.75)

    IQR = Q3 - Q1

    lower_limit = Q1 - 1.5 * IQR
    upper_limit = Q3 + 1.5 * IQR

    outliers = data[
        (data[feature] < lower_limit) |
        (data[feature] > upper_limit)
    ]

    outlier_summary.append({
        "Feature": feature,
        "Q1": Q1,
        "Q3": Q3,
        "IQR": IQR,
        "Lower Limit": lower_limit,
        "Upper Limit": upper_limit,
        "Outlier Count": len(outliers),
        "Outlier Percentage": (
            len(outliers) / len(data)
        ) * 100
    })

summary = pd.DataFrame(outlier_summary)

print(summary.round(2).to_string(index=False))

# Create output directory
os.makedirs("reports/eda", exist_ok=True)

# Prepare data for box plots
plot_data = data[features].melt(
    var_name="Feature",
    value_name="Value"
)

# Generate box plots
plt.figure(figsize=(12, 7))

sns.boxplot(
    data=plot_data,
    x="Feature",
    y="Value"
)

plt.title(
    "Outlier Analysis of Early-Warning Features",
    fontsize=14
)

plt.xlabel("Early-Warning Features")
plt.ylabel("Recorded Values")
plt.xticks(rotation=15)

plt.tight_layout()

# Save graph
output_path = "reports/eda/07_outlier_analysis.png"

plt.savefig(output_path, dpi=300)
plt.show()

print("\nGraph saved successfully at:")
print(output_path)

print("\nOUTLIER ANALYSIS COMPLETED!")
