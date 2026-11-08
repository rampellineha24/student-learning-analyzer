
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path

# File paths
DATA_PATH = Path("data/processed/final_early_warning_dataset.csv")
OUTPUT_DIR = Path("reports/eda")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 60)
print("EARLY LMS ACTIVITY ANALYSIS")
print("=" * 60)

# Load dataset
df = pd.read_csv(DATA_PATH)

print("\nDataset loaded:", df.shape)

# Calculate LMS click statistics by final result
print("\nEARLY LMS CLICKS BY FINAL RESULT")

stats = df.groupby("final_result")["early_total_clicks"].agg(
    ["count", "mean", "median", "min", "max"]
).round(2)

print(stats)

# Create a log-transformed column for visualisation
# This helps us display highly different click counts more clearly.
df["log_early_clicks"] = np.log1p(df["early_total_clicks"])

# Plot boxplot
sns.set_theme(style="whitegrid")

plt.figure(figsize=(10, 6))

order = ["Pass", "Distinction", "Fail", "Withdrawn"]

sns.boxplot(
    data=df,
    x="final_result",
    y="log_early_clicks",
    order=order
)

plt.title(
    "Early LMS Activity by Student Final Result",
    fontsize=15
)

plt.xlabel("Final Result", fontsize=12)
plt.ylabel("Log(1 + Early LMS Clicks)", fontsize=12)

plt.tight_layout()

# Save graph
output_file = OUTPUT_DIR / "02_early_clicks_by_result.png"

plt.savefig(output_file, dpi=300, bbox_inches="tight")
plt.show()

print("\nGraph saved successfully at:")
print(output_file)

print("\nEARLY LMS ACTIVITY ANALYSIS COMPLETED!")
