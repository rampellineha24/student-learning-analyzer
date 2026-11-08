
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# File paths
DATA_PATH = Path("data/processed/final_early_warning_dataset.csv")
OUTPUT_DIR = Path("reports/eda")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 60)
print("EARLY ACTIVE DAYS ANALYSIS")
print("=" * 60)

# Load dataset
df = pd.read_csv(DATA_PATH)

print("\nDataset loaded:", df.shape)

# Calculate active day statistics
print("\nEARLY ACTIVE DAYS BY FINAL RESULT")

stats = df.groupby("final_result")["early_active_days"].agg(
    ["count", "mean", "median", "min", "max"]
).round(2)

print(stats)

# Plot boxplot
sns.set_theme(style="whitegrid")

plt.figure(figsize=(10, 6))

order = ["Pass", "Distinction", "Fail", "Withdrawn"]

sns.boxplot(
    data=df,
    x="final_result",
    y="early_active_days",
    order=order
)

plt.title(
    "Early Active Learning Days by Student Final Result",
    fontsize=15
)

plt.xlabel("Final Result", fontsize=12)
plt.ylabel("Number of Early Active Days", fontsize=12)

plt.tight_layout()

# Save graph
output_file = OUTPUT_DIR / "03_active_days_by_result.png"

plt.savefig(output_file, dpi=300, bbox_inches="tight")
plt.show()

print("\nGraph saved successfully at:")
print(output_file)

print("\nEARLY ACTIVE DAYS ANALYSIS COMPLETED!")
