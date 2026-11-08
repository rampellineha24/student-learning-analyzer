
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Project paths
DATA_PATH = Path("data/processed/final_early_warning_dataset.csv")
OUTPUT_DIR = Path("reports/eda")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 60)
print("EXPLORATORY DATA ANALYSIS")
print("=" * 60)

# Load the final early-warning dataset
df = pd.read_csv(DATA_PATH)

print("\nDataset loaded successfully!")
print("Rows:", df.shape[0])
print("Columns:", df.shape[1])

# Check final result distribution
result_counts = df["final_result"].value_counts()

print("\nFINAL RESULT DISTRIBUTION")
print(result_counts)

print("\nFINAL RESULT PERCENTAGES")
print((df["final_result"].value_counts(normalize=True) * 100).round(2))

# Plot student outcome distribution
sns.set_theme(style="whitegrid")

plt.figure(figsize=(9, 6))

order = ["Pass", "Distinction", "Fail", "Withdrawn"]

ax = sns.countplot(
    data=df,
    x="final_result",
    order=order
)

plt.title("Distribution of Student Final Results", fontsize=15)
plt.xlabel("Final Result", fontsize=12)
plt.ylabel("Number of Students", fontsize=12)

# Display count labels above bars
for bar in ax.patches:
    height = bar.get_height()
    ax.annotate(
        f"{int(height):,}",
        (bar.get_x() + bar.get_width() / 2, height),
        ha="center",
        va="bottom",
        fontsize=10
    )

plt.tight_layout()

# Save graph
output_file = OUTPUT_DIR / "01_final_result_distribution.png"
plt.savefig(output_file, dpi=300, bbox_inches="tight")

plt.show()

print("\nGraph saved successfully at:")
print(output_file)

print("\nFIRST EDA ANALYSIS COMPLETED!")
