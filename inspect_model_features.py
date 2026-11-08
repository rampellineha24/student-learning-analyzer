
import pandas as pd
from pathlib import Path

# ============================================================
# FEATURE INSPECTION FOR MACHINE LEARNING
# ============================================================

print("=" * 70)
print("STUDENT LEARNING SUPPORT SYSTEM")
print("MACHINE LEARNING FEATURE INSPECTION")
print("=" * 70)

# Load dataset
dataset_path = Path("data/processed/final_early_warning_dataset.csv")
df = pd.read_csv(dataset_path)

print("\n1. DATASET INFORMATION")
print("-" * 70)

print("Dataset shape:", df.shape)
print("Total records:", len(df))
print("Total columns:", len(df.columns))

# Display all columns
print("\n2. ALL DATASET COLUMNS")
print("-" * 70)

for index, column in enumerate(df.columns, start=1):
    print(f"{index}. {column} ({df[column].dtype})")

# Target variable
print("\n3. TARGET VARIABLE")
print("-" * 70)

target = "final_result"

print("Target column:", target)
print("\nTarget distribution:")
print(df[target].value_counts())

# Unique student information
print("\n4. STUDENT AND COURSE IDENTIFIERS")
print("-" * 70)

identifier_columns = [
    "id_student",
    "code_module",
    "code_presentation"
]

for column in identifier_columns:
    if column in df.columns:
        print(f"{column}: {df[column].nunique()} unique values")

# Categorical columns
print("\n5. CATEGORICAL FEATURES")
print("-" * 70)

categorical_columns = df.select_dtypes(
    include=["object", "category"]
).columns.tolist()

for column in categorical_columns:
    print(f"\n{column}")
    print("Unique values:", df[column].nunique())
    print("Values:", df[column].unique()[:15])

# Numerical columns
print("\n6. NUMERICAL FEATURES")
print("-" * 70)

numerical_columns = df.select_dtypes(
    include=["int64", "float64"]
).columns.tolist()

for column in numerical_columns:
    print(f"{column}:")
    print(
        f"  Min={df[column].min()}, "
        f"Max={df[column].max()}, "
        f"Mean={df[column].mean():.2f}"
    )

# Missing values
print("\n7. MISSING VALUE CHECK")
print("-" * 70)

print(df.isnull().sum())

# Duplicate student-course combinations
print("\n8. STUDENT-COURSE DUPLICATE CHECK")
print("-" * 70)

if all(column in df.columns for column in identifier_columns):
    duplicate_count = df.duplicated(
        subset=identifier_columns
    ).sum()

    print("Duplicate student-course records:", duplicate_count)

# Potentially sensitive or leakage-related columns
print("\n9. COLUMNS REQUIRING FEATURE REVIEW")
print("-" * 70)

review_keywords = [
    "result",
    "score",
    "mark",
    "date",
    "cutoff",
    "registration",
    "id_"
]

for column in df.columns:
    if any(keyword in column.lower() for keyword in review_keywords):
        print("Review:", column)

# Save feature inspection report
report_path = Path("reports/eda/09_model_feature_inspection.txt")
report_path.parent.mkdir(parents=True, exist_ok=True)

with open(report_path, "w", encoding="utf-8") as file:
    file.write("MACHINE LEARNING FEATURE INSPECTION\n")
    file.write("=" * 70 + "\n\n")

    file.write(f"Dataset shape: {df.shape}\n\n")

    file.write("ALL COLUMNS:\n")
    for column in df.columns:
        file.write(f"{column}: {df[column].dtype}\n")

    file.write("\nCATEGORICAL FEATURES:\n")
    file.write(str(categorical_columns))

    file.write("\n\nNUMERICAL FEATURES:\n")
    file.write(str(numerical_columns))

    file.write("\n\nTARGET DISTRIBUTION:\n")
    file.write(df[target].value_counts().to_string())

print("\n" + "=" * 70)
print("FEATURE INSPECTION COMPLETED")
print("=" * 70)

print("Report saved at:", report_path)
