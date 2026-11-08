
import pandas as pd
import numpy as np
import joblib

from pathlib import Path
from sklearn.model_selection import GroupShuffleSplit
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# ============================================================
# ML PREPROCESSING AND STUDENT-LEVEL DATA SPLITTING
# ============================================================

print("=" * 65)
print("STUDENT LEARNING SUPPORT SYSTEM")
print("ML PREPROCESSING AND TRAIN-TEST SPLITTING")
print("=" * 65)

# 1. Load candidate dataset

input_path = Path(
    "data/processed/candidate_model_dataset.csv"
)

df = pd.read_csv(input_path)

print("\n1. DATASET LOADED")
print("-" * 65)
print("Dataset shape:", df.shape)

# 2. Define columns

target = "final_result"
group = "id_student"

categorical_features = [
    "code_module",
    "code_presentation"
]

numerical_features = [
    "cutoff_day",
    "num_of_prev_attempts",
    "studied_credits",
    "early_total_clicks",
    "early_active_days",
    "early_on_time_submissions",
    "early_late_submissions"
]

feature_columns = categorical_features + numerical_features

X = df[feature_columns]
y = df[target]
groups = df[group]

# 3. Student-level train-test split

print("\n2. STUDENT-LEVEL TRAIN-TEST SPLIT")
print("-" * 65)

splitter = GroupShuffleSplit(
    n_splits=1,
    test_size=0.20,
    random_state=42
)

train_indices, test_indices = next(
    splitter.split(X, y, groups=groups)
)

X_train = X.iloc[train_indices].copy()
X_test = X.iloc[test_indices].copy()

y_train = y.iloc[train_indices].copy()
y_test = y.iloc[test_indices].copy()

train_groups = groups.iloc[train_indices]
test_groups = groups.iloc[test_indices]

print("Training records:", len(X_train))
print("Testing records:", len(X_test))

print("Training students:", train_groups.nunique())
print("Testing students:", test_groups.nunique())

# 4. Verify no student leakage

overlap = set(train_groups) & set(test_groups)

print("\n3. STUDENT LEAKAGE CHECK")
print("-" * 65)
print("Students present in both sets:", len(overlap))

if len(overlap) != 0:
    raise ValueError("Student-level data leakage detected!")

print("Student-level split verified successfully.")

# 5. Target distribution

print("\n4. TRAINING TARGET DISTRIBUTION")
print("-" * 65)

print(
    y_train.value_counts().to_string()
)

print("\nTraining percentages:")
print(
    (y_train.value_counts(normalize=True) * 100)
    .round(2).to_string()
)

print("\n5. TESTING TARGET DISTRIBUTION")
print("-" * 65)

print(
    y_test.value_counts().to_string()
)

print("\nTesting percentages:")
print(
    (y_test.value_counts(normalize=True) * 100)
    .round(2).to_string()
)

# 6. Create preprocessing transformer

print("\n6. PREPROCESSING CONFIGURATION")
print("-" * 65)

preprocessor = ColumnTransformer(
    transformers=[
        (
            "categorical",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_features
        ),
        (
            "numerical",
            StandardScaler(),
            numerical_features
        )
    ]
)

print("Categorical features:", categorical_features)
print("Numerical features:", numerical_features)
print("OneHotEncoder: Configured")
print("StandardScaler: Configured")
print("ColumnTransformer: Configured")

# 7. Save split datasets

output_dir = Path("data/processed/ml")
output_dir.mkdir(parents=True, exist_ok=True)

train_df = X_train.copy()
train_df[group] = train_groups.values
train_df[target] = y_train.values

test_df = X_test.copy()
test_df[group] = test_groups.values
test_df[target] = y_test.values

train_df.to_csv(
    output_dir / "train_dataset.csv",
    index=False
)

test_df.to_csv(
    output_dir / "test_dataset.csv",
    index=False
)

# 8. Save preprocessing configuration

joblib.dump(
    {
        "preprocessor": preprocessor,
        "categorical_features": categorical_features,
        "numerical_features": numerical_features,
        "target": target,
        "group": group
    },
    output_dir / "preprocessing_config.pkl"
)

# 9. Final report

print("\n7. SAVED FILES")
print("-" * 65)

print(output_dir / "train_dataset.csv")
print(output_dir / "test_dataset.csv")
print(output_dir / "preprocessing_config.pkl")

print("\n" + "=" * 65)
print("ML DATA PREPARATION COMPLETED")
print("=" * 65)
