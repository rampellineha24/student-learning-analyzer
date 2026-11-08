
import pandas as pd
import joblib
import matplotlib.pyplot as plt

from pathlib import Path
from sklearn.inspection import permutation_importance
from sklearn.metrics import f1_score

print("=" * 70)
print("STUDENT LEARNING SUPPORT SYSTEM")
print("MODEL EXPLAINABILITY - PERMUTATION FEATURE IMPORTANCE")
print("=" * 70)

# 1. Load saved model

model_path = Path("models/best_student_risk_model.pkl")
test_path = Path("data/processed/ml/test_dataset.csv")

model = joblib.load(model_path)
test_df = pd.read_csv(test_path)

print("\n1. MODEL AND TEST DATA LOADED")
print("-" * 70)

# 2. Select the original nine model features

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

features = categorical_features + numerical_features

X_test = test_df[features]
y_test = test_df["final_result"]

print("Test records:", len(X_test))
print("Features:", len(features))

# 3. Calculate baseline Macro F1

baseline_predictions = model.predict(X_test)

baseline_f1 = f1_score(
    y_test,
    baseline_predictions,
    average="macro",
    zero_division=0
)

print("\n2. BASELINE MODEL PERFORMANCE")
print("-" * 70)
print(f"Baseline Macro F1: {baseline_f1:.4f}")

# 4. Calculate permutation importance

print("\n3. CALCULATING PERMUTATION IMPORTANCE")
print("-" * 70)

importance = permutation_importance(
    model,
    X_test,
    y_test,
    scoring="f1_macro",
    n_repeats=5,
    random_state=42,
    n_jobs=-1
)

# 5. Create importance report

importance_df = pd.DataFrame({
    "Feature": features,
    "Importance Mean": importance.importances_mean,
    "Importance Std": importance.importances_std
})

importance_df = importance_df.sort_values(
    by="Importance Mean",
    ascending=False
).reset_index(drop=True)

print("\n4. FEATURE IMPORTANCE RANKING")
print("-" * 70)

print(importance_df.round(5).to_string(index=False))

# 6. Save CSV report

report_dir = Path("reports")
report_dir.mkdir(parents=True, exist_ok=True)

csv_path = report_dir / "permutation_feature_importance.csv"

importance_df.to_csv(csv_path, index=False)

# 7. Generate feature importance graph

plt.figure(figsize=(11, 7))

plt.barh(
    importance_df["Feature"],
    importance_df["Importance Mean"],
    xerr=importance_df["Importance Std"],
    capsize=4
)

plt.axvline(
    x=0,
    linestyle="--",
    linewidth=1
)

plt.xlabel("Decrease in Macro F1 Score")
plt.ylabel("Model Features")

plt.title(
    "Permutation Feature Importance\n"
    "Random Forest Student Outcome Prediction"
)

plt.gca().invert_yaxis()

plt.tight_layout()

graph_path = report_dir / "permutation_feature_importance.png"

plt.savefig(graph_path, dpi=300, bbox_inches="tight")

plt.close()

# 8. Save text summary

summary_path = report_dir / "explainability_summary.txt"

with open(summary_path, "w", encoding="utf-8") as file:

    file.write("MODEL EXPLAINABILITY REPORT\n")
    file.write("=" * 60 + "\n\n")

    file.write("Model: Random Forest\n")
    file.write(f"Baseline Macro F1: {baseline_f1:.4f}\n")
    file.write(f"Test records: {len(X_test)}\n\n")

    file.write("PERMUTATION FEATURE IMPORTANCE\n")
    file.write("-" * 60 + "\n")

    file.write(importance_df.to_string(index=False))

    file.write(
        "\n\nInterpretation:\n"
        "Features with larger positive importance values "
        "contribute more to the model's predictive performance.\n"
        "Importance measures predictive contribution, not causation.\n"
        "Negative or near-zero values indicate that shuffling a "
        "feature did not reduce performance in this evaluation.\n"
    )

# 9. Final output

print("\n5. GENERATED FILES")
print("-" * 70)

print(csv_path)
print(graph_path)
print(summary_path)

print("\n" + "=" * 70)
print("MODEL EXPLAINABILITY COMPLETED")
print("=" * 70)
