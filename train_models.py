
import pandas as pd
import numpy as np
import joblib

from pathlib import Path
from time import time

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

# ============================================================
# TRAIN AND COMPARE THREE MACHINE LEARNING MODELS
# ============================================================

print("=" * 70)
print("STUDENT LEARNING SUPPORT SYSTEM")
print("MACHINE LEARNING MODEL TRAINING AND COMPARISON")
print("=" * 70)

# 1. Load training and testing datasets

data_dir = Path("data/processed/ml")

train_df = pd.read_csv(data_dir / "train_dataset.csv")
test_df = pd.read_csv(data_dir / "test_dataset.csv")

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

features = categorical_features + numerical_features

X_train = train_df[features]
y_train = train_df[target]

X_test = test_df[features]
y_test = test_df[target]

class_order = [
    "Pass",
    "Fail",
    "Distinction",
    "Withdrawn"
]

print("\n1. DATA LOADED")
print("-" * 70)
print("Training records:", len(X_train))
print("Testing records:", len(X_test))
print("Features:", len(features))

# 2. Create preprocessing pipeline

def create_preprocessor():

    return ColumnTransformer(
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

# 3. Define models

models = {
    "Logistic Regression": LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        random_state=42
    ),

    "Decision Tree": DecisionTreeClassifier(
        class_weight="balanced",
        random_state=42,
        max_depth=15,
        min_samples_leaf=5
    ),

    "Random Forest": RandomForestClassifier(
        n_estimators=200,
        class_weight="balanced_subsample",
        random_state=42,
        n_jobs=-1,
        min_samples_leaf=2
    )
}

# 4. Train and evaluate models

results = []
trained_models = {}

for model_name, classifier in models.items():

    print("\n" + "=" * 70)
    print("TRAINING:", model_name)
    print("=" * 70)

    start_time = time()

    pipeline = Pipeline(
        steps=[
            ("preprocessor", create_preprocessor()),
            ("classifier", classifier)
        ]
    )

    pipeline.fit(X_train, y_train)

    predictions = pipeline.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)

    precision = precision_score(
        y_test,
        predictions,
        average="macro",
        zero_division=0
    )

    recall = recall_score(
        y_test,
        predictions,
        average="macro",
        zero_division=0
    )

    macro_f1 = f1_score(
        y_test,
        predictions,
        average="macro",
        zero_division=0
    )

    weighted_f1 = f1_score(
        y_test,
        predictions,
        average="weighted",
        zero_division=0
    )

    elapsed = time() - start_time

    results.append({
        "Model": model_name,
        "Accuracy": accuracy,
        "Macro Precision": precision,
        "Macro Recall": recall,
        "Macro F1": macro_f1,
        "Weighted F1": weighted_f1,
        "Training Time (seconds)": elapsed
    })

    trained_models[model_name] = pipeline

    print(f"Accuracy:       {accuracy:.4f}")
    print(f"Macro Precision:{precision:.4f}")
    print(f"Macro Recall:   {recall:.4f}")
    print(f"Macro F1:       {macro_f1:.4f}")
    print(f"Weighted F1:    {weighted_f1:.4f}")
    print(f"Time:           {elapsed:.2f} seconds")

# 5. Compare all models

print("\n" + "=" * 70)
print("MODEL PERFORMANCE COMPARISON")
print("=" * 70)

results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    by="Macro F1",
    ascending=False
).reset_index(drop=True)

print(
    results_df.round(4).to_string(index=False)
)

# 6. Select best model using macro F1

best_model_name = results_df.iloc[0]["Model"]
best_model = trained_models[best_model_name]

print("\n" + "=" * 70)
print("BEST MODEL:", best_model_name)
print("=" * 70)

# 7. Detailed classification report

best_predictions = best_model.predict(X_test)

print("\nCLASSIFICATION REPORT")
print("-" * 70)

print(
    classification_report(
        y_test,
        best_predictions,
        labels=class_order,
        zero_division=0
    )
)

# 8. Confusion matrix

print("\nCONFUSION MATRIX")
print("-" * 70)

cm = confusion_matrix(
    y_test,
    best_predictions,
    labels=class_order
)

cm_df = pd.DataFrame(
    cm,
    index=class_order,
    columns=class_order
)

print(cm_df)

# 9. Save model and results

output_dir = Path("models")
output_dir.mkdir(parents=True, exist_ok=True)

joblib.dump(
    best_model,
    output_dir / "best_student_risk_model.pkl"
)

results_path = Path("reports/model_comparison.csv")
results_path.parent.mkdir(parents=True, exist_ok=True)

results_df.to_csv(results_path, index=False)

report_path = Path("reports/classification_report.txt")

with open(report_path, "w", encoding="utf-8") as file:

    file.write("BEST MODEL: " + best_model_name + "\n\n")

    file.write("MODEL COMPARISON\n")
    file.write(results_df.to_string(index=False))

    file.write("\n\nCLASSIFICATION REPORT\n")
    file.write(
        classification_report(
            y_test,
            best_predictions,
            labels=class_order,
            zero_division=0
        )
    )

    file.write("\nCONFUSION MATRIX\n")
    file.write(cm_df.to_string())

print("\n" + "=" * 70)
print("MODEL TRAINING COMPLETED")
print("=" * 70)

print("Best model:", best_model_name)
print("Saved model: models/best_student_risk_model.pkl")
print("Comparison: reports/model_comparison.csv")
print("Evaluation: reports/classification_report.txt")
