
from flask import Flask, render_template, request, redirect, url_for, flash
from pathlib import Path
from datetime import datetime
import pandas as pd
import joblib
import sqlite3

# ==========================================================
# STUDENT LEARNING SUPPORT AND INTERVENTION TRACKING SYSTEM
# ==========================================================

app = Flask(__name__)
app.secret_key = "student-support-local-prototype-key"

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "models" / "best_student_risk_model.pkl"
TEST_DATA_PATH = BASE_DIR / "data" / "processed" / "ml" / "test_dataset.csv"
IMPORTANCE_PATH = BASE_DIR / "reports" / "permutation_feature_importance.csv"
COMPARISON_PATH = BASE_DIR / "reports" / "model_comparison.csv"

DATABASE_DIR = BASE_DIR / "database"
DATABASE_PATH = DATABASE_DIR / "interventions.db"

DATABASE_DIR.mkdir(parents=True, exist_ok=True)

PER_PAGE = 25

FEATURES = [
    "code_module",
    "code_presentation",
    "cutoff_day",
    "num_of_prev_attempts",
    "studied_credits",
    "early_total_clicks",
    "early_active_days",
    "early_on_time_submissions",
    "early_late_submissions"
]

ALLOWED_INTERVENTIONS = [
    "Academic Counselling",
    "Assignment Support",
    "Study Plan",
    "Faculty Meeting",
    "Additional Learning Resources",
    "Other"
]

ALLOWED_STATUSES = [
    "Pending",
    "In Progress",
    "Completed"
]


# ==========================================================
# 1. LOAD MODEL AND DATASET
# ==========================================================

print("=" * 65)
print("STUDENT LEARNING SUPPORT SYSTEM")
print("STARTING FLASK APPLICATION")
print("=" * 65)

if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

if not TEST_DATA_PATH.exists():
    raise FileNotFoundError(f"Test dataset not found: {TEST_DATA_PATH}")

model = joblib.load(MODEL_PATH)

students_df = pd.read_csv(TEST_DATA_PATH)

missing_columns = [
    column for column in FEATURES + ["id_student"]
    if column not in students_df.columns
]

if missing_columns:
    raise ValueError(f"Missing required columns: {missing_columns}")

students_df = students_df.reset_index(drop=True)

print("\nGenerating student predictions...")

X = students_df[FEATURES]

predictions = model.predict(X)
probabilities = model.predict_proba(X)

class_names = list(model.classes_)

students_df["predicted_outcome"] = predictions

for index, class_name in enumerate(class_names):
    students_df[f"prob_{class_name}"] = probabilities[:, index]

students_df["needs_attention"] = (
    students_df["predicted_outcome"].isin(["Fail", "Withdrawn"])
)

print("Total records:", len(students_df))
print("Predictions generated successfully.")


# ==========================================================
# 2. DATABASE CONNECTION
# ==========================================================

def get_db_connection():
    connection = sqlite3.connect(DATABASE_PATH)
    connection.row_factory = sqlite3.Row
    return connection


def initialize_database():

    with get_db_connection() as connection:

        connection.execute("""
            CREATE TABLE IF NOT EXISTS interventions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                record_id INTEGER NOT NULL,
                student_id TEXT NOT NULL,
                intervention_type TEXT NOT NULL,
                intervention_date TEXT NOT NULL,
                faculty_remarks TEXT,
                follow_up_status TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        connection.commit()


initialize_database()


# ==========================================================
# 3. HELPER FUNCTIONS
# ==========================================================

def get_student_record(record_id):

    if record_id < 0 or record_id >= len(students_df):
        return None

    row = students_df.iloc[record_id]
    record = row.to_dict()

    record["record_id"] = record_id
    record["student_id"] = str(row["id_student"])
    record["needs_attention"] = bool(row["needs_attention"])

    return record


def get_interventions(record_id):

    with get_db_connection() as connection:

        return connection.execute("""
            SELECT *
            FROM interventions
            WHERE record_id = ?
            ORDER BY intervention_date DESC, id DESC
        """, (record_id,)).fetchall()


def get_importance_data():

    if not IMPORTANCE_PATH.exists():
        return []

    df = pd.read_csv(IMPORTANCE_PATH)

    return df.to_dict(orient="records")


def get_paginated_students(search="", selected_outcome="All", page=1):

    filtered_df = students_df.copy()

    if search:
        filtered_df = filtered_df[
            filtered_df["id_student"]
            .astype(str)
            .str.contains(search, regex=False)
        ]

    if selected_outcome != "All":
        filtered_df = filtered_df[
            filtered_df["predicted_outcome"] == selected_outcome
        ]

    total_filtered = len(filtered_df)

    total_pages = max(
        1,
        (total_filtered + PER_PAGE - 1) // PER_PAGE
    )

    page = max(1, min(page, total_pages))

    start = (page - 1) * PER_PAGE

    page_df = filtered_df.iloc[start:start + PER_PAGE]

    records = []

    for record_id, row in page_df.iterrows():

        probability_values = [
            float(row[f"prob_{name}"]) * 100
            for name in class_names
        ]

        records.append({
            "record_id": int(record_id),
            "student_id": str(row["id_student"]),
            "module": row["code_module"],
            "presentation": row["code_presentation"],
            "predicted_outcome": row["predicted_outcome"],
            "needs_attention": bool(row["needs_attention"]),
            "confidence": round(max(probability_values), 2)
        })

    return records, total_filtered, total_pages, page


def get_filter_parameters():

    search = request.args.get("search", "").strip()

    selected_outcome = request.args.get("outcome", "All")

    if selected_outcome not in ["All"] + class_names:
        selected_outcome = "All"

    page = request.args.get("page", 1, type=int)
    page = max(1, page)

    return search, selected_outcome, page


# ==========================================================
# 4. FACULTY DASHBOARD
# ==========================================================

@app.route("/")
def dashboard():

    outcome_counts = (
        students_df["predicted_outcome"]
        .value_counts()
        .reindex(class_names, fill_value=0)
        .to_dict()
    )

    total_students = len(students_df)

    attention_count = int(
        students_df["needs_attention"].sum()
    )

    with get_db_connection() as connection:

        intervention_count = connection.execute(
            "SELECT COUNT(*) FROM interventions"
        ).fetchone()[0]

        recent_interventions = connection.execute("""
            SELECT *
            FROM interventions
            ORDER BY created_at DESC, id DESC
            LIMIT 5
        """).fetchall()

    search, selected_outcome, page = get_filter_parameters()

    records, total_filtered, total_pages, page = get_paginated_students(
        search, selected_outcome, page
    )

    return render_template(
        "dashboard.html",
        total_students=total_students,
        outcome_counts=outcome_counts,
        attention_count=attention_count,
        intervention_count=intervention_count,
        students=records,
        search=search,
        selected_outcome=selected_outcome,
        page=page,
        total_pages=total_pages,
        total_filtered=total_filtered,
        class_names=class_names,
        importance_data=get_importance_data(),
        recent_interventions=recent_interventions,
        active_page="dashboard"
    )


# ==========================================================
# 5. STUDENTS PAGE
# ==========================================================

@app.route("/students")
def students_page():

    search, selected_outcome, page = get_filter_parameters()

    records, total_filtered, total_pages, page = get_paginated_students(
        search, selected_outcome, page
    )

    return render_template(
        "students.html",
        students=records,
        search=search,
        selected_outcome=selected_outcome,
        class_names=class_names,
        page=page,
        total_pages=total_pages,
        total_filtered=total_filtered,
        active_page="students"
    )


# ==========================================================
# 6. INDIVIDUAL STUDENT DETAILS
# ==========================================================

@app.route("/student/<int:record_id>")
def student_detail(record_id):

    student = get_student_record(record_id)

    if student is None:
        return "Student record not found", 404

    probabilities_dict = {}

    for class_name in class_names:

        probabilities_dict[class_name] = round(
            float(student[f"prob_{class_name}"]) * 100,
            2
        )

    interventions = get_interventions(record_id)

    return render_template(
        "student.html",
        student=student,
        probabilities=probabilities_dict,
        interventions=interventions,
        active_page="students"
    )


# ==========================================================
# 7. SAVE FACULTY INTERVENTION
# ==========================================================

@app.route(
    "/student/<int:record_id>/intervention",
    methods=["POST"]
)
def add_intervention(record_id):

    student = get_student_record(record_id)

    if student is None:
        return "Student record not found", 404

    intervention_type = request.form.get(
        "intervention_type", ""
    ).strip()

    intervention_date = request.form.get(
        "intervention_date", ""
    ).strip()

    remarks = request.form.get(
        "faculty_remarks", ""
    ).strip()

    follow_up_status = request.form.get(
        "follow_up_status", ""
    ).strip()

    if intervention_type not in ALLOWED_INTERVENTIONS:
        flash("Please select a valid intervention type.", "error")
        return redirect(url_for("student_detail", record_id=record_id))

    if follow_up_status not in ALLOWED_STATUSES:
        flash("Please select a valid follow-up status.", "error")
        return redirect(url_for("student_detail", record_id=record_id))

    if not intervention_date or not remarks:
        flash("Please complete all required fields.", "error")
        return redirect(url_for("student_detail", record_id=record_id))

    try:
        datetime.strptime(intervention_date, "%Y-%m-%d")
    except ValueError:
        flash("Invalid intervention date.", "error")
        return redirect(url_for("student_detail", record_id=record_id))

    with get_db_connection() as connection:

        connection.execute("""
            INSERT INTO interventions (
                record_id,
                student_id,
                intervention_type,
                intervention_date,
                faculty_remarks,
                follow_up_status
            )
            VALUES (?, ?, ?, ?, ?, ?)
        """, (
            record_id,
            student["student_id"],
            intervention_type,
            intervention_date,
            remarks,
            follow_up_status
        ))

        connection.commit()

    flash("Intervention recorded successfully.", "success")

    return redirect(url_for("student_detail", record_id=record_id))


# ==========================================================
# 8. MODEL INSIGHTS
# ==========================================================

@app.route("/model-insights")
def model_insights():

    model_comparison = []

    if COMPARISON_PATH.exists():

        comparison_df = pd.read_csv(COMPARISON_PATH)

        model_comparison = comparison_df.to_dict(
            orient="records"
        )

    return render_template(
        "model_insights.html",
        model_comparison=model_comparison,
        importance_data=get_importance_data(),
        active_page="insights"
    )


# ==========================================================
# 9. INTERVENTIONS
# ==========================================================

@app.route("/interventions")
def interventions_page():

    with get_db_connection() as connection:

        interventions = connection.execute("""
            SELECT *
            FROM interventions
            ORDER BY created_at DESC, id DESC
        """).fetchall()

        pending_count = connection.execute("""
            SELECT COUNT(*)
            FROM interventions
            WHERE follow_up_status = 'Pending'
        """).fetchone()[0]

        in_progress_count = connection.execute("""
            SELECT COUNT(*)
            FROM interventions
            WHERE follow_up_status = 'In Progress'
        """).fetchone()[0]

        completed_count = connection.execute("""
            SELECT COUNT(*)
            FROM interventions
            WHERE follow_up_status = 'Completed'
        """).fetchone()[0]

    return render_template(
        "interventions.html",
        interventions=interventions,
        total_interventions=len(interventions),
        pending_count=pending_count,
        in_progress_count=in_progress_count,
        completed_count=completed_count,
        active_page="interventions"
    )


# ==========================================================
# 10. RUN APPLICATION
# ==========================================================

if __name__ == "__main__":

    print("\n" + "=" * 65)
    print("FLASK APPLICATION READY")
    print("=" * 65)

    print("Dashboard: http://127.0.0.1:5000")
    print("Students: http://127.0.0.1:5000/students")
    print("Model Insights: http://127.0.0.1:5000/model-insights")
    print("Interventions: http://127.0.0.1:5000/interventions")
    print("Database:", DATABASE_PATH)

    app.run(
        host="127.0.0.1",
        port=5000,
        debug=True
    )
