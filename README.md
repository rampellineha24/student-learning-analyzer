README.md
# Student Learning Behaviour Analyzer

## Project Overview

The **Student Learning Behaviour Analyzer** is a web-based learning analytics system developed to analyse student learning activities and identify patterns associated with academic outcomes.

The system uses the Open University Learning Analytics Dataset (OULAD) and machine learning classification techniques to predict student academic outcomes. It provides faculty-oriented dashboards, student records, individual prediction details, model insights, and intervention tracking.

## Objectives

- Analyse student learning activity and assessment-related data.
- Identify patterns associated with academic outcomes.
- Predict student academic outcomes using machine learning.
- Present useful information through a faculty-oriented web dashboard.
- Record faculty interventions and monitor their follow-up status.

## Main Features

- **Faculty Dashboard:** Displays an overview of student records and academic outcome information.
- **Student Records:** Allows faculty to search and filter student records by predicted or recorded outcome.
- **Individual Student Details:** Displays student-level information and prediction probabilities.
- **Model Insights:** Presents machine learning model evaluation results and feature-importance information.
- **Intervention Tracking:** Allows faculty to record interventions, remarks, dates, and follow-up status.
- **Intervention History:** Displays previously recorded intervention details.

## Technologies Used

- Python
- Flask
- Pandas and NumPy
- Scikit-learn
- Joblib
- SQLite
- HTML, CSS and JavaScript
- Jinja templates
- Matplotlib and Seaborn

## Dataset

This project uses the **Open University Learning Analytics Dataset (OULAD)**.

Dataset source: https://analyse.kmi.open.ac.uk/open_dataset

The dataset contains information related to courses, students, assessments, registrations, and virtual learning environment activity.

## Machine Learning

The project compares the following classification algorithms:

1. Logistic Regression
2. Decision Tree
3. Random Forest

The models are evaluated using accuracy, macro precision, macro recall, macro F1-score, and weighted F1-score.

Random Forest achieved the best macro F1-score among the three evaluated models and was selected as the final model.

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/rampellineha24/student-learning-analyzer.git
```

### 2. Open the project directory

```bash
cd student-learning-analyzer
```

### 3. Create a virtual environment

```bash
python -m venv venv
```

### 4. Install the required packages

```bash
python -m pip install -r requirements.txt
```

## Execution

Run the Flask application from the project directory:

```bash
python app.py
```

Open the application in a web browser:

http://127.0.0.1:5000

## Application Pages

| Page | Route |
|---|---|
| Faculty Dashboard | `/` |
| Student Records | `/students` |
| Model Insights | `/model-insights` |
| Intervention Tracking | `/interventions` |

## Project Structure

```text
student-learning-analyzer/
│
├── app.py
├── requirements.txt
├── README.md
│
├── data/
│   ├── raw/
│   └── processed/
│
├── models/
│   └── best_student_risk_model.pkl
│
├── database/
│   └── db.py
│
├── reports/
│
├── templates/
│
└── static/
    └── css/
```

## Limitations

- The system is an academic prototype developed using historical OULAD data.
- Predictions are intended to support faculty analysis and should not be treated as definitive judgments about individual students.
- The system does not use live institutional LMS data.
- Intervention tracking records faculty actions and follow-up status; it does not establish that an intervention caused an improvement in academic performance.

## Project Team

- Suman Deep Prasad
- Rampelli Neha
- Embari Yashaswini
- D S Harshad Sai

**Project Guide:** Dr. Sudeshna Sani

**Programme:** B.Tech 

**Institution:** Faculty of Science and Technology, ICFAI Foundation for Higher Education, Hyderabad

**Academic Year:** 2026–2027