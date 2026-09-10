"""
Data with Roots - Machine Learning Project
Flask App - Universidad de Cundinamarca - Systems and Computing Engineering
"""

from flask import Flask, render_template, request
from Lr_model import DATASET_INFO, get_plot_base64, predict_consumption
from LogReg_model import (
    DATASET_INFO as LOGREG_INFO,
    get_plot_base64 as logreg_plot,
    get_confusion_matrix_plot_base64 as logreg_cm_plot,
    predict_risk,
    EVAL_METRICS as LOGREG_METRICS,
)
from ExtraTrees_model import (
    DATASET_INFO as ET_INFO,
    get_plot_base64 as et_plot,
    get_confusion_matrix_plot_base64 as et_cm_plot,
    predict_class as et_predict,
    EVAL_METRICS as ET_METRICS,
)

app = Flask(__name__)


# ---------- Home ----------
@app.route("/")
def home():
    return render_template("index.html")


# ---------- Machine Learning: Concepts & Types (Sergio) ----------
@app.route("/concepts")
def concepts():
    return render_template("concepts.html")


@app.route("/types")
def types():
    return render_template("types.html")


# ---------- Use Cases (Manuel) ----------
@app.route("/use-case-1")
def use_case_1():
    return render_template("use_case_1.html")


@app.route("/use-case-2")
def use_case_2():
    return render_template("use_case_2.html")


@app.route("/use-case-3")
def use_case_3():
    return render_template("use_case_3.html")


@app.route("/use-case-4")
def use_case_4():
    return render_template("use_case_4.html")


# ---------- Supervised: Linear Regression (Jonathan, Activity 1) ----------
@app.route("/linear-regression/concepts")
def lr_concepts():
    return render_template("lr_concepts.html")


@app.route("/linear-regression/application", methods=["GET", "POST"])
def lr_application():
    prediction = None
    submitted_value = None
    error = None

    if request.method == "POST":
        raw_value = request.form.get("temperature", "").strip()
        submitted_value = raw_value

        if raw_value == "":
            error = "Please enter a temperature value."
        else:
            try:
                temperature_c = float(raw_value)
                prediction = predict_consumption(temperature_c)
                submitted_value = temperature_c
            except ValueError:
                error = "Please enter a valid number (e.g. 5 or -3.5)."

    return render_template(
        "lr_application.html",
        dataset_info=DATASET_INFO,
        plot_url=get_plot_base64(),
        prediction=prediction,
        submitted_value=submitted_value,
        error=error,
    )


# ---------- Supervised: Logistic Regression (Manuel, Activity 2) ----------
@app.route("/logistic-regression/concepts")
def logreg_concepts():
    return render_template("logreg_concepts.html")


@app.route("/logistic-regression/application", methods=["GET", "POST"])
def logreg_application():
    prediction = None
    submitted_value = None
    error = None

    if request.method == "POST":
        raw_value = request.form.get("cholesterol", "").strip()
        submitted_value = raw_value

        if raw_value == "":
            error = "Please enter a cholesterol value."
        else:
            try:
                cholesterol = float(raw_value)
                prediction = predict_risk(cholesterol)
                submitted_value = cholesterol
            except ValueError:
                error = "Please enter a valid number."

    return render_template(
        "logreg_application.html",
        dataset_info=LOGREG_INFO,
        plot_url=logreg_plot(),
        prediction=prediction,
        submitted_value=submitted_value,
        error=error,
    )


@app.route("/logistic-regression/evaluation-metrics")
def logreg_evaluation():
    return render_template(
        "logreg_evaluation.html",
        metrics=LOGREG_METRICS,
        confusion_plot=logreg_cm_plot(),
    )


# ---------- Supervised: Extra Trees Classifier (Jonathan, Activity 2) ----------
@app.route("/extra-trees/concepts")
def et_concepts():
    return render_template("et_concepts.html")


@app.route("/extra-trees/application", methods=["GET", "POST"])
def et_application():
    result = None
    form_values = {}
    error = None

    if request.method == "POST":
        try:
            chol = float(request.form.get("chol", "").strip())
            age = float(request.form.get("age", "").strip())
            trestbps = float(request.form.get("trestbps", "").strip())
            thalach = float(request.form.get("thalach", "").strip())
            form_values = {
                "chol": chol,
                "age": age,
                "trestbps": trestbps,
                "thalach": thalach,
            }
            result = et_predict(chol, age, trestbps, thalach)
        except (ValueError, TypeError):
            error = "Please enter valid numeric values for all fields."

    return render_template(
        "et_application.html",
        dataset_info=ET_INFO,
        plot_url=et_plot(),
        result=result,
        form_values=form_values,
        error=error,
    )


@app.route("/extra-trees/evaluation-metrics")
def et_evaluation():
    return render_template(
        "et_evaluation.html",
        metrics=ET_METRICS,
        confusion_plot=et_cm_plot(),
    )


if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)