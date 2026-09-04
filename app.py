"""
Data with Roots - Machine Learning Project
Flask App - Universidad de Cundinamarca - Systems and Computing Engineering
"""

from flask import Flask, render_template, request
from Lr_model import DATASET_INFO, get_plot_base64, predict_consumption

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


# ---------- Supervised: Linear Regression (Jonathan) ----------
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


if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)