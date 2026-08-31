"""
Data with Roots - Machine Learning Project
Flask App - Universidad de Cundinamarca - Systems and Computing Engineering
"""

from flask import Flask, render_template

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


@app.route("/linear-regression/application")
def lr_application():
    return render_template("lr_application.html")


if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)