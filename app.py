"""
Machine Learning Project - Flask App (base structure)
Universidad de Cundinamarca - Systems and Computing Engineering

"""

from flask import Flask, render_template

app = Flask(__name__)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/machine-learning")
def machine_learning():
    return render_template("machine-learning.html")


@app.route("/types")
def types():
    return render_template("types.html")


if __name__ == "__main__":
    app.run(debug=True)