
import io
import base64

import pandas as pd
import matplotlib
matplotlib.use("Agg") 
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

DATASET_PATH = "dataset.csv"


df = pd.read_csv(DATASET_PATH)

X = df[["temperature_c"]]       
y = df["consumption_kwh"]        

MODEL = LinearRegression()
MODEL.fit(X, y)

_predictions_on_training_data = MODEL.predict(X)
R2_SCORE = round(r2_score(y, _predictions_on_training_data), 4)

DATASET_INFO = {
    "n_records": len(df),
    "independent_variable": "Daily Temperature",
    "independent_unit": "degrees Celsius (°C) average of the day's max and min temperature",
    "dependent_variable": "Electricity Consumption",
    "dependent_unit": "kWh/day",
    "source": "Real data from the Kaggle dataset \"Electricity Consumption Based On "
              "Weather Data\" (sudhirsingh27), "
              "kaggle.com/datasets/sudhirsingh27/electricity-consumption-based-on-weather-data. "
              "Daily temperature was computed as the average of the dataset's TMAX and TMIN columns.",
    "intercept": round(float(MODEL.intercept_), 4),
    "slope": round(float(MODEL.coef_[0]), 4),
    "r2_score": R2_SCORE,
}



def get_plot_base64() -> str:
    
    fig, ax = plt.subplots(figsize=(8, 5), dpi=110)

    ax.scatter(
        df["temperature_c"], df["consumption_kwh"],
        alpha=0.4, s=14, color="#4C72B0", label="Observed data"
    )

   
    x_line = pd.DataFrame(
        {"temperature_c": [df["temperature_c"].min(), df["temperature_c"].max()]}
    )
    y_line = MODEL.predict(x_line)
    ax.plot(
        x_line["temperature_c"], y_line,
        color="#C44E52", linewidth=2.5, label="Regression line"
    )

    ax.set_title("Electricity Consumption vs. Daily Temperature (Linear Regression)", fontsize=12)
    ax.set_xlabel("Daily Temperature (°C)", fontsize=11)
    ax.set_ylabel("Electricity Consumption (kWh/day)", fontsize=11)
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()

    buffer = io.BytesIO()
    fig.savefig(buffer, format="png")
    plt.close(fig)
    buffer.seek(0)

    return base64.b64encode(buffer.read()).decode("utf-8")



def predict_consumption(temperature_c: float) -> float:
   
    prediction = MODEL.predict(pd.DataFrame({"temperature_c": [temperature_c]}))
    return round(float(prediction[0]), 2)


if __name__ == "__main__":
    
    print("Dataset info:", DATASET_INFO)
    print("Prediction for 10 C:", predict_consumption(10), "kWh/day")
    print("Prediction for 25 C:", predict_consumption(25), "kWh/day")
    b64 = get_plot_base64()
    print("Plot base64 length:", len(b64))

