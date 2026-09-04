
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
    "independent_unit": "degrees Celsius (°C) — average of the day's max and min temperature",
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



