import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import json
import os
import joblib

# Load data
df = pd.read_csv("data/training_data.csv")

X = df.drop("shipping_days", axis=1)
y = df["shipping_days"]

models = {
    "LinearRegression": LinearRegression(),
    "RandomForest": RandomForestRegressor(n_estimators=50, random_state=42)
}

results = []
best_rmse = float("inf")
best_model = None
best_name = ""

mlflow.set_experiment("speedcargo-shipping-days")

for name, model in models.items():
    with mlflow.start_run(run_name=name):

        model.fit(X, y)
        preds = model.predict(X)

        mae = mean_absolute_error(y, preds)
        rmse = mean_squared_error(y, preds) ** 0.5

        mlflow.log_param("model", name)
        mlflow.log_metric("MAE", mae)
        mlflow.log_metric("RMSE", rmse)
        mlflow.set_tag("team", "logistics")

        mlflow.sklearn.log_model(model, "model")

        results.append({
            "name": name,
            "mae": mae,
            "rmse": rmse
        })

        if rmse < best_rmse:
            best_rmse = rmse
            best_model = model
            best_name = name

# Save best model
os.makedirs("models", exist_ok=True)
joblib.dump(best_model, "models/best_model.pkl")

# Save results JSON
output = {
    "experiment_name": "speedcargo-shipping-days",
    "models": results,
    "best_model": best_name,
    "best_metric_name": "rmse",
    "best_metric_value": best_rmse
}

os.makedirs("results", exist_ok=True)
with open("results/step1_s1.json", "w") as f:
    json.dump(output, f, indent=4)

print("Training complete!")

import mlflow.pyfunc

with mlflow.start_run() as run:
    mlflow.sklearn.log_model(best_model, "model")

    result = mlflow.register_model(
        f"runs:/{run.info.run_id}/model",
        "speedcargo-shipping-days-predictor"
    )

    with open("results/step3_s6.json", "w") as f:
        json.dump({
            "registered_model_name": "speedcargo-shipping-days-predictor",
            "version": result.version,
            "run_id": run.info.run_id,
            "source_metric": "rmse",
            "source_metric_value": best_rmse
        }, f, indent=4)