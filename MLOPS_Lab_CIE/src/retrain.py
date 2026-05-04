import pandas as pd
import json
import joblib
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import RandomForestRegressor

train_df = pd.read_csv("data/training_data.csv")
new_df = pd.read_csv("data/new_data.csv")

combined = pd.concat([train_df, new_df])

X = combined.drop("shipping_days", axis=1)
y = combined["shipping_days"]

model = RandomForestRegressor(n_estimators=50, random_state=42)
model.fit(X, y)

preds = model.predict(X)
mae = mean_absolute_error(y, preds)

champion_model = joblib.load("models/best_model.pkl")
champ_preds = champion_model.predict(X)
champ_mae = mean_absolute_error(y, champ_preds)

improvement = champ_mae - mae

action = "promoted" if improvement > 0 else "kept_champion"

if action == "promoted":
    joblib.dump(model, "models/best_model.pkl")

with open("results/step4_s8.json", "w") as f:
    json.dump({
        "original_data_rows": len(train_df),
        "new_data_rows": len(new_df),
        "combined_data_rows": len(combined),
        "champion_mae": champ_mae,
        "retrained_mae": mae,
        "improvement": improvement,
        "min_improvement_threshold": 0,
        "action": action
    }, f, indent=4)