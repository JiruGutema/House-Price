import os
import json
from datetime import datetime

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import joblib

# --------------------------------
# 1. Load dataset
# --------------------------------
CSV_PATH = "house.csv"
df = pd.read_csv(CSV_PATH)

print("Shape:", df.shape)
print("Columns:", df.columns.tolist())
print(df.head())

# --------------------------------
# 2. Features & target
# --------------------------------
TARGET_COL = "House_Price"

X = df.drop(columns=[TARGET_COL])
y = df[TARGET_COL]

# --------------------------------
# 3. Train-test split
# --------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, train_size=0.7, random_state=42
)

# --------------------------------
# 4. Pipeline
# --------------------------------
pipeline = Pipeline(steps=[("scaler", StandardScaler()), ("model", LinearRegression())])

pipeline.fit(X_train, y_train)

# --------------------------------
# 5. Evaluation
# --------------------------------
y_pred = pipeline.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\n=== EVALUATION ===")
print("MAE :", mae)
print("RMSE:", rmse)
print("R2  :", r2)

# --------------------------------
# 6. Save model bundle
# --------------------------------
BUNDLE_DIR = "house_price_model_bundle"
os.makedirs(BUNDLE_DIR, exist_ok=True)

MODEL_PATH = os.path.join(BUNDLE_DIR, "model.joblib")
joblib.dump(pipeline, MODEL_PATH)

# --------------------------------
# 7. Save schema
# --------------------------------
schema = {
    "required_columns": X.columns.tolist(),
    "target": TARGET_COL,
    "problem_type": "regression",
}

with open(os.path.join(BUNDLE_DIR, "schema.json"), "w") as f:
    json.dump(schema, f, indent=2)

# --------------------------------
# 8. Save metadata
# --------------------------------
metadata = {
    "created_at": datetime.utcnow().isoformat() + "Z",
    "model": "LinearRegression",
    "metrics": {"mae": float(mae), "rmse": float(rmse), "r2": float(r2)},
}

with open(os.path.join(BUNDLE_DIR, "metadata.json"), "w") as f:
    json.dump(metadata, f, indent=2)

# --------------------------------
# 9. Reload verification
# --------------------------------
reloaded = joblib.load(MODEL_PATH)
y_pred_2 = reloaded.predict(X_test)

if not np.allclose(y_pred, y_pred_2):
    raise RuntimeError("Reload validation FAILED")

print("\n✅ Reload validation PASSED")
