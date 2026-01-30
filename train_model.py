import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------
DATA_PATH = "feature_engineered_us_cities.csv"
RANDOM_STATE = 42
TEST_SIZE = 0.25

# --------------------------------------------------
# STEP 1: LOAD DATA
# --------------------------------------------------
df = pd.read_csv(DATA_PATH)

print("Dataset loaded")
print("Shape:", df.shape)

# --------------------------------------------------
# STEP 2: SPLIT FEATURES & TARGETS
# --------------------------------------------------
X = df[[c for c in df.columns if c.startswith("delta_")]]
y = df[["overall_cost_diff", "max_cost_diff"]]

print("Features shape:", X.shape)
print("Targets shape:", y.shape)

# --------------------------------------------------
# STEP 3: TRAIN–TEST SPLIT
# --------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE
)

# --------------------------------------------------
# STEP 4: MODEL DEFINITION
# --------------------------------------------------
base_model = GradientBoostingRegressor(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=3,
    subsample=0.9,
    random_state=RANDOM_STATE
)

model = MultiOutputRegressor(base_model)

# --------------------------------------------------
# STEP 5: TRAIN MODEL
# --------------------------------------------------
model.fit(X_train, y_train)

print("Model training completed")

# --------------------------------------------------
# STEP 6: EVALUATION
# --------------------------------------------------
preds = model.predict(X_test)

mae_overall = mean_absolute_error(y_test["overall_cost_diff"], preds[:, 0])
mae_max = mean_absolute_error(y_test["max_cost_diff"], preds[:, 1])

rmse_overall = mean_squared_error(
    y_test["overall_cost_diff"], preds[:, 0], squared=False
)
rmse_max = mean_squared_error(
    y_test["max_cost_diff"], preds[:, 1], squared=False
)

print("\nEvaluation Metrics")
print("-------------------")
print(f"MAE (Overall Cost Difference): {mae_overall:.2f}")
print(f"MAE (Max Category Difference): {mae_max:.2f}")
print(f"RMSE (Overall Cost Difference): {rmse_overall:.2f}")
print(f"RMSE (Max Category Difference): {rmse_max:.2f}")

# --------------------------------------------------
# STEP 7: FEATURE IMPORTANCE (PER OUTPUT)
# --------------------------------------------------
for i, target in enumerate(y.columns):
    importances = model.estimators_[i].feature_importances_
    importance_df = pd.DataFrame({
        "feature": X.columns,
        "importance": importances
    }).sort_values(by="importance", ascending=False)

    print(f"\nTop 10 features for target: {target}")
    print(importance_df.head(10))
