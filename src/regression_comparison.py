"""
Regression Model Comparison
============================
Compares 8 regression algorithms — Linear Regression, KNN, Decision Tree,
SVM (SVR), Random Forest, Gradient Boosting, XGBoost, and LightGBM — on the
California Housing dataset, then tunes the top 3 performers with GridSearchCV.

Run:
    python regression_model_comparison.py
"""

import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

# =========================================================
# 1. Load and inspect the data
# =========================================================
data = fetch_california_housing(as_frame=True)
df = data.frame
print(df.head())
print(df.describe())
print("Missing values:", df.isnull().sum().sum())

# =========================================================
# 2. Train/test split
# =========================================================
X = df.drop(columns="MedHouseVal")
y = df["MedHouseVal"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# =========================================================
# 3. Define the models
# =========================================================
# Distance-based models (KNN, SVR) are wrapped in a Pipeline with a scaler
models = {
    "Linear Regression": LinearRegression(),

    "KNN": Pipeline([
        ("scaler", StandardScaler()),
        ("model", KNeighborsRegressor()),
    ]),

    "Decision Tree": DecisionTreeRegressor(random_state=42),

    "SVM (SVR)": Pipeline([
        ("scaler", StandardScaler()),
        ("model", SVR()),
    ]),

    "Random Forest": RandomForestRegressor(random_state=42, n_jobs=-1),

    "Gradient Boosting": GradientBoostingRegressor(random_state=42),

    "XGBoost": XGBRegressor(random_state=42, n_jobs=-1),

    "LightGBM": LGBMRegressor(random_state=42, verbose=-1),
}

# =========================================================
# 4. Train, evaluate, and time every model
# =========================================================
def evaluate_model(model, name):
    t0 = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - t0

    t1 = time.time()
    pred = model.predict(X_test)
    predict_time = time.time() - t1

    rmse = np.sqrt(mean_squared_error(y_test, pred))
    r2 = r2_score(y_test, pred)

    return {
        "Model": name,
        "RMSE": round(rmse, 4),
        "R2": round(r2, 4),
        "Train time (s)": round(train_time, 3),
        "Predict time (s)": round(predict_time, 4),
    }

results = []
for name, model in models.items():
    print(f"Training {name} ...")
    res = evaluate_model(model, name)
    results.append(res)
    print(f"  -> RMSE={res['RMSE']}, R2={res['R2']}, "
          f"train={res['Train time (s)']}s, predict={res['Predict time (s)']}s")

results_df = pd.DataFrame(results).sort_values("RMSE").reset_index(drop=True)
print("\n===== COMPARISON TABLE (default parameters) =====")
print(results_df)

# =========================================================
# 5. Plots
# =========================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].barh(results_df["Model"], results_df["R2"], color="steelblue")
axes[0].set_xlabel("R2 Score")
axes[0].set_title("Model Accuracy (R2) - higher is better")
axes[0].invert_yaxis()

axes[1].barh(results_df["Model"], results_df["Train time (s)"], color="salmon")
axes[1].set_xlabel("Train time (s)")
axes[1].set_title("Training Time - lower is better")
axes[1].invert_yaxis()

plt.tight_layout()
plt.show()

# =========================================================
# 6. Tune the top 3 models
# =========================================================
top3 = results_df.head(3)["Model"].tolist()
print("\nTop 3 models selected for tuning:", top3)

param_grids = {
    "Random Forest": {
        "n_estimators": [200, 400],
        "max_depth": [None, 10, 20],
        "min_samples_leaf": [1, 2],
    },
    "Gradient Boosting": {
        "n_estimators": [200, 300],
        "learning_rate": [0.05, 0.1],
        "max_depth": [3, 5],
    },
    "XGBoost": {
        "n_estimators": [200, 300],
        "learning_rate": [0.05, 0.1],
        "max_depth": [4, 6],
        "subsample": [0.8, 1.0],
    },
    "LightGBM": {
        "n_estimators": [200, 300],
        "learning_rate": [0.05, 0.1],
        "num_leaves": [31, 63],
    },
    "KNN": {
        "model__n_neighbors": [5, 10, 15],
        "model__weights": ["uniform", "distance"],
    },
    "SVM (SVR)": {
        "model__C": [1, 10],
        "model__epsilon": [0.1, 0.2],
    },
    "Decision Tree": {
        "max_depth": [5, 10, 15, None],
        "min_samples_leaf": [1, 5, 10],
    },
    "Linear Regression": {},  # nothing to tune
}

base_models = {
    "Random Forest": RandomForestRegressor(random_state=42, n_jobs=-1),
    "Gradient Boosting": GradientBoostingRegressor(random_state=42),
    "XGBoost": XGBRegressor(random_state=42, n_jobs=-1),
    "LightGBM": LGBMRegressor(random_state=42, verbose=-1),
    "KNN": Pipeline([("scaler", StandardScaler()), ("model", KNeighborsRegressor())]),
    "SVM (SVR)": Pipeline([("scaler", StandardScaler()), ("model", SVR())]),
    "Decision Tree": DecisionTreeRegressor(random_state=42),
    "Linear Regression": LinearRegression(),
}

tuned_results = []
for name in top3:
    grid_params = param_grids.get(name, {})
    if not grid_params:
        # No hyperparameters to tune — just re-evaluate as-is
        res = evaluate_model(base_models[name], name + " (tuned)")
        tuned_results.append(res)
        continue

    print(f"\nTuning {name} ...")
    t0 = time.time()
    grid = GridSearchCV(
        base_models[name], grid_params, cv=3,
        scoring="neg_root_mean_squared_error", n_jobs=-1
    )
    grid.fit(X_train, y_train)
    train_time = time.time() - t0

    best_model = grid.best_estimator_
    pred = best_model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, pred))
    r2 = r2_score(y_test, pred)

    print(f"  Best params: {grid.best_params_}")
    tuned_results.append({
        "Model": name + " (tuned)",
        "RMSE": round(rmse, 4),
        "R2": round(r2, 4),
        "Train time (s)": round(train_time, 3),
        "Predict time (s)": None,
    })

tuned_df = pd.DataFrame(tuned_results)
print("\n===== TUNED MODELS RESULTS =====")
print(tuned_df)

# =========================================================
# 7. Final table: default + tuned
# =========================================================
final_comparison = pd.concat([results_df, tuned_df], ignore_index=True)
final_comparison = final_comparison.sort_values("RMSE").reset_index(drop=True)
print("\n===== FINAL COMPARISON TABLE =====")
print(final_comparison)
