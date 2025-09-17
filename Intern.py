# PREDICTION OF AGRICULTURE CROP PRODUCTION IN INDIA

# Importing libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

# ===========================
# 1. Load Dataset
# ===========================
data = pd.read_csv(r"C:\Users\iniya\Downloads\sample_crop_production.csv")


print("Dataset Shape:", data.shape)
print(data.head())

# ===========================
# 2. Data Preprocessing
# ===========================
data = data.dropna()


le = LabelEncoder()
data["Crop"] = le.fit_transform(data["Crop"])
data["State"] = le.fit_transform(data["State"])
data["Season"] = le.fit_transform(data["Season"])

X = data[["Crop", "State", "Season", "Area", "Rainfall"]]
y = data["Production"]

# ===========================
# 3. Exploratory Data Analysis (EDA)
# ===========================
plt.figure(figsize=(8,5))
sns.heatmap(X.corr(), annot=True, cmap="YlGnBu")
plt.title("Feature Correlation Heatmap")
plt.show()

plt.figure(figsize=(8,5))
sns.histplot(y, bins=30, kde=True)
plt.title("Distribution of Crop Production")
plt.show()

# ===========================
# 4. Train-Test Split
# ===========================
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ===========================
# 5. Model Training
# ===========================
models = {
    "Linear Regression": LinearRegression(),
    "Decision Tree": DecisionTreeRegressor(random_state=42),
    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
    "XGBoost": XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
}

results = {}

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    results[name] = {"MAE": mae, "RMSE": rmse, "R2 Score": r2}

# ===========================
# 6. Results Comparison
# ===========================
results_df = pd.DataFrame(results).T
print("\nModel Performance Comparison:\n")
print(results_df)

# Plot comparison
results_df["R2 Score"].plot(kind="bar", figsize=(8,5), title="Model R² Score Comparison", color="skyblue")
plt.ylabel("R² Score")
plt.show()

# ===========================
# 7. Save Best Model
# ===========================
from joblib import dump

best_model = RandomForestRegressor(n_estimators=100, random_state=42)
best_model.fit(X_train, y_train)

# Save trained model
dump(best_model, "crop_prediction_model.joblib")
print("Best model saved successfully!")
