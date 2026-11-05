import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
# 1. Dataset folder
dataset = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dataset")
# 2. Load datasets
sales = pd.read_csv(os.path.join(dataset, "sales.csv"))
features = pd.read_csv(os.path.join(dataset, "features.csv"))
stores = pd.read_csv(os.path.join(dataset, "stores.csv"))
print("Datasets loaded successfully")
# 3. Clean column names
for df in [sales, features, stores]:
    df.columns = df.columns.str.strip().str.lower()
print("Sales columns:", sales.columns.tolist())
print("Features columns:", features.columns.tolist())
print("Stores columns:", stores.columns.tolist())
# 4. Find matching column names
def find_column(df, possible_names):
    for col in df.columns:
        if col in possible_names:
            return col
    return None
sales_store = find_column(sales, ["store", "store_id", "store id"])
features_store = find_column(features, ["store", "store_id", "store id"])
stores_store = find_column(stores, ["store", "store_id", "store id"])
sales_date = find_column(sales, ["date"])
features_date = find_column(features, ["date"])
target = find_column(sales, ["weekly_sales", "weekly sales", "sales"])
# 5. Stop with a clear message if columns are missing
if not all([sales_store, features_store, stores_store,
            sales_date, features_date, target]):
    print("\nCould not identify all required columns.")
    print("Sales store column:", sales_store)
    print("Features store column:", features_store)
    print("Stores store column:", stores_store)
    print("Sales date column:", sales_date)
    print("Features date column:", features_date)
    print("Sales target column:", target)
    raise SystemExit("Please check the column names printed above.")
# 6. Merge datasets
data = pd.merge(
    sales, features,
    left_on=[sales_store, sales_date],
    right_on=[features_store, features_date],
    how="left",
    suffixes=("", "_feature")
)
data = pd.merge(
    data, stores,
    left_on=sales_store,
    right_on=stores_store,
    how="left",
    suffixes=("", "_store")
)
print("\nDatasets merged successfully!")
print("Dataset shape:", data.shape)
# 7. Prepare date features
data[sales_date] = pd.to_datetime(data[sales_date], errors="coerce")
data = data.dropna(subset=[sales_date, target])
data["year"] = data[sales_date].dt.year
data["month"] = data[sales_date].dt.month
data["day"] = data[sales_date].dt.day
data["day_of_week"] = data[sales_date].dt.dayofweek
# 8. Prepare machine learning data
numeric_data = data.select_dtypes(include=["number"]).copy()
X = numeric_data.drop(columns=[target])
y = numeric_data[target]
X = X.replace([float("inf"), float("-inf")], float("nan"))
valid = X.notna().all(axis=1) & y.notna()
X = X.loc[valid]
y = y.loc[valid]
if X.empty or len(X) < 2:
    raise SystemExit("Not enough valid data to train the model.")
# 9. Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
# 10. Train model
model = RandomForestRegressor(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)
print("\nTraining the model...")
model.fit(X_train, y_train)
# 11. Predict
predictions = model.predict(X_test)
# 12. Evaluate model
print("\n--- Retail Sales Forecasting Results ---")
print("Mean Absolute Error:", round(mean_absolute_error(y_test, predictions), 2))
print("Mean Squared Error:", round(mean_squared_error(y_test, predictions), 2))
print("R2 Score:", round(r2_score(y_test, predictions), 4))
# Save model performance metrics
metrics = pd.DataFrame({
    "Metric": ["MAE", "MSE", "R2 Score"],
    "Value": [
        mean_absolute_error(y_test, predictions),
        mean_squared_error(y_test, predictions),
        r2_score(y_test, predictions)
    ]
})
metrics.to_csv(
    os.path.join(dataset, "model_metrics.csv"),
    index=False
)
print("Model metrics saved successfully!")
# 13. Save predictions
results = pd.DataFrame({
    "Actual_Sales": y_test,
    "Predicted_Sales": predictions
})
output_path = os.path.join(dataset, "sales_predictions.csv")
results.to_csv(output_path, index=False)
# Save trained model
model_path = os.path.join(dataset, "sales_model.pkl")

joblib.dump({
    "model": model,
    "feature_columns": X.columns.tolist(),
    "default_values": X_train.median().to_dict()
}, model_path)

print("Trained model saved successfully!")
print("\nPredictions saved to:", output_path)
print("Project completed successfully!")
