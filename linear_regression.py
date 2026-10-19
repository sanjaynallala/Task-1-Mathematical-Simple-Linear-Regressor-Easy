
"""
Task 1: Simple Linear Regression from Scratch
Dataset: housing_data.csv
Target: price_lakhs
Feature: house_size_sqft
No scikit-learn used.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 1. Load the dataset
data = pd.read_csv("housing_data.csv")

# 2. Check the required columns
required_columns = ["house_size_sqft", "price_lakhs"]

if not all(col in data.columns for col in required_columns):
    raise ValueError("Dataset must contain house_size_sqft and price_lakhs.")

# 3. Remove missing and infinite values
data = data[required_columns].replace(
    [np.inf, -np.inf], np.nan
).dropna()

# 4. Extract input and target
X = data["house_size_sqft"].to_numpy(dtype=float)
y = data["price_lakhs"].to_numpy(dtype=float)

# 5. Validate the dataset
if len(X) < 2:
    raise ValueError("At least two valid rows are required.")

if np.unique(X).size < 2:
    raise ValueError("House sizes must contain at least two distinct values.")

if np.any(X < 0):
    raise ValueError("House sizes cannot be negative.")

# 6. Shuffle the data and split 80/20
rng = np.random.default_rng(42)
indices = rng.permutation(len(X))

split = int(0.8 * len(X))
train_idx = indices[:split]
test_idx = indices[split:]

X_train = X[train_idx]
X_test = X[test_idx]
y_train = y[train_idx]
y_test = y[test_idx]

# 7. Calculate means from training data
x_mean = np.mean(X_train)
y_mean = np.mean(y_train)

# 8. Calculate slope (m)
numerator = np.sum(
    (X_train - x_mean) * (y_train - y_mean)
)
denominator = np.sum((X_train - x_mean) ** 2)

m = numerator / denominator

# 9. Calculate intercept (b)
b = y_mean - m * x_mean

# 10. Predict test prices
y_pred = m * X_test + b

# 11. Calculate evaluation metrics
mse = np.mean((y_test - y_pred) ** 2)
rmse = np.sqrt(mse)
mae = np.mean(np.abs(y_test - y_pred))

# 12. Display model information
print("\nSIMPLE LINEAR REGRESSION")
print("=" * 40)
print(f"Total valid rows : {len(X)}")
print(f"Training rows    : {len(X_train)}")
print(f"Testing rows     : {len(X_test)}")
print(f"Slope (m)        : {m:.6f}")
print(f"Intercept (b)    : {b:.6f}")
print(f"Equation         : Price = {m:.6f} * Size + ({b:.6f})")

# 13. Display evaluation results
print("\nMODEL EVALUATION")
print("=" * 40)
print(f"Test MSE  : {mse:.4f}")
print(f"Test RMSE : {rmse:.4f} lakhs")
print(f"Test MAE  : {mae:.4f} lakhs")

# 14. Predict the price of a new house
new_size = 1750
predicted_price = m * new_size + b

print("\nNEW HOUSE PREDICTION")
print("=" * 40)
print(f"House size      : {new_size} sq. ft.")
print(f"Predicted price : {predicted_price:.2f} lakhs")

# 15. Plot actual test prices and regression line
order = np.argsort(X_test)

plt.figure(figsize=(10, 6))
plt.scatter(
    X_test, y_test,
    alpha=0.7,
    label="Actual test prices"
)
plt.plot(
    X_test[order],
    y_pred[order],
    linewidth=2,
    label="Regression line"
)

plt.xlabel("House Size (sq. ft.)")
plt.ylabel("House Price (₹ lakhs)")
plt.title("Housing Price Prediction using Linear Regression")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()

# 16. Save and display the graph
plt.savefig("regression_plot.png", dpi=150)
print("\nGraph saved as regression_plot.png")
plt.show()
