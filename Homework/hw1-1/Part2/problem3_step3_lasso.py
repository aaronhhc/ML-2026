import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, PolynomialFeatures
from sklearn.linear_model import Lasso, LinearRegression
from sklearn.metrics import mean_squared_error

os.makedirs("figure", exist_ok=True)

# 1. 讀資料
df = pd.read_csv("advertising.csv")
df = df[["TV", "Sales"]].dropna()

X = df[["TV"]].values
y = df["Sales"].values

# 2. train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 3. normalize
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 4. 固定 overfitting degree = 14
degree = 14
poly = PolynomialFeatures(degree=degree, include_bias=True)

X_train_poly = poly.fit_transform(X_train_scaled)
X_test_poly = poly.transform(X_test_scaled)

# 5. baseline
baseline_model = LinearRegression(fit_intercept=False)
baseline_model.fit(X_train_poly, y_train)

y_train_pred_base = baseline_model.predict(X_train_poly)
y_test_pred_base = baseline_model.predict(X_test_poly)

baseline_train_mse = mean_squared_error(y_train, y_train_pred_base)
baseline_test_mse = mean_squared_error(y_test, y_test_pred_base)

print("=== Unregularized degree=14 baseline ===")
print(f"Train MSE = {baseline_train_mse:.6f}")
print(f"Test MSE  = {baseline_test_mse:.6f}")
print("Coefficients:")
print(baseline_model.coef_)
print()

# 6. Lasso
lambdas = [0.001, 0.01, 0.1, 1.0]
lasso_train_mses = []
lasso_test_mses = []

plt.figure(figsize=(8, 5))
plt.scatter(X_train, y_train, s=25, label="train")
plt.scatter(X_test, y_test, s=25, label="test")

x_plot = np.linspace(X.min(), X.max(), 500).reshape(-1, 1)
x_plot_scaled = scaler.transform(x_plot)
X_plot_poly = poly.transform(x_plot_scaled)

for lam in lambdas:
    model = Lasso(alpha=lam, fit_intercept=False, max_iter=100000)
    model.fit(X_train_poly, y_train)

    y_train_pred = model.predict(X_train_poly)
    y_test_pred = model.predict(X_test_poly)

    train_mse = mean_squared_error(y_train, y_train_pred)
    test_mse = mean_squared_error(y_test, y_test_pred)

    lasso_train_mses.append(train_mse)
    lasso_test_mses.append(test_mse)

    print(f"lambda = {lam}")
    print(f"Train MSE = {train_mse:.6f}")
    print(f"Test MSE  = {test_mse:.6f}")
    print("Coefficients:")
    print(model.coef_)
    print()

    y_plot = model.predict(X_plot_poly)
    plt.plot(x_plot, y_plot, label=f"lambda={lam}")

plt.xlabel("TV")
plt.ylabel("Sales")
plt.title("Problem 3 Step 3: Lasso Regression (degree=14)")
plt.legend()
plt.tight_layout()
plt.savefig("figure/p3_step3_lasso_curves.png", dpi=200)
plt.close()

print("Saved: figure/p3_step3_lasso_curves.png")

plt.figure(figsize=(8, 5))
plt.plot(lambdas, lasso_train_mses, marker="o", label="Train MSE")
plt.plot(lambdas, lasso_test_mses, marker="o", label="Test MSE")
plt.xscale("log")
plt.xlabel("lambda (log scale)")
plt.ylabel("MSE")
plt.title("Problem 3 Step 3: Lasso MSE vs Lambda")
plt.legend()
plt.tight_layout()
plt.savefig("figure/p3_step3_lasso_mse_vs_lambda.png", dpi=200)
plt.close()

print("Saved: figure/p3_step3_lasso_mse_vs_lambda.png")