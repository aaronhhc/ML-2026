import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, PolynomialFeatures
from sklearn.linear_model import Lasso, LinearRegression
from sklearn.metrics import mean_squared_error
import os

# 建立資料夾
os.makedirs("figure", exist_ok=True)

# 1. 讀資料
df = pd.read_csv("dataset2.csv")

train_df = df[df["split"] == "train"]
val_df = df[df["split"] == "val"]

x_train = train_df["x"].values.reshape(-1, 1)
y_train = train_df["y"].values

x_val = val_df["x"].values.reshape(-1, 1)
y_val = val_df["y"].values

# 2. normalization（只用 train fit）
scaler = MinMaxScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_val_scaled = scaler.transform(x_val)

# 3. 固定 overfitting degree = 8
degree = 8
poly = PolynomialFeatures(degree=degree, include_bias=True)

X_train_poly = poly.fit_transform(x_train_scaled)
X_val_poly = poly.transform(x_val_scaled)

# 4. 先跑 unregularized baseline
baseline_model = LinearRegression(fit_intercept=False)
baseline_model.fit(X_train_poly, y_train)

y_train_pred_base = baseline_model.predict(X_train_poly)
y_val_pred_base = baseline_model.predict(X_val_poly)

baseline_train_mse = mean_squared_error(y_train, y_train_pred_base)
baseline_val_mse = mean_squared_error(y_val, y_val_pred_base)

print("=== Unregularized degree=8 baseline ===")
print(f"Train MSE = {baseline_train_mse:.6f}")
print(f"Val MSE   = {baseline_val_mse:.6f}")
print("Coefficients:")
print(baseline_model.coef_)
print()

# 5. 試不同 lambda
lambdas = [0.001, 0.01, 0.1, 1.0]

lasso_train_mses = []
lasso_val_mses = []

plt.figure(figsize=(8, 5))
plt.scatter(x_train, y_train, s=30, label="train")
plt.scatter(x_val, y_val, s=30, label="val")

x_plot = np.linspace(df["x"].min(), df["x"].max(), 500).reshape(-1, 1)
x_plot_scaled = scaler.transform(x_plot)
X_plot_poly = poly.transform(x_plot_scaled)

for lam in lambdas:
    model = Lasso(alpha=lam, fit_intercept=False, max_iter=100000)
    model.fit(X_train_poly, y_train)

    y_train_pred = model.predict(X_train_poly)
    y_val_pred = model.predict(X_val_poly)

    train_mse = mean_squared_error(y_train, y_train_pred)
    val_mse = mean_squared_error(y_val, y_val_pred)

    lasso_train_mses.append(train_mse)
    lasso_val_mses.append(val_mse)

    print(f"lambda = {lam}")
    print(f"Train MSE = {train_mse:.6f}")
    print(f"Val MSE   = {val_mse:.6f}")
    print("Coefficients:")
    print(model.coef_)
    print()

    y_plot = model.predict(X_plot_poly)
    plt.plot(x_plot, y_plot, label=f"lambda={lam}")

plt.xlabel("x")
plt.ylabel("y")
plt.title("Problem 2(c): Lasso Regression (degree=8)")
plt.legend()
plt.tight_layout()
plt.savefig("figure/p2c_lasso_curves.png", dpi=200)
plt.close()

print("Saved: figure/p2c_lasso_curves.png")

# 6. 畫 Train / Val MSE vs lambda
plt.figure(figsize=(8, 5))
plt.plot(lambdas, lasso_train_mses, marker="o", label="Train MSE")
plt.plot(lambdas, lasso_val_mses, marker="o", label="Validation MSE")
plt.xscale("log")
plt.xlabel("lambda (log scale)")
plt.ylabel("MSE")
plt.title("Problem 2(c): Lasso MSE vs Lambda")
plt.legend()
plt.tight_layout()
plt.savefig("figure/p2c_lasso_mse_vs_lambda.png", dpi=200)
plt.close()

print("Saved: figure/p2c_lasso_mse_vs_lambda.png")