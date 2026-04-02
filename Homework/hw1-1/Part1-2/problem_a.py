import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import os

# 建立存圖資料夾
os.makedirs("figure", exist_ok=True)

# 1. 讀資料
df = pd.read_csv("dataset2.csv")

# 2. 依 split 欄位切 train / val
train_df = df[df["split"] == "train"]
val_df = df[df["split"] == "val"]

x_train = train_df["x"].values.reshape(-1, 1)
y_train = train_df["y"].values

x_val = val_df["x"].values.reshape(-1, 1)
y_val = val_df["y"].values

# 3. 只用 training set fit scaler，再 transform val
scaler = MinMaxScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_val_scaled = scaler.transform(x_val)

# 4. 試 degree = 1 ~ 11
degrees = range(1, 12)
train_mses = []
val_mses = []

for d in degrees:
    poly = PolynomialFeatures(degree=d, include_bias=True)

    X_train_poly = poly.fit_transform(x_train_scaled)
    X_val_poly = poly.transform(x_val_scaled)

    model = LinearRegression(fit_intercept=False)
    model.fit(X_train_poly, y_train)

    y_train_pred = model.predict(X_train_poly)
    y_val_pred = model.predict(X_val_poly)

    train_mse = mean_squared_error(y_train, y_train_pred)
    val_mse = mean_squared_error(y_val, y_val_pred)

    train_mses.append(train_mse)
    val_mses.append(val_mse)

    print(f"degree = {d}, Train MSE = {train_mse:.6f}, Val MSE = {val_mse:.6f}")

# 5. 畫 Train MSE vs Validation MSE
plt.figure(figsize=(8, 5))
plt.plot(list(degrees), train_mses, marker="o", label="Train MSE")
plt.plot(list(degrees), val_mses, marker="o", label="Validation MSE")
plt.xlabel("Polynomial Degree")
plt.ylabel("MSE")
plt.title("Problem 2(a): Train MSE vs Validation MSE")
plt.legend()
plt.tight_layout()
plt.savefig("figure/p2a_mse_vs_degree.png", dpi=200)
plt.close()

print("Saved: figure/p2a_mse_vs_degree.png")

# 6. 畫 2x2 subplot: degree 1, 3, 8, 11
selected_degrees = [1, 3, 8, 11]
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
axes = axes.flatten()

for ax, d in zip(axes, selected_degrees):
    poly = PolynomialFeatures(degree=d, include_bias=True)
    X_train_poly = poly.fit_transform(x_train_scaled)

    model = LinearRegression(fit_intercept=False)
    model.fit(X_train_poly, y_train)

    x_plot = np.linspace(df["x"].min(), df["x"].max(), 500).reshape(-1, 1)
    x_plot_scaled = scaler.transform(x_plot)
    X_plot_poly = poly.transform(x_plot_scaled)
    y_plot = model.predict(X_plot_poly)

    ax.scatter(x_train, y_train, s=30, label="train")
    ax.scatter(x_val, y_val, s=30, label="val")
    ax.plot(x_plot, y_plot, label=f"degree={d}")
    ax.set_title(f"degree={d}")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.legend()

plt.tight_layout()
plt.savefig("figure/p2a_selected_degrees.png", dpi=200)
plt.close()

print("Saved: figure/p2a_selected_degrees.png")