import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# 讀資料
df = pd.read_csv("dataset1.csv")
x = df["x"].values.reshape(-1, 1)
y = df["y"].values

# normalize x
scaler = MinMaxScaler()
x_scaled = scaler.fit_transform(x)

degrees = range(1, 11)
train_mses = []

for d in degrees:
    # 建立 polynomial features
    poly = PolynomialFeatures(degree=d, include_bias=True)
    X_poly = poly.fit_transform(x_scaled)

    # fit model
    model = LinearRegression(fit_intercept=False)
    model.fit(X_poly, y)

    # predict on training data
    y_pred = model.predict(X_poly)
    mse = mean_squared_error(y, y_pred)
    train_mses.append(mse)

    print(f"degree = {d}, Train MSE = {mse:.6f}")

    # 畫 fitted curve
    x_plot = np.linspace(x.min(), x.max(), 500).reshape(-1, 1)
    x_plot_scaled = scaler.transform(x_plot)
    X_plot_poly = poly.transform(x_plot_scaled)
    y_plot = model.predict(X_plot_poly)

    plt.figure(figsize=(8, 5))
    plt.scatter(x, y, s=10, label="data")
    plt.plot(x_plot, y_plot, label=f"degree={d} fit")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title(f"Problem 1(b): degree={d}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"problem1b_degree_{d}.png", dpi=200)
    plt.close()

# 畫 MSE vs degree
plt.figure(figsize=(8, 5))
plt.plot(list(degrees), train_mses, marker="o")
plt.xlabel("Polynomial Degree")
plt.ylabel("Train MSE")
plt.title("Problem 1(b): Train MSE vs Degree")
plt.tight_layout()
plt.savefig("problem1b_mse_vs_degree.png", dpi=200)
plt.close()

print("All figures saved.")