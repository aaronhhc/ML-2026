import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# 1. 讀資料
df = pd.read_csv("dataset1.csv")
x = df["x"].values.reshape(-1, 1)
y = df["y"].values

# 2. normalize x
scaler = MinMaxScaler()
x_scaled = scaler.fit_transform(x)

# 3. 比較 degree 3 和 degree 8
degrees_compare = [3, 8]

plt.figure(figsize=(8, 5))
plt.scatter(x, y, s=10, label="data")

for d in degrees_compare:
    poly = PolynomialFeatures(degree=d, include_bias=True)
    X_poly = poly.fit_transform(x_scaled)

    model = LinearRegression(fit_intercept=False)
    model.fit(X_poly, y)

    y_pred = model.predict(X_poly)
    mse = mean_squared_error(y, y_pred)
    print(f"degree = {d}, MSE = {mse:.6f}")

    x_plot = np.linspace(x.min(), x.max(), 500).reshape(-1, 1)
    x_plot_scaled = scaler.transform(x_plot)
    X_plot_poly = poly.transform(x_plot_scaled)
    y_plot = model.predict(X_plot_poly)

    plt.plot(x_plot, y_plot, label=f"degree={d}")

plt.xlabel("x")
plt.ylabel("y")
plt.title("Problem 1(c): degree=3 vs degree=8")
plt.legend()
plt.tight_layout()
plt.savefig("figure/problem1c_degree3_vs_degree8.png", dpi=200)
plt.close()

print("Figure saved as figure/problem1c_degree3_vs_degree8.png")