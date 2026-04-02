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

# 2. 先把 x normalize 到 [0, 1]
# 作業提示有特別提醒高次 polynomial 前最好 normalize
scaler = MinMaxScaler()
x_scaled = scaler.fit_transform(x)

# 3. 建立 degree = 2 的 polynomial features
poly = PolynomialFeatures(degree=2, include_bias=True)
X_poly = poly.fit_transform(x_scaled)

# 4. fit regression
# 因為 include_bias=True 已經有常數項 1，所以這裡 fit_intercept=False
model = LinearRegression(fit_intercept=False)
model.fit(X_poly, y)

# 5. 取出係數
w0, w1, w2 = model.coef_
print("w0 =", w0)
print("w1 =", w1)
print("w2 =", w2)

# 6. 預測訓練資料並算 MSE
y_pred = model.predict(X_poly)
mse = mean_squared_error(y, y_pred)
print("Train MSE =", mse)

# 7. 畫圖
x_plot = np.linspace(x.min(), x.max(), 500).reshape(-1, 1)
x_plot_scaled = scaler.transform(x_plot)
X_plot_poly = poly.transform(x_plot_scaled)
y_plot = model.predict(X_plot_poly)

plt.figure(figsize=(8, 5))
plt.scatter(x, y, s=10, label="data")
plt.plot(x_plot, y_plot, label="degree=2 fit")
plt.xlabel("x")
plt.ylabel("y")
plt.title("Problem 1(a): Polynomial Regression (degree=2)")
plt.legend()
plt.tight_layout()
plt.savefig("problem1a_degree2.png", dpi=200)
print("Figure saved as problem1a_degree2.png")