import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

os.makedirs("figure", exist_ok=True)

# 1. 讀資料
df = pd.read_csv("advertising.csv")

# 2. 選 feature 和 target
X_col = "TV"
Y_col = "Sales"

df = df[[X_col, Y_col]].dropna()

X = df[[X_col]].values
y = df[Y_col].values

# 3. train / test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 4. normalization（只用 train fit）
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 5. 試不同 degree
degrees = range(1, 21)
train_mses = []
test_mses = []

for d in degrees:
    poly = PolynomialFeatures(degree=d, include_bias=True)

    X_train_poly = poly.fit_transform(X_train_scaled)
    X_test_poly = poly.transform(X_test_scaled)

    model = LinearRegression(fit_intercept=False)
    model.fit(X_train_poly, y_train)

    y_train_pred = model.predict(X_train_poly)
    y_test_pred = model.predict(X_test_poly)

    train_mse = mean_squared_error(y_train, y_train_pred)
    test_mse = mean_squared_error(y_test, y_test_pred)

    train_mses.append(train_mse)
    test_mses.append(test_mse)

    print(f"degree = {d}, Train MSE = {train_mse:.6f}, Test MSE = {test_mse:.6f}")

# 6. 畫 Train/Test MSE vs degree
plt.figure(figsize=(8, 5))
plt.plot(list(degrees), train_mses, marker="o", label="Train MSE")
plt.plot(list(degrees), test_mses, marker="o", label="Test MSE")
plt.xlabel("Polynomial Degree")
plt.ylabel("MSE")
plt.title("Problem 3 Step 2: Train MSE vs Test MSE")
plt.legend()
plt.tight_layout()
plt.savefig("figure/p3_step2_mse_vs_degree.png", dpi=200)
plt.close()

print("Saved: figure/p3_step2_mse_vs_degree.png")

# 7. 找最佳 degree（test MSE 最低）
best_idx = np.argmin(test_mses)
best_degree = list(degrees)[best_idx]
best_train_mse = train_mses[best_idx]
best_test_mse = test_mses[best_idx]

print()
print("=== Best Model ===")
print(f"Best degree = {best_degree}")
print(f"Best Train MSE = {best_train_mse:.6f}")
print(f"Best Test MSE  = {best_test_mse:.6f}")

# 8. 畫 under-fit / good-fit / over-fit
# 先用一個簡單策略：
# under-fit = 1
# good-fit = best_degree
# over-fit = 14
selected_degrees = [1, best_degree, 14]
titles = ["Under-fit", "Good-fit", "Over-fit"]

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

x_plot = np.linspace(X.min(), X.max(), 500).reshape(-1, 1)
x_plot_scaled = scaler.transform(x_plot)

for ax, d, title in zip(axes, selected_degrees, titles):
    poly = PolynomialFeatures(degree=d, include_bias=True)
    X_train_poly = poly.fit_transform(X_train_scaled)
    X_plot_poly = poly.transform(x_plot_scaled)

    model = LinearRegression(fit_intercept=False)
    model.fit(X_train_poly, y_train)

    y_plot = model.predict(X_plot_poly)

    ax.scatter(X_train, y_train, s=25, label="train")
    ax.scatter(X_test, y_test, s=25, label="test")
    ax.plot(x_plot, y_plot, label=f"degree={d}")
    ax.set_title(f"{title} (degree={d})")
    ax.set_xlabel(X_col)
    ax.set_ylabel(Y_col)
    ax.legend()

plt.tight_layout()
plt.savefig("figure/p3_step2_fit_examples.png", dpi=200)
plt.close()

print("Saved: figure/p3_step2_fit_examples.png")