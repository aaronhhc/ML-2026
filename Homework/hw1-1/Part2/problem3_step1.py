import pandas as pd
import matplotlib.pyplot as plt
import os

os.makedirs("figure", exist_ok=True)

# 1. 讀資料
df = pd.read_csv("advertising.csv")

# 2. 選 feature 和 target
X_col = "TV"
Y_col = "Sales"

# 3. 只保留需要欄位，並移除缺失值
df = df[[X_col, Y_col]].dropna()

# 4. 基本資訊
n_samples = len(df)
x_mean = df[X_col].mean()
x_std = df[X_col].std()
x_min = df[X_col].min()
x_max = df[X_col].max()

y_mean = df[Y_col].mean()
y_std = df[Y_col].std()
y_min = df[Y_col].min()
y_max = df[Y_col].max()

print("=== Dataset Summary ===")
print(f"Number of samples: {n_samples}")
print(f"Feature (X): {X_col}")
print(f"Target (Y): {Y_col}")
print()
print("X statistics:")
print(f"mean = {x_mean:.4f}, std = {x_std:.4f}, min = {x_min:.4f}, max = {x_max:.4f}")
print()
print("Y statistics:")
print(f"mean = {y_mean:.4f}, std = {y_std:.4f}, min = {y_min:.4f}, max = {y_max:.4f}")

# 5. 畫 scatter plot
plt.figure(figsize=(8, 5))
plt.scatter(df[X_col], df[Y_col], s=25)
plt.xlabel(X_col)
plt.ylabel(Y_col)
plt.title("Problem 3 Step 1: Scatter Plot")
plt.tight_layout()
plt.savefig("figure/p3_step1_scatter.png", dpi=200)
plt.close()

print("Saved: figure/p3_step1_scatter.png")