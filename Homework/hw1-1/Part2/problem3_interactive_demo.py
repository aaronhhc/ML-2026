import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error

st.set_page_config(page_title="多項式迴歸與正規化互動展示", layout="wide")

st.title("互動式展示：多項式迴歸、過擬合與正規化")
st.markdown("""
本應用程式展示了 **多項式迴歸 (Polynomial Regression)** 模型在不同多項式次數 ($m$)、
切割資料集比例以及正規化 (Regularization) 設定下的表現。您可以透過左側選單調整參數。
""")

# 1. 讀取資料
@st.cache_data
def load_data():
    df = pd.read_csv("advertising.csv")
    df = df[["TV", "Sales"]].dropna()
    X = df[["TV"]].values
    y = df["Sales"].values
    return X, y

X, y = load_data()

st.sidebar.header("🔧 模型參數設定")

# 控制：切分比例
test_ratio = st.sidebar.slider("測試資料比例 (Test Ratio)", min_value=0.1, max_value=0.9, value=0.2, step=0.05)

# 控制：多項式次數
degree = st.sidebar.slider("多項式次數 (Polynomial Degree)", min_value=1, max_value=20, value=14, step=1)

# 控制：正規化選項
reg_type = st.sidebar.selectbox("正規化方法 (Regularization)", ["None (無)", "Ridge (L2)", "Lasso (L1)"])

# 控制：正規化強度 (Lambda / Alpha)
reg_alpha = 0.0
if reg_type != "None (無)":
    # 讓使用者可以選擇以對數或線性方式調整 alpha
    reg_alpha_log = st.sidebar.slider("正規化強度 log10(Lambda)", min_value=-4.0, max_value=2.0, value=-1.0, step=0.5)
    reg_alpha = 10 ** reg_alpha_log
    st.sidebar.write(f"目前 $\\lambda$ (Alpha) = {reg_alpha:.4f}")

# 隨機種子設定 (確保結果可重現)
random_state = st.sidebar.number_input("Random State (隨機種子)", value=42, step=1)

# 2. 資料分割與標準化
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=test_ratio, random_state=random_state
)

scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 多項式特徵轉換
poly = PolynomialFeatures(degree=degree, include_bias=True)
X_train_poly = poly.fit_transform(X_train_scaled)
X_test_poly = poly.transform(X_test_scaled)

# 3. 建立與訓練模型
if reg_type == "None (無)":
    model = LinearRegression(fit_intercept=False)
elif reg_type == "Ridge (L2)":
    model = Ridge(alpha=reg_alpha, fit_intercept=False)
else:
    model = Lasso(alpha=reg_alpha, fit_intercept=False, max_iter=10000)

model.fit(X_train_poly, y_train)

# 4. 預測與評估
y_train_pred = model.predict(X_train_poly)
y_test_pred = model.predict(X_test_poly)

train_mse = mean_squared_error(y_train, y_train_pred)
test_mse = mean_squared_error(y_test, y_test_pred)

# 顯示結果佈局
col1, col2 = st.columns((2, 1))

with col1:
    st.subheader("擬合曲線")
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # 繪製原始資料點
    ax.scatter(X_train, y_train, s=30, label="Train Data", alpha=0.8)
    ax.scatter(X_test, y_test, s=30, label="Test Data", alpha=0.8)
    
    # 產生畫線用的平滑X
    x_plot = np.linspace(X.min(), X.max(), 500).reshape(-1, 1)
    x_plot_scaled = scaler.transform(x_plot)
    X_plot_poly = poly.transform(x_plot_scaled)
    y_plot = model.predict(X_plot_poly)
    
    ax.plot(x_plot, y_plot, color="red", linewidth=2, label="Fitted Curve")
    ax.set_xlabel("TV")
    ax.set_ylabel("Sales")
    ax.set_title(f"Fit Result (Degree={degree}, Reg={reg_type})")
    ax.legend()
    st.pyplot(fig)

with col2:
    st.subheader("訓練結果")
    st.metric(label="訓練集 (Train) MSE", value=f"{train_mse:.4f}")
    st.metric(label="測試集 (Test) MSE", value=f"{test_mse:.4f}")
    
    st.markdown("---")
    st.subheader("模型係數")
    
    coef_df = pd.DataFrame({
        "Feature": [f"x^{i}" for i in range(degree + 1)],
        "Coefficient": model.coef_
    })
    st.dataframe(coef_df, use_container_width=True)

# 顯示所有degree的比較圖
st.markdown("---")
st.subheader("不同多項式次數的 MSE 變化 (固定目前其他設定)")

with st.spinner("計算中，請稍候..."):
    degrees_range = list(range(1, 21))
    train_mses = []
    test_mses = []
    
    for d in degrees_range:
        p = PolynomialFeatures(degree=d, include_bias=True)
        Xt_p = p.fit_transform(X_train_scaled)
        Xv_p = p.transform(X_test_scaled)
        
        if reg_type == "None (無)":
            m = LinearRegression(fit_intercept=False)
        elif reg_type == "Ridge (L2)":
            m = Ridge(alpha=reg_alpha, fit_intercept=False)
        else:
            m = Lasso(alpha=reg_alpha, fit_intercept=False, max_iter=10000)
            
        m.fit(Xt_p, y_train)
        train_mses.append(mean_squared_error(y_train, m.predict(Xt_p)))
        test_mses.append(mean_squared_error(y_test, m.predict(Xv_p)))
    
    fig2, ax2 = plt.subplots(figsize=(10, 5))
    ax2.plot(degrees_range, train_mses, marker="o", label="Train MSE")
    ax2.plot(degrees_range, test_mses, marker="o", label="Test MSE")
    ax2.set_xlabel("Polynomial Degree")
    ax2.set_ylabel("MSE")
    ax2.set_title("Train & Test MSE vs Polynomial Degree")
    ax2.set_xticks(degrees_range)
    
    # 標示目前選擇的 degree
    ax2.axvline(x=degree, color='red', linestyle='--', alpha=0.8, linewidth=2, label=f'Current Degree ({degree})')
    
    ax2.legend()
    ax2.grid(True, linestyle='--', alpha=0.6)
    
    st.pyplot(fig2)
