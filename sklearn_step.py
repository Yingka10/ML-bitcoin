# 匯入必要模組
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import joblib

# 假設已有一個包含日期、收盤價、Fear & Greed Index 的 CSV 檔案
df = pd.read_csv("bitcoin_fear_greed_2018_2024.csv")  # 範例檔案名稱
print(df.head())

# 轉換日期格式與排序
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values('Date')

# 建立移動特徵，例如過去5天的 Fear & Greed Index 與收盤價
for i in range(1, 6):
    df[f'fgi_lag{i}'] = df['Fear_Greed_Index'].shift(i)
    df[f'Close_lag{i}'] = df['Close'].shift(i)

# 預測目標：隔日收盤價
df['target'] = df['Close'].shift(-1)

# 移除缺失值（因為 shift 會產生 NaN）
df.dropna(inplace=True)

# 特徵與目標
features = [col for col in df.columns if 'lag' in col]
X = df[features]
y = df['target']

# 資料切分
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

# 建立回歸模型
model = LinearRegression()
model.fit(X_train, y_train)

# 預測與評估
predictions = model.predict(X_test)
print("MSE:", mean_squared_error(y_test, predictions))
print("R^2 Score:", r2_score(y_test, predictions))

# 存模型
joblib.dump(model, "fear_greed_regression_model.pkl")