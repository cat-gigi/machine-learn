import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

# 1. 原始数据准备（模拟数据）
data = {
    'pH': [6.8, 6.5, 6.9, 7.0, 6.7, 6.9, 6.3, 6.6, 6.8, 7.1,
           6.6, 6.9, 6.7, 6.4, 6.3, 6.8, 6.2, 7.1, 6.6, 6.7,
           6.8, 6.5, 6.4, 6.9, 6.8, 7.0, 6.5, 6.7, 6.8, 6.5],
    'Salinity': [0.5, 0.4, 0.6, 0.5, 0.3, 0.6, 0.4, 0.3, 0.4, 0.6,
                 0.4, 0.5, 0.4, 0.5, 0.5, 0.5, 0.4, 0.9, 0.8, 0.7,
                 0.6, 0.5, 0.5, 0.6, 0.6, 0.8, 0.6, 0.7, 0.6, 0.6],
    'Moisture Content': [36.2, 34.7, 38.9, 40.3, 35.4, 31.5, 34.1, 35.8, 32.7, 41.1,
                         30.2, 39.1, 36.5, 37.2, 36.8, 30.8, 38.3, 39.2, 33.2, 36.8,
                         23.7, 28.9, 27.8, 25.6, 28.1, 27.3, 29.6, 24.1, 23.7, 30.8],
    'Organic Carbon Level': [24.5, 22.8, 25.2, 27.6, 23.9, 23.1, 22.0, 23.8, 26.2, 26.2,
                             23.5, 25.7, 24.2, 22.1, 23.3, 23.8, 22.5, 24.5, 21.7, 22.1,
                             20.8, 19.2, 21.7, 21.7, 23.0, 21.3, 22.3, 20.3, 23.0, 22.1],
    'Total Carbon Level': [29.7, 27.1, 31.0, 33.8, 28.4, 27.6, 27.3, 28.4, 33.5, 34.7,
                           30.3, 32.6, 30.1, 28.7, 28.1, 30.2, 28.4, 29.0, 28.8, 30.1,
                           26.3, 23.6, 26.6, 28.5, 27.5, 27.1, 28.1, 27.4, 29.2, 28.5],
    'Microbial Abundance': [0.57, 0.51, 0.59, 0.62, 0.54, 0.60, 0.53, 0.50, 0.63, 0.62,
                            0.55, 0.49, 0.51, 0.50, 0.45, 0.52, 0.45, 0.56, 0.50, 0.55,
                            0.47, 0.38, 0.48, 0.50, 0.54, 0.53, 0.46, 0.47, 0.46, 0.49]
}

df = pd.DataFrame(data)

# 2. 提取特征和目标变量
X = df.drop('Microbial Abundance', axis=1)
y = df['Microbial Abundance']

# 3. 数据标准化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 4. 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# 5. 训练随机森林模型
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# 6. 进行预测
y_train_pred = rf_model.predict(X_train)
y_test_pred = rf_model.predict(X_test)

# 7. 计算模型评估指标
train_r2 = r2_score(y_train, y_train_pred)
test_r2 = r2_score(y_test, y_test_pred)
train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))

# 输出评估结果
print("Train R²:", train_r2)
print("Test R²:", test_r2)
print("Train RMSE:", train_rmse)
print("Test RMSE:", test_rmse)

# 8. 可视化优化

# 设置整体的风格和背景
sns.set(style="whitegrid")  # 选择一个更简洁的背景
plt.rcParams.update({'axes.facecolor': '#f4f4f4', 'figure.facecolor': '#f4f4f4'})  # 设置图表背景色
plt.rcParams['grid.color'] = 'white'  # 设置网格线颜色为白色

# 可视化训练集与测试集的实际值与预测值对比
plt.figure(figsize=(8, 6))
plt.scatter(y_train, y_train_pred, color='#1f77b4', edgecolor='black', alpha=0.7, label='Train')  # 蓝色
plt.scatter(y_test, y_test_pred, color='#ff7f0e', edgecolor='black', alpha=0.7, label='Test')  # 橙色
plt.plot([0, 1], [0, 1], '--', color='gray')  # 对角线
plt.title(f'Actual vs Predicted Microbial Abundance\nTrain R²={train_r2:.3f}, Test R²={test_r2:.3f}', fontsize=14)
plt.xlabel('Actual Microbial Abundance', fontsize=12)
plt.ylabel('Predicted Microbial Abundance', fontsize=12)
plt.legend()
plt.grid(True)
plt.show()

# 可视化训练集与测试集的特征重要性
feature_importances = rf_model.feature_importances_
features = X.columns

plt.figure(figsize=(8, 6))
sns.barplot(x=feature_importances, y=features, color='#2ca02c')  # 绿色
plt.title('Feature Importance', fontsize=14)
plt.xlabel('Importance', fontsize=12)
plt.ylabel('Feature', fontsize=12)
plt.show()

# 预测结果的分布（训练集与测试集的分布）
plt.figure(figsize=(8, 6))
sns.histplot(y_train_pred, kde=True, color='#1f77b4', bins=10, label='Train', alpha=0.7)
sns.histplot(y_test_pred, kde=True, color='#ff7f0e', bins=10, label='Test', alpha=0.7)
plt.title(f'Distribution of Predicted Microbial Abundance\nTrain RMSE={train_rmse:.3f}, Test RMSE={test_rmse:.3f}', fontsize=14)
plt.xlabel('Predicted Microbial Abundance', fontsize=12)
plt.ylabel('Frequency', fontsize=12)
plt.legend()
plt.grid(True)
plt.show()
