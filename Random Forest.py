#C:/Users/Xiaofeng Zhang/Desktop/machine learning.csv

# 导入所需库
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score
import joblib

# 设置Seaborn全局样式
sns.set_theme(style="whitegrid", font_scale=1.2)


# 定义预测模块
class MicrobialAbundancePredictor:
    def __init__(self):
        self.model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.scaler = StandardScaler()

    def train(self, file_path):
        """
        从本地文件夹加载数据并训练随机森林模型
        :param file_path: 数据文件的路径（CSV 格式）
        """
        # 从 CSV 文件加载数据
        df = pd.read_csv(file_path)

        # 提取特征和目标变量
        X = df.drop('Microbial_Abundance', axis=1)
        y = df['Microbial_Abundance']

        # 划分训练集和测试集
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # 标准化特征数据
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        # 训练模型
        self.model.fit(X_train_scaled, y_train)

        # 测试模型性能
        y_pred = self.model.predict(X_test_scaled)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        print(f"模型训练完成！\nMean Absolute Error (MAE): {mae:.3f}\nR2 Score: {r2:.3f}")

        # 可视化部分
        self._visualize_data(df, y_test, y_pred)

    def _visualize_data(self, df, y_test, y_pred):
        """
        数据和结果可视化
        :param df: 数据集
        :param y_test: 测试集真实值
        :param y_pred: 测试集预测值
        """
        # 绘制特征之间的相关性热图
        plt.figure(figsize=(10, 8))
        sns.heatmap(df.corr(), annot=True, fmt=".2f", cmap="coolwarm", cbar=True)
        plt.title("Feature Correlation Heatmap", fontsize=14)
        plt.savefig("feature_correlation_heatmap.png", dpi=300)
        plt.show()

        # 绘制真实值 vs 预测值
        plt.figure(figsize=(8, 6))
        plt.scatter(y_test, y_pred, color="blue", alpha=0.7)
        plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], '--', color='red', linewidth=2)
        plt.title("Actual vs Predicted Microbial Abundance", fontsize=14)
        plt.xlabel("Actual Values")
        plt.ylabel("Predicted Values")
        plt.grid(True)
        plt.savefig("actual_vs_predicted.png", dpi=300)
        plt.show()

    def save_model(self, model_path, scaler_path):
        """
        保存模型和标准化器
        :param model_path: 模型保存路径
        :param scaler_path: 标准化器保存路径
        """
        joblib.dump(self.model, model_path)
        joblib.dump(self.scaler, scaler_path)
        print("模型和标准化器已保存！")

    def load_model(self, model_path, scaler_path):
        """
        加载模型和标准化器
        :param model_path: 模型保存路径
        :param scaler_path: 标准化器保存路径
        """
        self.model = joblib.load(model_path)
        self.scaler = joblib.load(scaler_path)
        print("模型和标准化器已加载！")

    def predict(self, input_data):
        """
        根据输入数据预测微生物丰度
        :param input_data: 包含环境因子的新数据字典
        :return: 预测的微生物丰度
        """
        input_df = pd.DataFrame([input_data])
        input_scaled = self.scaler.transform(input_df)
        prediction = self.model.predict(input_scaled)
        return prediction[0]


# 示例：从本地文件夹加载数据并训练模型
file_path = 'C:/Users/Xiaofeng Zhang/Desktop/machine learning.csv'  # 本地 CSV 文件路径

# 假设文件内容如下：
# Grain_Size_mm,pH,Organic_Carbon_percent,Salinity_ppt,Water_Content_percent,Microbial_Abundance
# 0.445636,7.022307,3.204789,24.017657,39.346105,3.143673
# 0.474517,7.409161,3.249379,24.097253,37.314276,3.758208
# ...

predictor = MicrobialAbundancePredictor()

# 训练模型并可视化
predictor.train(file_path)

# 保存模型和标准化器
predictor.save_model('random_forest_model.pkl', 'scaler.pkl')

# 加载模型和标准化器
predictor.load_model('random_forest_model.pkl', 'scaler.pkl')

# 预测示例
input_data = {
    'Grain_Size_mm': 0.5,
    'pH': 7.1,
    'Organic_Carbon_percent': 3.2,
    'Salinity_ppt': 25,
    'Water_Content_percent': 39.5
}

predicted_abundance = predictor.predict(input_data)
print(f"预测的微生物丰度: {predicted_abundance:.3f}")

