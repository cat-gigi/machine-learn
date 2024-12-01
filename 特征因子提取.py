import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 假设这是您的数据框
data = {
    'pH': [6.8, 6.5, 6.9, 7.0, 6.7],
    'Salinity': [0.5, 0.4, 0.6, 0.5, 0.3],
    'Moisture Content': [36.2, 34.7, 38.9, 40.3, 35.4],
    'Organic Carbon Level': [24.5, 22.8, 25.2, 27.6, 23.9],
    'Total Carbon Level': [29.7, 27.1, 31.0, 33.8, 28.4],
    'Microbial Abundance': [0.57, 0.51, 0.59, 0.62, 0.54]
}

df = pd.DataFrame(data)

# 计算皮尔逊相关性矩阵
correlation_matrix = df.corr()

# 设置图形大小和背景颜色
plt.figure(figsize=(8, 6), facecolor='#f7f7f7')

# 创建热图
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1, fmt='.2f', linewidths=0.5,
            cbar_kws={'shrink': 0.75, 'label': 'Correlation Coefficient'}, annot_kws={'size': 12})

# 调整标题和轴标签
plt.title('Correlation Heatmap', fontsize=16, fontweight='bold', color='#333333')
plt.xticks(fontsize=12, rotation=45, ha='right', color='#555555')
plt.yticks(fontsize=12, rotation=0, ha='right', color='#555555')

# 设置图表的背景颜色
plt.gca().set_facecolor('#f7f7f7')

# 显示热图
plt.tight_layout()
plt.show()
