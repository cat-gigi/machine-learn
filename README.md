机器学习里RF的生物多样性应用，可用于根据特征因子预测目标变量变化，用于菌群丰度啥的预测，这个库后面会慢慢更新，下面是要导的库：
大家有啥建议直接评论哈，欢迎一起交流！
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.ensemble import RandomForestRegressor

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler

from sklearn.metrics import mean_absolute_error, r2_score

import joblib
