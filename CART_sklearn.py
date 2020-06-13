import pandas as pd

from sklearn import tree

from loadData import load_data_csv

import numpy as np


#加载数据
data = load_data_csv()

#数据集长度为79，留8个用作测试集
x_train=data.iloc[:-8,0:-1]
y_train=data.iloc[:-8,-1]
x_test=data.iloc[-8:,0:-1]
y_test=data.iloc[-8:,-1]


#创建决策树模型对象，默认为CART
dt = tree.DecisionTreeClassifier()
dt.fit(x_train, y_train)
    
#显示训练结果
print(dt.score(x_test, y_test))
