import numpy as np 
from loadData import load_data_csv
from sklearn import ensemble 
import random
from sklearn.metrics import mean_squared_error 
from sklearn.model_selection import cross_val_score

#加载数据
data=load_data_csv()


#数据划分
x_train=data.iloc[:-8,0:-1]
y_train=data.iloc[:-8,-1]
x_test=data.iloc[-8:,0:-1]
y_test=data.iloc[-8:,-1]


#训练
clf = ensemble.GradientBoostingClassifier(n_estimators=100, learning_rate=1.0,max_depth=1, random_state=0).fit(x_train, y_train)

#输出准确率
print(clf.score(x_test, y_test)) 




