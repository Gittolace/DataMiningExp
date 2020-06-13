from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
from loadData import load_data_csv
import warnings
warnings.filterwarnings("ignore")

#加载数据
data=load_data_csv()

#数据划分
x_train=data.iloc[:-8,0:-1]
y_train=data.iloc[:-8,-1]
x_test=data.iloc[-8:,0:-1]
y_test=data.iloc[-8:,-1]

# 标准化转换
scaler = StandardScaler()

# 训练标准化对象
scaler.fit(x_train)

# 转换数据集
x_train= scaler.transform(x_train)
x_test= scaler.transform(x_test)


#默认信息增益为gini
clf = RandomForestClassifier()

#训练
clf.fit(x_train,y_train)
#预测测试集
predict_results=clf.predict(x_test)
#准确率
print(accuracy_score(predict_results, y_test))
#分类矩阵
conf_mat = confusion_matrix(y_test, predict_results)
print(conf_mat)
#分类各项指标P,R,F1
print(classification_report(y_test, predict_results))
