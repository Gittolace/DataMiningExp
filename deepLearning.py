import os
os.environ['CUDA_VISIBLE_DEVICES'] = "0"
from keras.callbacks import ModelCheckpoint
from keras.optimizers import SGD
from keras.utils import np_utils
from model import myModel
from loadData import load_data_csv
import numpy as np
print('loading data.....')
#读取数据，将dataframe类型转换为npdarray
data=load_data_csv().values

#数据划分
x_train=data[:-8,0:-1]
y_train=data[:-8,-1:]

#展平输入
x_train=x_train.reshape(len(x_train),-1)


#转换输出
y_train=np.array([int(i-1) for i in y_train])
y_train=np_utils.to_categorical(y_train)

#建立模型
model=myModel.build((8,),4)

#随机梯度下降，学习率0.0001,动量0.9
sgd=SGD(lr=0.0001,momentum=0.9)

#交叉熵损失
model.compile(loss='categorical_crossentropy',metrics=['accuracy'],optimizer=sgd)

#添加checkPoint保存最好模型
checkPoint=ModelCheckpoint(filepath='best.h5',monitor='acc',verbose=1,save_best_only='true',save_weights_only='False',mode='auto',period=1)

#训练
model.fit(x_train, y_train,batch_size=71,callbacks=[checkPoint] ,epochs=3000,shuffle=True)


print('saving model and weights......')
#保存模型
model_json=model.to_json()
open('model.json','w').write(model_json)

#保存权重
model.save_weights('last.h5', overwrite=True)
print("complete")