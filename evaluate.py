import os
os.environ['CUDA_VISIBLE_DEVICES'] = "0"
from keras.models import model_from_json
from keras.utils import np_utils
from keras.optimizers import SGD
from loadData import load_data_csv
import numpy as np

#载入模型和权重
loaded_model_json = open('model.json', 'r').read()
model=model_from_json(loaded_model_json)
model.load_weights('best.h5')

model.summary()

print('loading dataset...')

#加载数据
data=load_data_csv().values
x_test=data[-8:,0:-1]
y_test=data[-8:,-1]

#展平
x_test=x_test.reshape(len(x_test),-1)


#转换输出
y_test=np.array([int(i-1) for i in y_test])
y_test=np_utils.to_categorical(y_test)



sgd=SGD(lr=0.0001,momentum=0.9)
model.compile(loss='categorical_crossentropy',metrics=['accuracy'],optimizer=sgd)

print('evaluating...')
#评估
score=model.evaluate(x_test,y_test,batch_size=32)
print('loss: {}'.format(score[0]))
print('acc:{}'.format(score[1]))
