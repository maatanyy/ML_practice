import pandas as pd
import math
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import LSTM,Dense
train= pd.read_csv('train.csv')
test= pd.read_csv('test.csv')

train_value=train.iloc[:, 2:6].values
#print(train_value)

sc_x=MinMaxScaler(feature_range=(0,1))
sc_y=MinMaxScaler(feature_range=(0,1))

xtrain=[]
ytrain=[]

for i in range(0,len(train_value)):
    xtrain.append(train_value[i,0:3])
    ytrain.append(train_value[i,3:4])

xtrain=sc_x.fit_transform(xtrain)
ytrain=sc_y.fit_transform(ytrain)

xtrain, ytrain = np.array(xtrain), np.array(ytrain)

xtrain= np.reshape(xtrain,(xtrain.shape[0],xtrain.shape[1],1))

model = Sequential()
model.add(LSTM(128, dropout=0.3, return_sequences= True,input_shape=(xtrain.shape[1], 1)))
model.add(LSTM(32, return_sequences=False, dropout=0.3))
model.add(Dense(50))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
print('Train...')

history=model.fit(xtrain,ytrain,validation_data=(xtrain,ytrain),verbose=2,epochs=500)

model.summary()

test_value=test.iloc[:, 2:7].values
#print(test_value)

xtest= []
real_price=[]
height=[]
real_bmi=[]
for i in range(0,len(test_value)):
     xtest.append(test_value[i,0:3]) #creating input for lstm prediction
     real_price.append(test_value[i,3:4])
     height.append(test_value[i,2:3])
     real_bmi.append(test_value[i,4:5])

xtest=sc_x.fit_transform(xtest)
xtest = np.array(xtest)
'''
predicted_value : 0~1로 된 예측값
inv_real : 0~1로 된 실제값

real_price : 실제 몸무게(스케일링 X)
predicted_value : 인버스된 예측 몸무게
'''
real_price=np.array(real_price)

xtest= np.reshape(xtest,(xtest.shape[0],xtest.shape[1],1))
predicted_value= model.predict(xtest)
#숫자로 만들기
inv_predicted_value=sc_y.inverse_transform(predicted_value)
inv_real=sc_y.fit_transform(real_price)

#print("predicted", inv_predicted_value)

predicted_bmi=[]
for i in range(len(xtest)):
    predicted_bmi.append(inv_predicted_value[i]/((height[i]/100)*(height[i]/100)))

A=0
for i in range(len(xtest)):
     A+=(predicted_value[i]-inv_real[i])**2

RMSE=math.sqrt(A/(len(xtest)))
print("Used Weight Value RMSE : ",RMSE)

plt.figure(figsize=(15,8))

plt.subplots_adjust(hspace=0.5,wspace=0.4)

gridshape=(1,1)

loc=(0,0)
plt.subplot2grid(gridshape,loc)
plt.ylabel('loss')
plt.xlabel('epochs')

y_loss=history.history['loss']
x_len = np.arange(len(y_loss))
plt.plot(x_len, y_loss, marker='.', c='red', label="Train-set Loss")
plt.legend(loc='upper right')

# loc=(0,0)
# plt.subplot2grid(gridshape,loc)
# plt.ylabel('BMI')
# plt.xlabel('People')
# plt.title('Real Value')
# plt.ylim([10, 30])
# for i in range(len(xtest)):
#     if(10.5<=real_bmi[i]<14.9):
#         plt.scatter(i,real_bmi[i],c="r")
#     elif(14.9<=real_bmi[i]<19.3):
#         plt.scatter(i,real_bmi[i],c="b")
#     # elif(16.37<=real_bmi[i]<17.84):
#     #     plt.scatter(i,real_bmi[i],c="g")
#     # elif(17.84<=real_bmi[i]<19.3):
#     #     plt.scatter(i,real_bmi[i],c="c")
#     elif(19.3<=real_bmi[i]<23.7):
#         plt.scatter(i,real_bmi[i],c="m")
#     # elif(20.77<=real_bmi[i]<22.24):
#     #     plt.scatter(i,real_bmi[i],c="y")
#     # elif(22.24<=real_bmi[i]<23.7):
#     #     plt.scatter(i,real_bmi[i],c="k")
#     elif(23.7<=real_bmi[i]<28.1):
#         plt.scatter(i,real_bmi[i],c="violet")
#
# loc=(0,1)
# plt.subplot2grid(gridshape,loc)
# plt.ylabel('BMI')
# plt.xlabel('People')
# plt.title('Predict BMI')
# plt.ylim([10, 30])
# for j in range(len(xtest)):
#     if(10.5<=predicted_bmi[j]<14.9):
#         plt.scatter(j,predicted_bmi[j], c="r")
#
#     elif(14.9<=predicted_bmi[j]<19.3):
#         plt.scatter(j,predicted_bmi[j], c="b")
#
#     # elif(16.37<=predicted_bmi[j]<17.84):
#     #     plt.scatter(j,predicted_bmi[j], c="g")
#     #
#     # elif(17.84<=predicted_bmi[j]<19.3):
#     #     plt.scatter(j,predicted_bmi[j], c="c")
#
#     elif(19.3<=predicted_bmi[j]<23.7):
#         plt.scatter(j,predicted_bmi[j], c="m")
#
#     # elif(20.77<=predicted_bmi[j]<22.24):
#     #     plt.scatter(j,predicted_bmi[j], c="y")
#     #
#     # elif(22.24<=predicted_bmi[j]<23.7):
#     #     plt.scatter(j,predicted_bmi[j], c="k")
#
#     elif(23.7<=predicted_bmi[j]<28.1):
#         plt.scatter(j,predicted_bmi[j], c="violet")
#
# loc=(1,0)
# plt.subplot2grid(gridshape,loc,colspan=2)
# plt.ylabel('BMI')
# plt.xlabel('People')
# plt.title('Compare')
# plt.ylim([10, 30])
# count=0
# for j in range(len(xtest)):
#     if(10.5<=predicted_bmi[j]<14.9 and 10.5<=real_bmi[j]<14.9):
#         plt.scatter(j, real_bmi[j], c="r")
#         count=count+1
#     elif(14.9<=predicted_bmi[j]<19.3 and 14.9<=real_bmi[j]<19.3):
#         plt.scatter(j, real_bmi[j], c="b")
#         count = count + 1
#     elif(19.3<=predicted_bmi[i]<23.7 and 19.3<=real_bmi[j]<23.7):
#         plt.scatter(j, real_bmi[j], c="m")
#         count = count + 1
#     elif(23.7<=predicted_bmi[j]<28.1 and 23.7<=real_bmi[j]<28.1):
#         plt.scatter(j, real_bmi[j], c="violet")
#         count = count + 1
#
# plt.legend(loc='upper right')
#
# print("성공률 : " ,count/len(xtest)*100)
# print("Count : ",count)
plt.show()