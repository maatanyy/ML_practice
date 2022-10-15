#import wandb

#wandb.init(project="helpme", entity="minsungno")

#wandb.config = {
  #"learning_rate": 0.001,
  #"epochs": 100,
  #"batch_size": 128
#}

import os
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#import wandb
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from torch.autograd import Variable
#from tqdm import tqdm_notebook
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


data=pd.read_excel('Data_Timeseries.xlsx')
print(data.dtypes)

y=pd.get_dummies(data['bmi_target'])

# data.set_index('Date', inplace=True)
X = data.iloc[:,2:9]
y = y.iloc[0:84]
print(X)
print(y)

ms = MinMaxScaler()
mv = MinMaxScaler()
ss = StandardScaler()

X_ss = ss.fit_transform(X)
# y_ms = ms.fit_transform(y)
y_ms= y

X_train = X_ss[:57, :]
X_test = X_ss[57:84, :]

y_train = y_ms[:57]
y_test = y_ms[57:]

y_train=y_train.to_numpy()
y_test=y_test.to_numpy()
print("Training Shape", X_train.shape, y_train.shape)
print("Testing Shape", X_test.shape, y_test.shape)

X_train_tensors = Variable(torch.Tensor(X_train))
X_test_tensors = Variable(torch.Tensor(X_test))

type(y_train)


y_train_tensors = Variable(torch.Tensor(y_train))
y_test_tensors = Variable(torch.Tensor(y_test))

X_train_tensors_f = torch.reshape(X_train_tensors,   (X_train_tensors.shape[0], 1, X_train_tensors.shape[1]))
X_test_tensors_f = torch.reshape(X_test_tensors,  (X_test_tensors.shape[0], 1, X_test_tensors.shape[1]))

y_train_tensors_f = torch.reshape(y_train_tensors,   (y_train_tensors.shape[0], 1, y_train_tensors.shape[1]))
y_test_tensors_f = torch.reshape(y_test_tensors,  (y_test_tensors.shape[0], 1, y_test_tensors.shape[1]))


print("Training Shape", X_train_tensors_f.shape, y_train_tensors_f.shape)
print("Testing Shape", X_test_tensors_f.shape, y_test_tensors_f.shape)


#wandb.init()


class LSTM(nn.Module):
    def __init__(self, num_classes, input_size, hidden_size, num_layers, seq_length):
        super(LSTM, self).__init__()
        self.num_classes = num_classes
        self.num_layers = num_layers
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.seq_length = seq_length

        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                            num_layers=num_layers, batch_first=True)
        self.fc_1 = nn.Linear(hidden_size, 128)
        self.fc = nn.Linear(128, num_classes)

        self.relu = nn.ReLU()

    def forward(self, x):
        h_0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size))
        c_0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size))

        output, (hn, cn) = self.lstm(x, (h_0, c_0))
        hn = hn.view(-1, self.hidden_size)
        out = self.relu(hn)
        out = self.fc_1(out)
        out = self.relu(out)
        out = self.fc(out)
        return out


num_epochs = 1000
learning_rate = 0.0001

input_size = 7
hidden_size = 2
num_layers = 1

num_classes = 4
model = LSTM(num_classes, input_size, hidden_size, num_layers, X_train_tensors_f.shape[1])

criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

for epoch in range(num_epochs):
    outputs = model.forward(X_train_tensors_f)

    optimizer.zero_grad()
    loss = criterion(outputs, y_train_tensors)
    loss.backward()
    optimizer.step()
    #wandb.log({"accurate" "loss": loss})
    if epoch % 100 == 0:
        print("Epoch: %d,loss: %1.5f" % (epoch, loss.item()))


df_x_ss = ss.transform(data.iloc[:57, 2:9])
# df_y_ms = ms.transform(y.iloc[:57])
df_y_ms=y_train
df_x_ss = Variable(torch.Tensor(df_x_ss))
df_y_ms = Variable(torch.Tensor(df_y_ms))
df_x_ss = torch.reshape(df_x_ss, (df_x_ss.shape[0], 1, df_x_ss.shape[1]))

train_predict = model(df_x_ss)
predicted = train_predict.data.numpy()
label_y = df_y_ms.data.numpy()


#predicted= ms.inverse_transform(predicted)
#label_y = ms.inverse_transform(label_y)

print(len(label_y))
print(predicted)


#plt.plot(label_y, label='Actual Data')
#plt.plot(predicted, label='Predicted Data')

#plt.title('Time-Series Prediction')
#plt.legend()
#plt.show()