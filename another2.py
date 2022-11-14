#RNN을 이용한 kospi 주가예측 -> 데이터셋은 내가 바꿨음
#출처 : https://yeong-jin-data-blog.tistory.com/entry/%ED%8C%8C%EC%9D%B4%ED%86%A0%EC%B9%98-%EC%8A%A4%ED%84%B0%EB%94%94-RNN?category=993411

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

bmiTestData = pd.read_csv('Data2_test.csv')
bmiTrainData = pd.read_csv('Data2_train.csv')

bmiTestData['ID'] = pd.to_numeric(bmiTestData['ID'], errors='coerce')
bmiTestData['ID'] = bmiTestData['ID'].astype(float)
bmiTestData['collect_datetime'] = pd.to_numeric(bmiTestData['collect_datetime'], errors='coerce')
bmiTestData['collect_datetime'] = bmiTestData['collect_datetime'].astype(float)
bmiTestData = bmiTestData.drop(columns=['ID','collect_datetime'])

bmiTrainData['ID'] = pd.to_numeric(bmiTrainData['ID'], errors='coerce')
bmiTrainData['ID'] = bmiTrainData['ID'].astype(float)
bmiTrainData['collect_datetime'] = pd.to_numeric(bmiTrainData['collect_datetime'], errors='coerce')
bmiTrainData['collect_datetime'] = bmiTrainData['collect_datetime'].astype(float)
bmiTrainData = bmiTrainData.drop(columns=['ID','collect_datetime'])

scaler_x = MinMaxScaler()
bmiTestData[['height','step_count','eat_calory','weight']] = scaler_x.fit_transform(bmiTestData[['height','step_count','eat_calory','weight']])
scaler_y = MinMaxScaler()
bmiTestData['BMI'] = scaler_y.fit_transform(bmiTestData['BMI'].values.reshape(-1,1))

scaler_a = MinMaxScaler()
bmiTrainData[['height','step_count','eat_calory','weight']] = scaler_a.fit_transform(bmiTrainData[['height','step_count','eat_calory','weight']])
scaler_b = MinMaxScaler()
bmiTrainData['BMI'] = scaler_b.fit_transform(bmiTrainData['BMI'].values.reshape(-1,1))
pd.set_option('mode.chained_assignment',  None)

# 넘파이 배열로 변경
x = bmiTestData[['height','step_count','eat_calory','weight']].values
x = bmiTrainData[['BMI']].values
y = bmiTestData[['height','step_count','eat_calory','weight']].values
y = bmiTrainData[['BMI']].values



# 시퀀스 데이터 생성
def seq_data(x, y, sequence_length):
    x_seq = []
    y_seq = []
    for i in range(len(x) - sequence_length):
        x_seq.append(x[i: i + sequence_length])
        y_seq.append(y[i + sequence_length])

    # gpu용 텐서로 변환
    return torch.FloatTensor(x_seq).to(device), torch.FloatTensor(y_seq).to(device).view(-1, 1)


split = 2000
sequence_length = 45
x_seq, y_seq = seq_data(x, y, sequence_length)

# 순서대로 200개는 학습, 나머지는 평가
x_train_seq = x_seq[:split]
y_train_seq = y_seq[:split]

x_test_seq = x_seq[split:]
y_test_seq = y_seq[split:]

print(x_train_seq.size(), y_train_seq.size())
print(x_test_seq.size(), y_test_seq.size())

train = torch.utils.data.TensorDataset(x_train_seq, y_train_seq)
test = torch.utils.data.TensorDataset(x_test_seq, y_test_seq)

batch_size = 32
train_loader = torch.utils.data.DataLoader(dataset=train, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test, batch_size=batch_size)

# 하이퍼 파라미터 정의
input_size = x_seq.size(2)
num_layers = 2
hidden_size = 8


class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, sequence_length, num_layers, device):
        super(LSTM, self).__init__()
        self.device = device
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size * sequence_length, 1)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size()[0], self.hidden_size).to(self.device)
        c0 = torch.zeros(self.num_layers, x.size()[0], self.hidden_size).to(self.device)
        out, _ = self.lstm(x, (h0, c0))
        out = out.reshape(out.shape[0], -1)
        out = self.fc(out)
        return out


model = LSTM(input_size = input_size,
             hidden_size = hidden_size,
             sequence_length = sequence_length,
             num_layers = num_layers,
             device = device).to(device)

criterion = nn.MSELoss()
num_epochs = 401
optimizer = optim.Adam(model.parameters(), lr=1e-3)

loss_graph = []
n = len(train_loader)

for epoch in range(num_epochs):
    running_loss = 0.0

    for data in train_loader:
        seq, target = data  # 배치 데이터
        out = model(seq)
        loss = torch.sqrt(criterion(out, target))   #RMSE

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    loss_graph.append(running_loss / n)
    if epoch % 100 == 0:
        print('[epoch: %d] loss: %.4f' % (epoch, running_loss / n))

concatdata = torch.utils.data.ConcatDataset([train, test])
data_loader = torch.utils.data.DataLoader(dataset=concatdata, batch_size=100, shuffle=False)

with torch.no_grad():
    pred = []
    model.eval()
    for data in data_loader:
        seq, target = data
        out = model(seq)
        pred += out.cpu().tolist()

plt.figure(figsize=(20, 10))
plt.plot(np.ones(100) * len(train), np.linspace(0, 1, 100), '--', linewidth=0.6)
plt.plot(bmiTestData['BMI'][sequence_length:].values, '--')
plt.plot(pred, 'b', linewidth=0.6)
plt.legend(['train boundary', 'actual', 'prediction'])
plt.show()
