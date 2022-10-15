import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler

data = pd.read_excel('2weeks_summary.xlsx')
#ID	height	weight	step_count	burn_calorie	eat_calorie	sleep	bmi	bmi_target

scaler = MinMaxScaler()
data[['Weight','BMI','Step','Burn','Eat','Sleep']] = scaler.fit_transform(data[['Weight','BMI','Step','Burn','Eat','Sleep']])

Y_real = data[['Label']]
Y_real = Y_real.to_numpy()

scaler2 = MinMaxScaler()
data[['Label']] = scaler2.fit_transform(data[['Label']])


X = data[['Weight','BMI','Step','Burn','Eat','Sleep']].values
Y = data[['Label']].values

print(Y)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def seq_data(x, y, sequence_length):
    x_seq = []
    y_seq = []
    for i in range(len(x) - sequence_length):
        x_seq.append(x[i:i + sequence_length])
        y_seq.append(y[i + sequence_length])

    return torch.FloatTensor(x_seq).to(device), torch.FloatTensor(y_seq).to(device).view(-1, 1)


split = 200
#split = int(0.8 * len(X))
sequence_length = 5
x_seq, y_seq = seq_data(X, Y, sequence_length)

x_train_seq = x_seq[:split]
y_train_seq = y_seq[:split]
x_test_seq = x_seq[split:]
y_test_seq = y_seq[split:]

train = torch.utils.data.TensorDataset(x_train_seq, y_train_seq)
test = torch.utils.data.TensorDataset(x_test_seq, y_test_seq)

#print("hello",len(y_train_seq))
#print("Hi",len(y_test_seq))

trainloader = torch.utils.data.DataLoader(dataset=train, batch_size=20, shuffle=True)
testloader = torch.utils.data.DataLoader(dataset=test, batch_size=20)

input_size = x_train_seq.size(2)  # x_train_seq.size()
num_layers = 2
hidden_size = 8



### input size = 입력 피처의 개수 / hidden size = 은닉층의 피처 개수 / num_layers = LSTM layer를 몇층으로 쌓을지 / bias = bias 여부 / dropout = dropout 비율
class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, sequence_length, num_layers, device):
        super(LSTM, self).__init__()
        self.device = device
        self.input_size = input_size
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


model = LSTM(input_size=input_size,
             hidden_size=hidden_size,
             num_layers=num_layers,
             sequence_length=sequence_length,
             device=device).to(device)

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

n = len(trainloader)
loss_graph = []

for epoch in range(1000):
    running_loss = 0.0
    for data in trainloader:
        seq, target = data
        outputs = model(seq)
        optimizer.zero_grad()
        loss = criterion(outputs, target)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    loss_graph.append(running_loss / n)
    if epoch % 100 == 0:
        print("Epoch : %d loss : %.4f" % (epoch, running_loss / n))

concatdata = torch.utils.data.ConcatDataset([train, test])
data_loader = torch.utils.data.DataLoader(concatdata, batch_size=100)


with torch.no_grad():
    pred = []
    model.eval()
    for data in data_loader:
        seq, target = data
        out = model(seq)
        pred += out.cpu().tolist()

pred = scaler2.inverse_transform(pred)
Y = scaler2.inverse_transform(Y)
length = len(pred)
#print(Y[sequence_length:])
#print(data['height'][sequence_length:].values)


print("pred:", len(pred))
print("Y[split:]", len(Y[split:]))
count = 0

for i in range(length-split-sequence_length):
    if round(pred[split+i][0])==round(Y_real[split+i][0]):
        count = count+1



plt.figure(figsize=(20, 10))
plt.plot(np.ones(100) * len(train), np.linspace(0, 10, 100), '-', linewidth=0.6)
plt.plot(Y[split+sequence_length:], 'r')
plt.plot(pred[split+sequence_length:], 'b', linewidth=0.6)
plt.legend(['hi'])
plt.show()


print("count : ",count, "length:",length-split-sequence_length)
print("Accuracy: ", count/length)

#test
#전체 27047
#split 200 sequence 5

