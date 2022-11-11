import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from imblearn.over_sampling import SMOTE

data = pd.read_excel('2weeks_summary2.xlsx')
#ID	height	weight	step_count	burn_calorie	eat_calorie	sleep	bmi	bmi_target

scaler = MinMaxScaler()
data[['Weight','BMI','Step','Burn','Eat','Sleep']] = scaler.fit_transform(data[['Weight','BMI','Step','Burn','Eat','Sleep']])

Y_real = data[['Label']]
Y_real = Y_real.to_numpy()

scaler2 = MinMaxScaler()
data[['Label']] = scaler2.fit_transform(data[['Label']])


X = data[['Weight','BMI','Step','Burn','Eat','Sleep']].values
data = data.astype({'Label':'int'})
Y = data[['Label']].values

print(Y)

input_size = 6        # input_size
print(input_size)

#device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")

def seq_data(x, y, sequence_length):
    x_seq = []
    y_seq = []
    for i in range(len(x) - sequence_length):
        x_seq.append(x[i:i + sequence_length])
        y_seq.append(y[i + sequence_length])

    return torch.FloatTensor(x_seq).to(device), torch.FloatTensor(y_seq).to(device).view(-1, 1)

split = 200
#split = int(0.8 * len(X))
sequence_length = 8
x_seq, y_seq = seq_data(X, Y, sequence_length)
#split = int(0.8 * len(x_seq))

x_train_seq = x_seq[:split]
y_train_seq = y_seq[:split]
x_test_seq = x_seq[split:]
y_test_seq = y_seq[split:]


#print(x_train_seq.shape)
# print(y_train_seq.shape)
# print(x_test_seq.shape)
# print(y_test_seq.shape)

nsamples, nx, ny = x_train_seq.shape
print(nsamples)
print(nx)
print(ny)

x_train_seq = x_train_seq.reshape((nsamples, nx*ny))

#print(x_train_seq.shape)
#print(y_train_seq.shape)

sm = SMOTE(random_state=0)
x_smote, y_smote = sm.fit_resample(x_train_seq, y_train_seq)

print('After SMOTE OverSampling, the shape of x: {}'.format(x_smote.shape))
print('After SMOTE OverSampling, the shape of y: {} \n'.format(y_smote.shape))

x_smote = torch.Tensor(x_smote)
tempx, tempx2  = x_smote.shape
x_smote = x_smote.reshape((tempx, sequence_length, input_size))

y_smote = torch.Tensor(y_smote)
tempy, = y_smote.shape
y_smote = y_smote.reshape((tempy,1))

print(y_smote.shape)
print(y_test_seq.shape)


train = torch.utils.data.TensorDataset(x_smote, y_smote)
test = torch.utils.data.TensorDataset(x_test_seq, y_test_seq)

trainloader = torch.utils.data.DataLoader(dataset=train, batch_size=20, shuffle=True)
testloader = torch.utils.data.DataLoader(dataset=test, batch_size=20)

num_layers = 2
hidden_size = 6

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

for epoch in range(200):
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
    #if epoch % 100 == 0:
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
    if round(pred[split+i][0]) == round(Y_real[split+i][0]):
        count = count+1

plt.figure(figsize=(20, 10))
plt.plot(np.ones(100) * len(train), np.linspace(0
                                                , 10, 100), '-', linewidth=0.6)
plt.plot(Y[split+sequence_length:], 'r')                        #빨간색이 실제
plt.plot(pred[split+sequence_length:], 'b', linewidth=0.6)      #파란색이 예측
plt.legend(['hi'])
plt.show()


print("count : ",count, "length:",length-split-sequence_length)
print("Accuracy: ", count/length)

#test
#전체 27047
#split 200 sequence 5

