import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import math as mt

from sklearn.preprocessing import MinMaxScaler

data = pd.read_excel('2weeks_summary2.xlsx')
#ID	height	weight	step_count	burn_calorie	eat_calorie	sleep	bmi	bmi_target

scaler = MinMaxScaler()
data[['Weight','BMI','Step','Burn','Eat','Sleep']] = scaler.fit_transform(data[['Weight','BMI','Step','Burn','Eat','Sleep']])

Y_real = data[['Label']]            #실제 정답(Label) 을 Y_real에 담아 둔다
Y_real = Y_real.to_numpy()

scaler2 = MinMaxScaler()
data[['Label']] = scaler2.fit_transform(data[['Label']])


X = data[['Weight', 'BMI', 'Step', 'Burn', 'Eat', 'Sleep']].values     #X에 scale한 변수가 들어감
Y = data[['Label']].values                                             #Y에 scale한 라벨이 들어감

#print(Y)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def seq_data(x, y, sequence_length):        #데이터를 seqeunce_length 수 만큼 묶는 함수
    x_seq = []
    y_seq = []
    for i in range(len(x) - sequence_length):
        x_seq.append(x[i:i + sequence_length])
        y_seq.append(y[i + sequence_length])

    return torch.FloatTensor(x_seq).to(device), torch.FloatTensor(y_seq).to(device).view(-1, 1)


split = 200
#split = int(0.8 * len(X))  로 해야함 지금은 test용이라 10으로
sequence_length = 10
x_seq, y_seq = seq_data(X, Y, sequence_length)

#print("x_seq:", len(x_seq))     전체 개수 - sequence_length가 나옴
#print("y_seq:", len(y_seq))     전체 개수 - sequence_length가 나옴

x_train_seq = x_seq[:split]
y_train_seq = y_seq[:split]
x_test_seq = x_seq[split:]
y_test_seq = y_seq[split:]

#print("x_train_seq", len(x_train_seq))
#print("y_train_seq", len(y_train_seq))
#print("x_test_seq", len(x_test_seq))
#print("y_test_seq", len(y_test_seq))
#print("total num", len(x_train_seq)+len(x_test_seq))

#Dataset 은 sample 과 정답(label)을 저장
train = torch.utils.data.TensorDataset(x_train_seq, y_train_seq)
test = torch.utils.data.TensorDataset(x_test_seq, y_test_seq)

# print("train:", len(train))      200
# print("test", len(test))         1722

#DataLoader는 Dataset을 샘플에 쉽게 접근할 수 있도록 순회 가능한 객체로 함
trainloader = torch.utils.data.DataLoader(dataset=train, batch_size=16, shuffle=True)      #shuffle 쓰면 데이터를 섞어서 분할 -> 여기선 왜 True?
testloader = torch.utils.data.DataLoader(dataset=test, batch_size=16)

#print("trainloader", len(trainloader))           trainloader/batch_size 값이 나온다
#print("testloader",len(testloader))              testloader/batch_size 값이 나온다


#print("1: ",x_train_seq.size(0))   split
#print("2: ",x_train_seq.size(1))   sequence_length
#print("3: ",x_train_seq.size(2))   input_size

input_size = x_train_seq.size(2)     #x_train_seq.size() -> input_size
num_layers = 2                       #lstm layer 개수
hidden_size = 8



### input size = 입력 피처의 개수 / hidden size = 은닉층의 피처 개수 / num_layers = LSTM layer를 몇층으로 쌓을지 / bias = bias 여부 / dropout = dropout 비율
class LSTM(nn.Module):                                                                 # Module class는 인공신경망의 기본 클래스
    def __init__(self, input_size, hidden_size, sequence_length, num_layers, device):  # 신경망에 사용될 계층(Layer) 초기화
        super(LSTM, self).__init__()
        self.device = device
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size * sequence_length, 1)
        self.relu = nn.ReLU()            #내가 임의로 추가

    def forward(self, x):                                                               # forward -> 순방향 메서드, 모델이 데이터 입력받아 학습하는 과정 정의
        h0 = torch.zeros(self.num_layers, x.size()[0], self.hidden_size).to(self.device)
        c0 = torch.zeros(self.num_layers, x.size()[0], self.hidden_size).to(self.device)
        out, _ = self.lstm(x, (h0, c0))
        out = out.reshape(out.shape[0], -1)
        out = self.relu(out)   #내가 임의로 추가
        out = self.fc(out)
        return out

    # Jamin LSTM
    #def forward(self, x):
        #h_0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size))
        #c_0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size))

        #output, (hn, cn) = self.lstm(x, (h_0, c_0))
        #hn = hn.view(-1, self.hidden_size)
        #out = self.relu(hn)
        #out = self.fc_1(out)
        #out = self.relu(out)
        #out = self.fc(out)
        #return out

model = LSTM(input_size=input_size,
             hidden_size=hidden_size,
             num_layers=num_layers,
             sequence_length=sequence_length,
             device=device).to(device)


learning_rate = 0.0001
num_epochs = 100
#learning_rate = 1e-3         기존 참고 learning_rate
criterion = nn.MSELoss()                                           # MSE 사용
optimizer = optim.Adam(model.parameters(), lr=learning_rate)       # Adam 사용

n = len(trainloader)
loss_graph = []

for epoch in range(num_epochs):
    running_loss = 0.0
    for data in trainloader:
        seq, target = data        # x,y

        outputs = model(seq)       #model.forward()랑 그냥이랑 무슨차이 -> 그냥 model이 더 좋은듯
        optimizer.zero_grad()
        loss = criterion(outputs, target)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    loss_graph.append(running_loss / n)
    if epoch % 100 == 0:
        print("Epoch : %d loss : %.4f" % (epoch, running_loss / n))



concatdata = torch.utils.data.ConcatDataset([train, test])
data_loader = torch.utils.data.DataLoader(concatdata, batch_size=16)


# https://076923.github.io/posts/Python-pytorch-9/ 참고하면 테스트 데이터 정의해서 모델 평가하는 것도 확인 가능 -> 입력과 차원을 같게
with torch.no_grad():           #gradient 옵션을 그만할 때 사용, 보통 더이상 학습 안하고 학습된 모델로 결과를 볼 때 사용
    pred = []
    model.eval()                #evaluation 과정에서 사용하지 않아도 되는 layer들을 off 시켜줌
    for data in data_loader:
        seq, target = data
        out = model(seq)
        pred += out.cpu().tolist()

pred = scaler2.inverse_transform(pred)
Y = scaler2.inverse_transform(Y)
length = len(pred)
#print(Y[sequence_length:])
#print(data['height'][sequence_length:].values)


print("pred: ", len(pred))
print("Y: ", len(Y))
count = 0


#실제값 확인
#for i in range(length-split-sequence_length):
#    print(Y_real[split+i+sequence_length][0])

#예측값 확인
for i in range(length-split-sequence_length):
    print(pred[split+i][0])

for i in range(length-split-sequence_length):
    if round(pred[split+i][0]) == (Y_real[split+i+sequence_length][0]):   # Y_real[split+i+sequence_length에 주목!!
        count = count+1


#print("Y 비교값 수 확인", len(Y[split+sequence_length:]))
#print("pred 비교값 수 확인", len(pred[split:]))

plt.figure(figsize=(20, 10))
plt.plot(np.ones(100) * len(train), np.linspace(0, 5, 100), '-', linewidth=0.6)
plt.plot(Y[split+sequence_length:], 'r', linewidth=1.2)
plt.plot(pred[split:], 'b', linewidth=0.6)
plt.legend(['hi'])
plt.show()

#length : 전체 개수 - len - split
print("count : ", count, "length:", length-split)
print("Accuracy: ", count/(length-split))

#test
#전체 27047
#split 200 sequence 10