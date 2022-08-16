# 출처 : https://eunhye-zz.tistory.com/entry/Pytorch%EB%A5%BC-%ED%99%9C%EC%9A%A9%ED%95%9C-Timeseries-%EC%98%88%EC%B8%A1-%EB%AA%A8%EB%8D%B81-LSTM#google_vignette
# 주가 예측 모델 LSTM
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import TensorDataset    #텐서데이터셋
from torch.utils.data import DataLoader       #데이터로더

                                              # 데이터 불러오기         -> 데이터 불러와서 전처리 해야함
df = pd.read_csv('./data-02-stock_daily.csv')
bmiData = pd.read_csv('bimtestyo.csv')
bmiData['ID'] = pd.to_numeric(bmiData['ID'], errors='coerce')    #object 타입을 에러 없애고 float으로 변환
bmiData['ID'] = bmiData['ID'].astype(float)
bmiData['birth'] = pd.to_numeric(bmiData['birth'], errors='coerce')    #object 타입을 에러 없애고 float으로 변환
bmiData['birth'] = bmiData['birth'].astype(float)

bmiData = bmiData.drop(columns=['ID','isSupporter','birth'])         # gender, grade, birth, height, weight, bmi
pd.set_option('mode.chained_assignment',  None)          #pandas 경고 끄기 옵션    #https://blog.naver.com/PostView.nhn?blogId=wideeyed&logNo=221817400937 참고
#print(bmiData)
#print(df)

seq_length = 7                  #데이터 양 date 수 넣으면 됨
batch = 16                          #배치 사이즈는 임의로 지정

# 데이터를 역순으로 정렬하여 전체 데이터의 70% 학습, 30% 테스트에 사용
bmiData = bmiData[::-1]
#print(bmiData)
train_size = int(len(bmiData)*0.7)
train_set = bmiData[0:train_size]
test_set = bmiData[train_size-seq_length:]

#print(train_set)
#print(test_set)

# Input scale                           #데이터 스케일링, 각 칼럼을 0-1 사이의 값으로 스케일링
scaler_x = MinMaxScaler()               #MinMaxScaler 사용
scaler_x.fit(train_set.iloc[:, :-1])

train_set.iloc[:, :-1] = scaler_x.transform(train_set.iloc[:, :-1])
test_set.iloc[:, :-1] = scaler_x.transform(test_set.iloc[:, :-1])

# Output scale
scaler_y = MinMaxScaler()
scaler_y.fit(train_set.iloc[:, [-1]])

train_set.iloc[:, -1] = scaler_y.transform(train_set.iloc[:, [-1]])
test_set.iloc[:, -1] = scaler_y.transform(test_set.iloc[:, [-1]])


# 확인용 print(train_set.iloc[:, -1])
# 확인용 print(test_set.iloc[:, -1])

# 데이터셋 생성 함수                             #파이토치에서는 3D 텐서의 입력을 받으므로 torch.FloatTensor를 사용하여 np.arrary 형태에서 tensor 형태로 바꿔준다.
def build_dataset(time_series, seq_length):
    dataX = []
    dataY = []
    for i in range(0, len(time_series)-seq_length):
        _x = time_series[i:i+seq_length, :]
        _y = time_series[i+seq_length, [-1]]
        #print(_x, "-->",_y)
        dataX.append(_x)
        dataY.append(_y)

    return np.array(dataX), np.array(dataY)

#print(np.array(train_set))
trainX, trainY = build_dataset(np.array(train_set), seq_length)
testX, testY = build_dataset(np.array(test_set), seq_length)

# 텐서로 변환
trainX_tensor = torch.FloatTensor(trainX)
trainY_tensor = torch.FloatTensor(trainY)

testX_tensor = torch.FloatTensor(testX)
testY_tensor = torch.FloatTensor(testY)

# 텐서 형태로 데이터 정의
dataset = TensorDataset(trainX_tensor, trainY_tensor)
# 데이터로더는 기본적으로 2개의 인자를 입력받으며 배치크기는 통상적으로 2의 배수를 사용
dataloader = DataLoader(dataset,
                        batch_size=batch,
                        shuffle=True,
                        drop_last=True)

# 설정값
data_dim = 5                                                       #입력 칼럼 수정해줘야함!!!!!!
hidden_dim = 10                                                    #히든 스테이트 계속 바꿔봐야함!!!!
output_dim = 1                                                     #output 1개
learning_rate = 0.01                                               #학습률 0.01
nb_epochs = 1000                                                #에폭
device = torch.device("cpu")
#device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Net(nn.Module):
    # # 기본변수, layer를 초기화해주는 생성자
    def __init__(self, input_dim, hidden_dim, seq_len, output_dim, layers):
        super(Net, self).__init__()
        self.hidden_dim = hidden_dim
        self.seq_len = seq_len
        self.output_dim = output_dim
        self.layers = layers

        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=layers,
                            # dropout = 0.1,
                            batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim, bias=True)

        # 학습 초기화를 위한 함수

    def reset_hidden_state(self):
        self.hidden = (
            torch.zeros(self.layers, self.seq_len, self.hidden_dim),
            torch.zeros(self.layers, self.seq_len, self.hidden_dim))

    # 예측을 위한 함수
    def forward(self, x):
        x, _status = self.lstm(x)
        x = self.fc(x[:, -1])
        return x


def train_model(model, train_df, num_epochs=None, lr=None, verbose=10, patience=10):
    criterion = nn.MSELoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    nb_epochs = num_epochs

    # epoch마다 loss 저장
    train_hist = np.zeros(nb_epochs)

    for epoch in range(nb_epochs):
        avg_cost = 0
        total_batch = len(train_df)

        for batch_idx, samples in enumerate(train_df):
            x_train, y_train = samples

            # seq별 hidden state reset
            model.reset_hidden_state()

            # H(x) 계산
            outputs = model(x_train)

            # cost 계산
            loss = criterion(outputs, y_train)

            # cost로 H(x) 개선
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            avg_cost += loss / total_batch

        train_hist[epoch] = avg_cost

        if epoch % verbose == 0:
            print('Epoch:', '%04d' % (epoch), 'train loss :', '{:.4f}'.format(avg_cost))

        # patience번째 마다 early stopping 여부 확인
        #if (epoch % patience == 0) & (epoch != 0):

            # loss가 커졌다면 early stop
            #if train_hist[epoch - patience] < train_hist[epoch]:
                #print('\n Early Stopping')

                #break

    return model.eval(), train_hist   # model.eval()을 사용하여 evaluation 과정에서 사용되지 말아야할 layer들을 알아서 꺼주는 함수다.

# 모델 학습
net = Net(data_dim, hidden_dim, seq_length, output_dim, 1).to(device)
model, train_hist = train_model(net, dataloader, num_epochs = nb_epochs, lr = learning_rate, verbose = 100, patience = 50)

#print(net)   모델확인

# epoch별 손실값
#fig = plt.figure(figsize=(10, 4))
#plt.plot(train_hist, label="Training loss")
#plt.legend()
#plt.show()

# 모델 저장
PATH = "./Timeseries_LSTM_bmi_.pth"
torch.save(model.state_dict(), PATH)

# 불러오기
model = Net(data_dim, hidden_dim, seq_length, output_dim, 1).to(device)
model.load_state_dict(torch.load(PATH), strict=False)
model.eval()

# 예측 테스트
with torch.no_grad():
    pred = []
    for pr in range(len(testX_tensor)):

        model.reset_hidden_state()

        predicted = model(torch.unsqueeze(testX_tensor[pr], 0))
        predicted = torch.flatten(predicted).item()
        pred.append(predicted)

    # INVERSE
    pred_inverse = scaler_y.inverse_transform(np.array(pred).reshape(-1, 1))
    testY_inverse = scaler_y.inverse_transform(testY_tensor)

def MAE(true, pred):
    return np.mean(np.abs(true-pred))

print('MAE SCORE : ', MAE(pred_inverse, testY_inverse))
print(pred_inverse-testY_inverse)
#print(testY_inverse)

fig = plt.figure(figsize=(8,3))
plt.plot(np.arange(len(pred_inverse)), pred_inverse, label = 'pred')
plt.plot(np.arange(len(testY_inverse)), testY_inverse, label = 'true')
plt.title("Loss plot")
plt.show()

