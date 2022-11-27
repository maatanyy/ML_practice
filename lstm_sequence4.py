# SMOTE 적용
# 원핫인코딩 적용 안함
# Y transform 안함
# 결과가 나오는데 반올림하면 전부 유지 그래서 원핫 인코딩을 하는 게 좋을 것 같다.

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from imblearn.over_sampling import SMOTE


# 엑셀 파일을 읽어옴
data = pd.read_excel('2weeks_summary2.xlsx')

# 변수
# ID	height	weight	step_count	burn_calorie	eat_calorie	sleep	bmi	bmi_target


# MinMaxScaler()로 정규화
scaler = MinMaxScaler()
data[['Weight','BMI','Step','Burn','Eat','Sleep']] = scaler.fit_transform(data[['Weight','BMI','Step','Burn','Eat','Sleep']])


# Y_real 에는 마지막 원래 라벨값 넣어둠 -> 예측된 값과 비교하기 위해 쓰임
Y_real = data[['Label']]
Y_real = Y_real.to_numpy()

#Y_real = pd.get_dummies('Label')
#print(Y_real)

# MinMaxScaler()로 정규화
#scaler2 = MinMaxScaler()
#data[['Label']] = scaler2.fit_transform(data[['Label']])


# 이유는 모르겠는데 라벨 타입을 인트로 바꿈 (에러 해결하기 위해 이랬음)
X = data[['Weight','BMI','Step','Burn','Eat','Sleep']].values
data = data.astype({'Label':'int'})
Y = data[['Label']].values


input_size = 6        # input_size, 입력 변수의 개수
print(input_size)

# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")


# seq_length 만큼 데이터를 묶어주는 함수임
def seq_data(x, y, sequence_length):
    x_seq = []
    y_seq = []
    for i in range(len(x) - sequence_length):
        x_seq.append(x[i:i + sequence_length])
        y_seq.append(y[i + sequence_length])

    return torch.FloatTensor(x_seq).to(device), torch.FloatTensor(y_seq).to(device).view(-1, 1)




split = 1600           # split 개수 만큼 train이 된다
# split = int(0.8 * len(x_seq))
sequence_length = 6    # 함께 묵을 날짜 수


# seq_data() 함수를 통해 데이터를 묶은 후 각각 x_seq, y_seq에 넣어 줌
x_seq, y_seq = seq_data(X, Y, sequence_length)


# train / test로 쪼갬
x_train_seq = x_seq[:split]
y_train_seq = y_seq[:split]
x_test_seq = x_seq[split:]
y_test_seq = y_seq[split:]


# shape 확인용 출력
# print(x_train_seq.shape)
# print(y_train_seq.shape)
# print(x_test_seq.shape)
# print(y_test_seq.shape)


# SMOTE 사용하기 위해 차원을 바꿔 줌
nsamples, nx, ny = x_train_seq.shape
# print("nsamples :", nsamples)
# print("nx :", nx)
# print("ny :", ny)

x_train_seq = x_train_seq.reshape((nsamples, nx*ny))
# print('Number of x_seq :', len(x_seq))
# print('Number of y_seq :', len(y_seq))

# SOMTE 적용 전 shape 확인용
print('Before SMOTE OverSampling, the shape of x: {}'.format(x_seq.shape))
print('Before SMOTE OverSampling, the shape of y: {} \n'.format(y_seq.shape))

# print(x_train_seq.shape)
# print(y_train_seq.shape)


# SMOTE 기법 사용하여 train 늘려 줌
sm = SMOTE(random_state=0)
x_smote, y_smote = sm.fit_resample(x_train_seq, y_train_seq)


# SMOTE 적용 후 shape 확인용
print('After SMOTE OverSampling, the shape of x: {}'.format(x_smote.shape))       # train만 SMOTE 적용 (x)
print('After SMOTE OverSampling, the shape of y: {} \n'.format(y_smote.shape))    # train만 SMOTE 적용 (y)


# 차원 바꾸는 과정 Tensor로 바꿨다가 reshape 함
x_smote = torch.Tensor(x_smote)
tempx, tempx2  = x_smote.shape
x_smote = x_smote.reshape((tempx, sequence_length, input_size))

y_smote = torch.Tensor(y_smote)
tempy, = y_smote.shape
y_smote = y_smote.reshape((tempy, 1))

# shape 확인용
# print("y_smote.shape: ", y_smote.shape)
# print("y_test_seq.shape: ", y_test_seq.shape)

# TensorDataset에 넣어줌
train = torch.utils.data.TensorDataset(x_smote, y_smote)
test = torch.utils.data.TensorDataset(x_test_seq, y_test_seq)


# 길이 확인용 train은 smote 한 후 길이가 늘어나 있고
# test는 smote를 적용안해서 전체 길이 - seq_length가 들어가 있음
print("len(train): ", len(train))
print("len(test): ", len(test))


# trainloader 와 testloader에 각각 train과 test셋을 넣어 줌
trainloader = torch.utils.data.DataLoader(dataset=train, batch_size=6, shuffle=True)
testloader = torch.utils.data.DataLoader(dataset=test, batch_size=6)



num_layers = 2          # lstm 층의 수, Number of recurrent layers
#setting num_layers=2 would mean stacking two LSTMs together to form stacked LSTM, with the second LSTM taking in outputs of the first LSTM and computing the final results
hidden_size = 12         # 은닉층의 피처 개수

### input size = 입력 피처의 개수 / hidden size = 은닉층의 피처 개수 / num_layers = LSTM layer를 몇층으로 쌓을지 / bias = bias 여부 / dropout = dropout 비율
#  https://stackoverflow.com/questions/53475803/understanding-the-softmax-output-layer-of-rnn          softmax부분 구조 참고
#  https://dhpark1212.tistory.com/entry/RNN-LSTM-%EC%84%A4%EB%AA%85-%EB%B0%8F-%EA%B5%AC%ED%98%84pytorch


class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, sequence_length, num_layers, device):
        super(LSTM, self).__init__()
        self.device = device
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm1 = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)           # batch first 가 True, input and output tensors are provided as (batch, seq, feature)
        self.fc = nn.Linear(hidden_size * sequence_length, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size()[0], self.hidden_size).to(self.device)
        c0 = torch.zeros(self.num_layers, x.size()[0], self.hidden_size).to(self.device)
        out, _ = self.lstm1(x, (h0, c0))
        out = out.reshape(out.shape[0], -1)
        out = self.fc(out)
        out = self.relu(out)
        return out


# 모델 정의
model = LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, sequence_length=sequence_length, device=device).to(device)

print(model)

criterion = nn.MSELoss()    # MSE 손실함수 사용
optimizer = optim.Adam(model.parameters(), lr=1e-3)   # Adam optimizer 사용



n = len(trainloader)    # n에 trainloader 길이 넣어 줌
loss_graph = []         # 손실값 구하는 데 사용할 배열


# epoch 만큼 반복하며 loss 구하며 최적화
for epoch in range(100):
    running_loss = 0.0
    for data in trainloader:
        seq, target = data
        outputs = model(seq)   # model.forward()랑 그냥이랑 무슨차이 -> 그냥 model이 더 좋은듯
        optimizer.zero_grad()
        loss = criterion(outputs, target)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    loss_graph.append(running_loss / n)
    #if epoch % 100 == 0:
    print("Epoch : %d loss : %.4f" % (epoch, running_loss / n))


concatdata = torch.utils.data.ConcatDataset([test])
data_loader = torch.utils.data.DataLoader(concatdata)

#print(len(data_loader))


# https://076923.github.io/posts/Python-pytorch-9/ 참고하면 테스트 데이터 정의해서 모델 평가하는 것도 확인 가능 -> 입력과 차원을 같게
with torch.no_grad():   # gradient 옵션을 그만할 때 사용, 보통 더이상 학습 안하고 학습된 모델로 결과를 볼 때 사용
    pred = []
    model.eval()        # evaluation 과정에서 사용하지 않아도 되는 layer들을 off 시켜줌
    for data in data_loader:
        seq, target = data
        out = model(seq)
        pred += out.cpu().tolist()


print(pred)
#pred = scaler2.inverse_transform(pred)
#Y = scaler2.inverse_transform(Y)
length = len(pred)

#print(Y[sequence_length:])
#print(data['height'][sequence_length:].values)


#print("pred:", len(pred))
#print("Y[split:]", len(Y[split+sequence_length:]))
count = 0


forTestLength = len(X)-sequence_length-split
#print("Num of Test : ", forTestLength)


#print("pred:", len(pred))
#print("Y_real:", len(Y_real))


for i in range(forTestLength):
     if round(pred[i][0]) == round(Y_real[forTestLength+i][0]):
         count = count+1


#for i in range(forTestLength):    예측 라벨 확인
#     print(pred[i][0])

#for i in range(forTestLength):    실제 라벨 값 확인
#     print(Y_real[forTestLength+i][0])

plt.figure(figsize=(20, 10))
plt.title("BMI prediction")
plt.plot(Y[split+sequence_length:], 'r', label='real value')          # 빨간색이 실제
plt.plot(pred[:], 'b', linewidth=0.6, label='prediction')             # 파란색이 예측
plt.legend()                                                          # 범례 적용
plt.show()


print("count : ", count, "length : ", forTestLength)
print("Accuracy: ", count/forTestLength)



