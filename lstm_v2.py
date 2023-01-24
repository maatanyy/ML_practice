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
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import TensorDataset, DataLoader
from imblearn.over_sampling import SMOTE
import torch.nn.functional as F
from sklearn.metrics import precision_recall_fscore_support as sk
from sklearn.metrics import f1_score  ## F1 Score 구하기
from sklearn.metrics import accuracy_score

# 엑셀 파일을 읽어옴
data = pd.read_excel('Continous_2weeks_2day_1term.xlsx')

# StandardScaler()로 정규화
scaler = StandardScaler()

data[['Height', 'Weight', 'Step', 'Burn', 'Eat', 'Sleep']] = scaler.fit_transform(data[['Height', 'Weight', 'Step', 'Burn', 'Eat', 'Sleep']])
X = data[['Height','Weight','Step','Burn','Eat','Sleep']].values

y = data[['Label']].values
sequence_length = 6    # 함께 묵을 날짜 수

def seq_data(x, y, sequence_length):
    x_seq = []
    y_seq = []
    for i in range(len(x) - sequence_length):
        x_seq.append(x[i:i + sequence_length])
        y_seq.append(y[i + sequence_length])

    return np.array(x_seq), np.array(y_seq)


# seq_data() 함수를 통해 데이터를 묶은 후 각각 x_seq, y_seq에 넣어 줌
X, y = seq_data(X, y, sequence_length)


EPOCHS = 100
input_size = 6        # input_size, 입력 변수의 개수
print(input_size)


#결과 넣을 배열
Result = [[0 for j in range(4)] for i in range(10)]
Count = int(322/10)*83
pred_list = []
Device = torch.device("cpu")

class LSTM(nn.Module):
    def __init__(self, input_size, num_classes, hidden_size, num_layers, device):
        super(LSTM, self).__init__()
        self.device = device
        self.input_size = input_size  # input size
        self.num_classes = num_classes  # number of classes
        self.hidden_size = hidden_size  # hidden state
        self.num_layers = num_layers  # number of layers

        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                            num_layers=num_layers, batch_first=True)  # lstm
        self.fc_1 = nn.Linear(hidden_size, 128)  # fully connected 1
        self.fc = nn.Linear(128, num_classes)  # fully connected last layer
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        h_0 = torch.zeros(self.num_layers, x.size()[0], self.hidden_size).to(self.device)
        c_0 = torch.zeros(self.num_layers, x.size()[0], self.hidden_size).to(self.device)
        # Propagate input through LSTM
        output, (hn, cn) = self.lstm(x, (h_0, c_0))  # lstm with input, hidden, and internal state
        hn = hn.view(-1, self.hidden_size)  # reshaping the data for Dense layer next
        out = self.relu(hn)
        out = self.fc_1(out)  # first Dense
        out = self.relu(out)  # relu
        out = self.fc(out)  # Final Output
        out = self.softmax(out)
        return out


num_layers = 1  # lstm 층의 수, Number of recurrent layers
# setting num_layers=2 would mean stacking two LSTMs together to form stacked LSTM, with the second LSTM taking in outputs of the first LSTM and computing the final results
hidden_size = 6  # 은닉층의 피처 개수
num_classes = 3  # number of output classes


criterion = nn.MSELoss()  # MSE 손실함수 사용


def train(model, trainloader, optimizer):
    model.train()
# epoch 만큼 반복하며 loss 구하며 최적화
    for batch_idx, (data, target) in enumerate(trainloader):
        # 학습 데이터를 DEVICE의 메모리로 보냄
        data, target = data.to(Device), target.to(Device)
        # 매 반복(iteration) 마다 기울기를 계산하기 위해 zero_grad() 호출
        optimizer.zero_grad()
        # 실제 모델의 예측값(output) 받아오기
        output = model(data)
        # 정답 데이터와의 CrossEntropyLoss 계산
        # 손실함수에 output, label 값을 대입하여 손실을 계산합니다.
        loss = F.cross_entropy(output, target)
        # 기울기 계산
        loss.backward()
        # 계산된 Gradient를 업데이트 합니다.
        optimizer.step()



def evaluate(model, data_loader):
    model.eval()
    test_loss = 0
    correct = 0
    pred_list = []
    pred_array = []
    with torch.no_grad():  # 평가 과정에서는 기울기를 계산하지 않으므로, no_grad명시
        for data, target in data_loader:
            data, target = data.to(Device), target.to(Device)
            output = model(data)

            _, pred = output.max(dim=1)
            pred_array = pred.tolist()
            pred_list += pred_array
            # confusion matrix를 위해 pred 리턴 값

            # 모든 오차 더하기
            test_loss += F.cross_entropy(output, target, reduction="sum").item()

            # 가장 큰 값을 가진 클래스가 모델의 예측입니다.
            # 예측 클래스(pred)과 정답 클래스를 비교하여 일치할 경우 correct에 1을 더합니다.
            pred = output.max(1, keepdim=True)[1]
            # eq() 함수는 값이 일치하면 1을, 아니면 0을 출력.
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(data_loader.dataset)
    # 정확도 계산
    test_accuracy = 100. * correct / len(data_loader.dataset)
    return test_loss, test_accuracy, pred_list


#한 사람당 데이터 수
Count_1 = int(83*0.1)
#한 사람당 데이터 수
Count_2 = 83

X_test = pd.DataFrame()
X_train = pd.DataFrame()
y_test = pd.DataFrame()
y_train = pd.DataFrame()
empty = pd.DataFrame()

#결과 넣을 배열
Result = [[0 for j in range(4)] for i in range(10)]

for i in range(10):
    # 모델 정의
    model = LSTM(input_size=input_size, num_classes=num_classes, hidden_size=hidden_size, num_layers=num_layers,device=Device).to(Device)
    # 옵티마이저를 정의합니다. 옵티마이저에는 model.parameters()를 지정해야 합니다.
    optimizer = optim.Adam(model.parameters(), lr=0.01)    # Adam optimizer 사용
    # 손실함수(loss function)을 지정합니다. Multi-Class Classification 이기 때문에 CrossEntropy 손실을 지정하였습니다.
    loss_fn = nn.CrossEntropyLoss()

    X_test = empty
    X_train = empty
    y_test = empty
    y_train = empty

    y_test_list = []


    ##########################    차원 바꾸는 과정
    nsamples, nx, ny = X.shape
    X = X.reshape((nsamples, nx * ny))

    X = pd.DataFrame(X)
    y = pd.DataFrame(y)
    ##########################    차원 바꾸는 과정

    for j in range(322):
        X_temp_test = X.iloc[Count_2 * j + Count_1 * i:Count_2 * j + Count_1 * (i + 1)]
        X_test = pd.concat([X_test, X_temp_test])
        X_temp_train = X.iloc[Count_2 * j + Count_1:Count_2 * (j + 1)]
        X_train = pd.concat([X_train, X_temp_train])
        y_temp_test = y.iloc[Count_2 * j + Count_1 * i:Count_2 * j + Count_1 * (i + 1)]
        y_test = pd.concat([y_test, y_temp_test])
        y_temp_train = y.iloc[Count_2 * j + Count_1:Count_2 * (j + 1)]
        y_train = pd.concat([y_train, y_temp_train])

    print('SMOTE 적용 전 Train 레이블 값 분포: \n', y_train.value_counts())
    print('SMOTE 적용 전 Test 레이블 값 분포: \n', y_test.value_counts())



    # SMOTE 적용
    smote = SMOTE(random_state=0)
    X_train, y_train = smote.fit_resample(X_train, y_train)
    #     X_test,y_test = smote.fit_resample(X_test,y_test)

    print('SMOTE 적용 후 학습용 피처/레이블 데이터 세트: ', X_test.shape, y_test.shape)
    print('SMOTE 적용 후 Train 레이블 값 분포: \n', y_train.value_counts())
    print('SMOTE 적용 후 Test 레이블 값 분포: \n', y_test.value_counts())


    print(X_train)
    X_train = torch.FloatTensor(X_train.to_numpy())
    tempx, tempx2 = X_train.shape
    X_train = X_train.reshape((tempx, sequence_length, input_size))


    # 모든 데이터 torch로 변환

    #X_train = torch.FloatTensor(X_train.to_numpy())
    X_test = torch.FloatTensor(X_test.to_numpy())
    print("X_test", len(X_test))
    y_train = y_train.to_numpy()
    y_train = np.ravel(y_train, order='C')
    y_train = torch.LongTensor(y_train)
    y_test = y_test.to_numpy()
    y_test = np.ravel(y_test, order='C')
    y_test = torch.LongTensor(y_test)
    print("y_test", y_test)
    # train_dataset, test_dataset을 구별하여 정의
    train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
    test_dataset = torch.utils.data.TensorDataset(X_test, y_test)
    train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=16, shuffle=False)


    for epoch in range(1, EPOCHS + 1):
        train(model, train_dataloader, optimizer)
        test_loss, test_accuracy, predict = evaluate(model, test_dataloader)

        print('[{}] Test Loss: {:.4f}, Accuracy: {:.2f}%'.format(epoch, test_loss, test_accuracy))

    #     print("[{}]Predict : {}".format(i,predict))
    # Accuracy
    # test label 데이터 토치에서 list로 변경
    y_test_list = y_test.tolist()

    accuracy = accuracy_score(y_test_list, predict) * 100
    print("[{}]Accuracy : {}".format(i, accuracy))
    # f1score
    f1 = f1_score(y_test_list, predict, average='weighted')
    print("[{}]F1score : {}".format(i, f1))
    # precision/recall
    p_rlist = sk(y_test_list, predict, average='weighted')
    print("[{}]Precision : {}".format(i, p_rlist[0]))
    print("[{}]Recall : {}".format(i, p_rlist[1]))
    print()
    # 결과 배열에 넣기
    Result[i][0] = accuracy
    Result[i][1] = f1
    Result[i][2] = p_rlist[0]
    Result[i][3] = p_rlist[1]
    del accuracy
    del f1
    del p_rlist

Result_df=pd.DataFrame(Result,columns=['Accuracy','F1-Score','Precision','Recall'])
Result_df

Matrix=pd.DataFrame(Result_df['Accuracy'],columns=['Accuracy'])
Matrix['Accuracy']=Result_df['Accuracy']
A=[Result_df['Accuracy'].mean(),Result_df['F1-Score'].mean(),Result_df['Precision'].mean(),Result_df['Recall'].mean()]
A=pd.DataFrame(A,columns=['Accuracy'])
Matrix=pd.concat([Matrix,A])
Matrix=Matrix.transpose()
Matrix.to_excel('./PFMatrix2.xlsx')

