import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.python.keras.models import Sequential
from keras.layers import LSTM
from tensorflow.python.keras.layers import Dense,Dropout,Activation
from tensorflow.python.keras import metrics
from sklearn.preprocessing import StandardScaler
from tensorflow.python import keras
import tensorflow as tf
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE
from sklearn.metrics import f1_score ## F1 Score 구하기
from sklearn.metrics import precision_recall_fscore_support as sk
import keras.backend as K

data = pd.read_excel('Continous_2weeks_2day_1term.xlsx')

X = data.iloc[:, [1, 3, 4, 5, 6, 7]]
y = data.iloc[:, [-1]]

scaler = StandardScaler()
X = scaler.fit_transform(X)
X = pd.DataFrame(X)

#결과 넣을 배열
Result = [[0 for j in range(4)] for i in range(10)]
pred_list = []

#한 사람당 데이터 수
Count_1 = int(181*0.1)
#한 사람당 데이터 수
Count_2 = 181

X_test = pd.DataFrame()
X_train = pd.DataFrame()
y_test = pd.DataFrame()
y_train = pd.DataFrame()
empty = pd.DataFrame()

#결과 넣을 배열
Result = [[0 for j in range(4)] for i in range(10)]

K.clear_session()
model = Sequential()
model.add(LSTM(6, input_shape=(6,1)))
model.add(Dense(1, activation='softmax'))
model.compile(loss=tf.keras.losses.CategoricalCrossentropy(),
              optimizer='adam',
              metrics=['accuracy'])



for i in range(10):
    X_test = empty
    X_train = empty
    y_test = empty
    y_train = empty
    y_test_list = []
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
    X_test, y_test = smote.fit_resample(X_test, y_test)
    print('SMOTE 적용 후 학습용 피처/레이블 데이터 세트: ', X_test.shape, y_test.shape)
    print('SMOTE 적용 후 Train 레이블 값 분포: \n', y_train.value_counts())
    print('SMOTE 적용 후 Test 레이블 값 분포: \n', y_test.value_counts())

    X_train = X_train.values
    X_test = X_test.values
    y_train = y_train.values
    y_test = y_test.values

    X_train = X_train.reshape(X_train.shape[0], 6, 1)
    X_test = X_test.reshape(X_test.shape[0], 6, 1)

    # 원핫인코딩
    # 예시 : 1 , 2 -> (1,0) , (0,1)

    print(y_train)
    #y_train = pd.get_dummies(y_train[0])
    #y_test = pd.get_dummies(y_test[0])

    history = model.fit(X_train, y_train, epochs=100, validation_data=(X_test,y_test), batch_size=18)
    predicted = model.predict(X_test)
    results = model.evaluate(X_test, y_test, batch_size=8)
    print("[{}]Accuracy : {}".format(i, results[1]))

    predicted = pd.DataFrame(predicted)
    predicted = predicted.idxmax(axis=1)
    y_test = y_test.idxmax(axis=1)

    print("predicted", predicted)
    print("y_test", y_test)

    # f1score
    f1 = f1_score(y_test, predicted, average='weighted')
    print("[{}]F1score : {}".format(i, f1))
    # precision/recall
    p_rlist = sk(y_test, predicted, average='weighted')
    print("[{}]Precision : {}".format(i, p_rlist[0]))
    print("[{}]Recall : {}".format(i, p_rlist[1]))
    print()
    # 결과 배열에 넣기
    Result[i][0] = results[1]
    Result[i][1] = f1
    Result[i][2] = p_rlist[0]
    Result[i][3] = p_rlist[1]
    del results
    del f1
    del p_rlist

Result_df=pd.DataFrame(Result,columns=['Accuracy','F1-Score','Precision','Recall'])
Result_df
print("Average of Accuracy {}".format(Result_df['Accuracy'].mean()))
print("Average of F1-Score {}".format(Result_df['F1-Score'].mean()))
print("Average of Precision {}".format(Result_df['Precision'].mean()))
print("Average of Recall {}".format(Result_df['Recall'].mean()))
Matrix=pd.DataFrame(Result_df['Accuracy'],columns=['Accuracy'])
Matrix['Accuracy']=Result_df['Accuracy']
A=[Result_df['Accuracy'].mean(),Result_df['F1-Score'].mean(),Result_df['Precision'].mean(),Result_df['Recall'].mean()]
A=pd.DataFrame(A,columns=['Accuracy'])
Matrix=pd.concat([Matrix,A])
Matrix=Matrix.transpose()
Matrix.to_excel('./PFMatrix2.xlsx')

