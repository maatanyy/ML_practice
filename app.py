#2022-06-21
import pandas as pd
import numpy as np
import tensorflow as tf

data = pd.read_csv('gpascore.csv')
print(data.isnull().sum())
data.isnull().sum() #빈칸세주기

data = data.dropna()  # 빈칸데이터 행삭제
#print(data['gpa'].min())  # .mix  .count  .max 같은 유용한 함수들

y데이터 = data['admit'].values  #리스트로 담아줌
x데이터 = []

for i, rows in data.iterrows():  #dataf라는 데이터프레임을 가로 한줄씩 출력해주라
   x데이터.append([ rows['gre'], rows['gpa'], rows['rank'] ])

model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(64, activation ='tanh'),
    tf.keras.layers.Dense(128, activation ='tanh'),
    tf.keras.layers.Dense(1, activation ='sigmoid'),
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'] )

model.fit( np.array(x데이터), np.array(y데이터), epochs=1000)  #epochs 학습시킬 횟수


#예측
예측값 = model.predict( [[750, 3.70, 3], [400, 2.2, 1]])
print(예측값)






