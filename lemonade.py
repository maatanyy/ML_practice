# 생활코딩 Tensorflow(python)을 참고하였습니다.
import pandas as pd
import tensorflow as tf


#과거의 데이터 준비하는 부분
filepath1 = 'https://raw.githubusercontent.com/blackdew/tensorflow1/master/csv/lemonade.csv'
filepath2 = 'https://raw.githubusercontent.com/blackdew/tensorflow1/master/csv/boston.csv'
filepath3 = 'https://raw.githubusercontent.com/blackdew/tensorflow1/master/csv/iris.csv'

lemonade = pd.read_csv(filepath1)
boston = pd.read_csv(filepath2)
iris = pd.read_csv(filepath3)

#칼럼 이름 출력
'''
print(lemonade.columns)
print(boston.columns)
print(iris.columns)
'''

#독립변수, 종속변수 구분
독립 = lemonade[['온도']]
종속 = lemonade[['판매량']]
#print(독립.shape, 종속.shape)    #확인법

# 모델의 구조를 만드는 부분
#데이터를 준비할 때 독립변수 / 종속변수를 구분하는 과정이 중요함 왜냐하면 이게 이제 모델의 구조를 만들 때 영향을 준다
#shape 뒤에 있는 숫자가 독립변수의 수를 의미, Dense 뒤에 있는 수가 종속변수의 수를 의미
X = tf.keras.layers.Input(shape=[1])
Y = tf.keras.layers.Dense(1)(X)
model = tf.keras.models.Model(X, Y)
model.compile(loss='mse')

#모델 학습 시키는 부분
model.fit(독립, 종속, epochs= 10)

print(model.predict(독립))

#모델을 이용
print(model.predict([[15]]))










