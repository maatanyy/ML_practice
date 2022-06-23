# 생활코딩 Tensorflow(python)을 참고하였습니다.
import pandas as pd
import tensorflow as tf


#과거의 데이터 준비하는 부분
filepath = 'https://raw.githubusercontent.com/blackdew/tensorflow1/master/csv/boston.csv'
boston = pd.read_csv(filepath)


#독립변수, 종속변수 구분
독립 = boston[['crim', 'zn', 'indus', 'chas', 'nox', 'rm', 'age', 'dis', 'rad', 'tax','ptratio', 'b', 'lstat']]
종속 = boston[['medv']]
print(독립.shape, 종속.shape)



# 모델의 구조를 만드는 부분
X = tf.keras.layers.Input(shape=[13])


#BatchNormailization Layer 이용

H = tf.keras.layers.Dense(8)(X)
H = tf.keras.layers.BatchNormalization()(H)
H = tf.keras.layers.Activation('swish')(H)

H = tf.keras.layers.Dense(8)(H)
H = tf.keras.layers.BatchNormalization()(H)
H = tf.keras.layers.Activation('swish')(H)

H = tf.keras.layers.Dense(8)(H)
H = tf.keras.layers.BatchNormalization()(H)
H = tf.keras.layers.Activation('swish')(H)

Y = tf.keras.layers.Dense(1)(H)
model = tf.keras.models.Model(X, Y)
model.compile(loss='mse')

#model.summary()  #잘 만들어졌나 확인

#모델 학습 시키는 부분
model.fit(독립, 종속, epochs= 1000)

print(model.predict(독립[0:5]))


































