# 생활코딩 Tensorflow(python)을 참고하였습니다.
# 기존과 다르게 이건 범주형.. 종속변수가 양적인 경우 회귀 알고리즘 사용,,, 범주형인경우 분류 알고리즘 사용!!
import pandas as pd
import tensorflow as tf


#과거의 데이터 준비하는 부분
아이리스 = pd.read_csv('iris.csv')  #범주형 칼럼과 수치형 칼럼이 iris에 섞여있음
iris = pd.get_dummies(아이리스)   #원핫인 코딩,,, 데이터내 범주형만 골라서 원핫인코딩된 결과를 보여줌캨
#칼럼 이름 출력
print(iris.columns)


#독립변수, 종속변수 구분
독립 = iris[['꽃잎길이', '꽃잎폭', '꽃받침길이', '꽃받침폭']]
종속 = iris[['품종_setosa', '품종_versicolor', '품종_virginica']]


# 모델의 구조를 만드는 부분
X = tf.keras.layers.Input(shape=[4])
Y = tf.keras.layers.Dense(3, activation='softmax')(X)
model = tf.keras.models.Model(X, Y)
model.compile(loss='categorical_crossentropy', metrics='accuracy')  #문제에 따라 loss를 다르게 사용한다

#모델 학습 시키는 부분
model.fit(독립, 종속, epochs=100)

#모델을 이용
print(model.predict(독립[-5:]))
print(종속[-5:])

#학습한 가중치
model.get_weights()








