#2022-06-21

import tensorflow as tf
height = [170, 180, 175, 160]
shoesSize = [260, 270, 265, 255]

#??? = ax + b  # solve a and b if height x is given
#shoesSize = height * a + b

a = tf.Variable(0.1)
b = tf.Variable(0.2)

def lossFunction():

    predictedValue = height * a + b
    return tf.square(260 - predictedValue)

opt = tf.keras.optimizers.Adam(learning_rate=0.1)    #경사하강법을 도와주는 도구, 여러 optimizer가 있는데 gradient를 알아서 스마트하게 바꿔줌

for i in range(300):
    opt.minimize(lossFunction, var_list=[a,b])  #  minimize 에 옵션을 2개 줘야함 첫번째는 함수 두번째는 var_list에는 경사하강법으로 업데이트할 weight Variable 목록을 넣어줌
    print(a.numpy(),b.numpy())






"""
텐서 = tf.constant( [3,4,5] )  #숫자도 되고 리스트도 되고 텐서에 담음
텐서2 = tf.constant( [6,7,8] )
텐서3 = tf.constant ( [ [1,2],
                        [3,4] ])

 tf.matmul() 행렬곱샘연산
 tf.zeros() 0만 담긴 텐서 만들어줌
()안에 넣을 행렬 만듬 예를들어 [2,2] 넣으면 2*2 0으로 찬 tensor 만듬

print(텐서)
tf.cast() 로 데이터타입 바꾸기가능
print(tf.add(텐서, 텐서2))
print(텐서 + 텐서2)

tf.constant 를 써서 변하지 않는 상수를 넣을 수 있는데
tf.Variable 을 넣으면 weight 만들 때 씀
ex) w = tf.Variable(1.0)
 print(w.numpy()) 해서 그 값 출력 가능
 .assign() 써서 값 재할당 가능

Variable은 변경이 가능함
"""
