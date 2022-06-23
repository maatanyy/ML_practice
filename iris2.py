import pandas as pd
import tensorflow as tf

filepath = 'https://raw.githubusercontent.com/blackdew/tensorflow1/master/csv/iris2.csv'
아이리스 = pd.read_csv(filepath)
#print(아이리스.dtypes)

#품종 타입을 범주형으로 바꾸어 준다.
아이리스['품종'] = 아이리스['품종'].astype('category')
print(아이리스.dtypes)

#원핫인코딩

인코딩 = pd.get_dummies(아이리스)
인코딩.head()

# NA값을 체크해 봅시다.
print(아이리스.isna().sum())

#mean 써서 평균값 구해서 NA 에 넣어주기  N/A 란 not available을 의미
mean = 아이리스['꽃잎폭'].mean()
아이리스['꽃잎폭'] = 아이리스['꽃잎폭'].fillna(mean)
print(아이리스.tail())



