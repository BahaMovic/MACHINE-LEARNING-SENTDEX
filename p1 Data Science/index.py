import pandas as pd
import numpy as np
import math
from sklearn import preprocessing ,svm
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
def get_train_test_split(X,Y,test_size=0.3):
    test_size = test_size * len(X)
    test_size = int(test_size)
    train_size = len(X) - test_size
    train_size = int(train_size)

    X_Train = X[:train_size]
    # print(len(X_Train))
    X_Test = X[train_size:train_size+test_size]
    Y_Train = Y[:train_size]
    Y_Test = Y[train_size:train_size+test_size]
    return X_Train , X_Test , Y_Train , Y_Test

df = pd.read_csv('p1 Data Science/HistoricalQuotes.csv')
df['HL_PCT'] = (df['high'] - df['close'])/df['close'] *100
df['HL_CHANGE'] = (df['close'] - df['open'])/df['open'] * 100
df = df[['close','HL_PCT','HL_CHANGE','volume']]
df['HL_PCT'][3] = np.NaN
df.fillna(-99999,inplace=True)

forecast_out = int(math.ceil(.01*(len(df))))

df['label'] = df['close'].shift(-forecast_out)
df.dropna(inplace=True)

X = preprocessing.scale(np.array(df.drop('label',axis=1)),1)
Y = np.array(df['label'])
# print(len(Y),len(X),Y)

X_Train , X_Test , Y_Train , Y_Test = get_train_test_split(X,Y,test_size=.2)

# print(len(X_Train),len(Y_Train))
cls = svm.SVR()
cls.fit(X_Train,Y_Train)
pred = cls.predict(X_Test)
print(pred,Y_Test)

plt.plot(range(len(pred)),pred,color="blue")

plt.scatter(range(len(pred)),Y_Test,color="red")
plt.show()


plt.show()
