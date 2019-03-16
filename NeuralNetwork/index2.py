import numpy as np

def nn(x , w1 , y , w2 , p):
    return sigmoid(x*w1 + y*w2 +p)

def sigmoid(x):
    return 1/(1+np.exp(-x))

w1 = np.random.randn()
w2 = np.random.randn()
p = np.random.randn()

print(nn(1,w1,2,w2,p))
