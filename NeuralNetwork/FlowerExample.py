'''
We Have red and blue flowers  
red where length 3,4,3.5,5.5
blue where length 2,3,2,1
red where width 1.5,1.5,.5,1
blue where width 1,1,.5,1
so if you found flower with 4.5 length and 1 width 
how can to predict the output of this information
'''

'''
closest to 1 is red
closest to o is blue
'''
import numpy as np
import matplotlib.pyplot as plt
from gtts import gTTS
import os

def nn(x,y,w1,w2,b):
    return sigmoid(x*w1 + y*w2 + b)

def sigmoid(x):
    return (1)/(1+np.exp(-x))

def cost(x,pred_x):
    return (pred_x-x)**2

def scope(x_pred,x):
    return 2*(x_pred-x)
def sigmoid_p(x):
    return sigmoid(x)*(1-sigmoid(x))

flowers = [[3,1.5,1],[4,1.5,1],[3.5,.5,1],[5.5,1,1],[2,1,0],[3,1,0],[2,.5,0],[1,1,0]]
w1 = np.random.randn()
w2 = np.random.randn()
b = np.random.randn()
h = .1


for i in range(10000):
    ri = np.random.randint(len(flowers))
    point = flowers[ri]
    z = point[0]*w1 + point[1]*w2 +b
    pred = sigmoid(z)
    cost = (pred-point[2])**2
    scopeA = 2*(pred-point[2])
    dbw1 = sigmoid_p(z)*scopeA*point[0]
    dbw2 = sigmoid_p(z)*scopeA*point[1]
    dbb = sigmoid_p(z)*scopeA*1

    w1 = w1 - h*dbw1
    w2 = w2 - h*dbw2
    b = b - h*dbb

result = nn(5,1.5,w1,w2,b)
if result > .5:
    tts = gTTS(text='Flower is red', lang='en')
else:
    tts = gTTS(text='Flower is blue', lang='en')

tts.save("good.mp3")
os.system("mpg321 good.mp3")
os.startfile("good.mp3")