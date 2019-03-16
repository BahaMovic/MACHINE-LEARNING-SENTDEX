from statistics import mean
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
import math
style.use('fivethirtyeight')
ys = np.array([1,2,3,4,5,6])
xs = np.array([5,4,4,3,2,1])

'''
    mean(x)*mean(y) - mean(x*y)
M = ---------------------------
    mean(x)*mean(x) - mean(x*x)
    
'''
def get_best_slope(X,Y):

    M = ((X*Y).mean() - X.mean()*Y.mean()) / ((X**2).mean() - (X.mean())**2)

    P = mean(Y) - M*mean(X)
    # M = math.sqrt(M**2)
    # print(M)
    return M , P


def r_squared(ys , yp):
    return sum((yp - ys)**2)

def get_coefficient_determination(ys,yp):
    ys_avrg = [mean(ys) for y in ys]
    y_original = r_squared(ys,ys_avrg)
    y_pred = r_squared(ys,yp)
    return 1-(y_pred/y_original)

m , p = get_best_slope(xs,ys)
regresstion = [(x*m+p) for x in xs]

print(get_coefficient_determination(ys,regresstion))
plt.scatter(xs,ys)
plt.plot(xs,regresstion)
plt.show()

