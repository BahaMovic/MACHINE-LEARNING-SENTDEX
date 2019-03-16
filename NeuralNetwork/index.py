import numpy as np

def sigmoid(x):
    return 1/(1+np.exp(-x))

def sigmoid_derivative(x):
    return x * (1-x)

x = np.array([[0,0,1],
            [1,1,1],
            [1,0,1],
            [0,1,1]])
y = np.array([0,1,1,0]).reshape((4,1))
np.random.seed(1)

systematic_layer =  2 * np.random.random((3,1)) - 1
print(systematic_layer)
for iteration in range(10000):
    input_layer = x

    outputs = sigmoid(np.dot(input_layer,systematic_layer))

    error = y - outputs
    adjust = error * sigmoid_derivative(outputs)

    systematic_layer += np.dot(input_layer.T ,adjust)
print("adj")
print(sigmoid(np.dot([1,1,0],systematic_layer)))
print(systematic_layer)
print("after training")
print(outputs)


