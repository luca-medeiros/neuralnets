import numpy as np

def sigmoid(x):
    return 1/(1+np.exp(-x))

learnrate = 0.2
learnt =1
epochs = 25
x = np.array([2, 3])
y = np.array(0.9)

# Initial weights
del_w = np.array([0.0, 0.0])

for i in range(epochs):
    nn_output = sigmoid(np.dot(x, del_w))
    error = y - nn_output
    del_w += learnrate * error * nn_output * (1 - nn_output) * x


print('Neural Network output: ', i)
print(nn_output)
print('Amount of Error:')
print(error)
print('Change in Weights:')
print(del_w)