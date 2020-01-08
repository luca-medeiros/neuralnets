import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def stepFunction(t):
    if t >= 0:
        return 1
    return 0

def prediction(X, W, b):
    return stepFunction((np.matmul(X,W)+b)[0])

def perceptronStep(X, y, W, b, learn_rate = 0.01):
    for i in range(len(X)):
        y_hat = prediction(X[i],W,b)
        if y[i] - y_hat == 1:  # False Negative
            W[0] += X[i][0]*learn_rate
            W[1] += X[i][1]*learn_rate
            b += learn_rate
        elif y[i] - y_hat == -1:   # False Positive
            W[0] -= X[i][0]*learn_rate
            W[1] -= X[i][1]*learn_rate
            b -= learn_rate
    return W, b
    

def trainPerceptronAlgorithm(X, y, learn_rate = 0.01, num_epochs = 100):
    x_min, x_max = min(X.T[0]), max(X.T[0])
    y_min, y_max = min(X.T[1]), max(X.T[1])
    W = np.array(np.random.rand(2,1))
    b = np.random.rand(1)[0] + x_max
    # These are the solution lines that get plotted below.
    boundary_lines = []
    for i in range(num_epochs):
        # In each epoch, we apply the perceptron step.
        W, b = perceptronStep(X, y, W, b, learn_rate)
        boundary_lines.append((-W[0]/W[1], -b/W[1]))
        
    return boundary_lines

df=pd.read_csv('data.csv', sep=',',header=None)

X = df[[0,1]].values
y = df[2].values
result = trainPerceptronAlgorithm(X, y)

plt.figure()
for inputt, target in zip(X, y):
    plt.plot(inputt[0],inputt[1], 'ro' if (target == 1.0) else 'bo')


# plt.figure()
x = np.linspace(-1, 1, 5)
c = 1

for i in result:
    if c == 1 or c == 10 or c == 25 or c ==100:
        x_r = float(i[0])
        y_r = float(i[1])
        y = x_r*x+y_r
        plt.plot(x, y, ':', label=str(c))
    c+=1
plt.legend(loc='upper left')
plt.show()