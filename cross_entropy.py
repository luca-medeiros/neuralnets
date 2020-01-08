import numpy as np

def cross_entropy(Y, P):
    Y = np.float_(Y)
    P = np.float_(P)
    return -np.sum(Y * np.log(P) + (1 - Y) * np.log(1 - P))

    # valid, invalid = 0, 0
    # for i in range(len(Y)):
    #     valid -= Y[i]*np.log(P[i])
    #     invalid -= (1-Y[i])*np.log(1-P[i])
    # return (valid + invalid)
    

print(cross_entropy([1, 1, 0], [0.8, 0.7, 0.1]))