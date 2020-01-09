import numpy as np

def softmax(L):
    expL = np.exp(L)
    return np.divide(expL, expL.sum() )

    # for i in range(len(L)):
    #     den = 0
    #     for j in L:
    #         num = np.exp(L[i])
    #         den += np.exp(j)
    #         result = num/den
    #     prob.append(result)


print(softmax([5,1,2]))