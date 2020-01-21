import numpy as np

li = ['w', 'k']
gradients = {n: np.zeros_like(0) for n in li}
print(gradients)
for n in li:
    print(n.gradients[n])