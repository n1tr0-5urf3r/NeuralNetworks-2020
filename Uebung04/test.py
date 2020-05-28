import random
import numpy as np

_weights = np.zeros([5])


for i, w in enumerate(_weights):
     _weights[i] = random.uniform(-0.1 ,0.1)

print(_weights)

