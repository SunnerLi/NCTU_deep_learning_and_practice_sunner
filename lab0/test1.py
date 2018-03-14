import numpy as np

a = np.reshape(np.asarray(range(8)), [4, 2])
b = np.reshape(np.asarray(range(8)), [4, 3])
print(a*b)