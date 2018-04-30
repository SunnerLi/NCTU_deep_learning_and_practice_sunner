import numpy as np

a = list(range(10)) * 10
a = np.reshape(list(range(10)) * 10, [10, 10])
print(a)