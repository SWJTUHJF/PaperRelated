from collections import defaultdict
import numpy as np

arr = np.arange(16).reshape((4,4))
print(arr.T)
arr.T[0] = 99
print(arr.T, arr)