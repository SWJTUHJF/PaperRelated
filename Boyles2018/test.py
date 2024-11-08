from collections import defaultdict
import numpy as np

arr = np.arange(9).reshape((3,3))
print(arr)
iden = np.identity(3)
print(iden)
res = iden - arr.T
print(np.linalg.inv(res) @ res)