import numpy as np

X_arr = np.array([-1, -1, 0, 3])
condition = (X_arr != 0.)
print(condition)
X_filter = np.extract(condition, X_arr)
print(list(X_filter))