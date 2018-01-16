import numpy as np
np.random.seed(42)
shape = (1, 4, 4)  # Three-dimension matrix
num_samples = 10  # The number of samples
num_ft = shape[0] * shape[1] * shape[2] # The number of features per sample
# Randomly generate X
X = np.random.rand(num_samples, num_ft)
beta = np.random.rand(num_ft, 1) # Define beta
# Add noise to y
y = np.dot(X, beta) + 0.001 * np.random.rand(num_samples, 1)
X_train = X[0:6, :]
y_train = y[0:6]
X_test = X[6:10, :]
y_test = y[6:10]

import parsimony.estimators as estimators
import parsimony.algorithms as algorithms
ols = estimators.LinearRegression(algorithm_params=dict(max_iter=1000))


import parsimony.estimators as estimators
import parsimony.algorithms as algorithms
import parsimony.functions.nesterov.tv as tv

l = 0.0  # l1 lasso coefficient
k = 0.0  # l2 ridge regression coefficient
g = 1.0  # tv coefficient
A = tv.linear_operator_from_shape(shape)  # Memory allocation for TV
olstv = estimators.LinearRegressionL1L2TV(l, k, g, A, mu=0.0001,
                                         algorithm=algorithms.proximal.FISTA(),
                                         algorithm_params=dict(max_iter=1000))

res = olstv.fit(X_train, y_train)
print("Estimated beta error = ", np.linalg.norm(olstv.beta - beta))
print("Prediction error = ", np.linalg.norm(olstv.predict(X_test) - y_test))