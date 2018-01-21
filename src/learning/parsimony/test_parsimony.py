# -*- coding: utf-8 -*-
"""
Created on Mon Dec 23 15:00:10 2013

Copyright (c) 2013-2014, CEA/DSV/I2BM/Neurospin. All rights reserved.

@author:  Edouard Duchesnay
@email:   edouard.duchesnay@cea.fr
@license: BSD 3-clause.
"""
import numpy as np
import scipy.sparse as sparse
import matplotlib.pyplot as plt
import parsimony.datasets as datasets
import parsimony.functions.nesterov.tv as nesterov_tv
import parsimony.estimators as estimators
import parsimony.utils as utils
from parsimony.utils.penalties import l1_max_logistic_loss
from sklearn.metrics import precision_recall_fscore_support
from sklearn.linear_model import LogisticRegression

###############################################################################
# Dataset
n_samples = 500
shape = (50, 50, 1)

X3d, y, beta3d, proba = datasets.classification.dice5.load(n_samples=n_samples,
                                                           shape=shape, snr=10, random_seed=1)

X = X3d.reshape((n_samples, np.prod(beta3d.shape)))

n_train = 300

Xtr = X[:n_train, :]
ytr = y[:n_train]
Xte = X[n_train:, :]
yte = y[n_train:]

# Empirically set the global penalty, based on maximum l1 penaly
alpha = l1_max_logistic_loss(Xtr, ytr)

###############################################################################

# LogisticRegressionL1L2TV
l1, l2, tv = alpha * np.array((.05, .75, .2))  # l1, l2, tv penalties
# l1, l2, tv = alpha * np.array((.33, .33, 33))  # l1, l2, gn penalties
A = nesterov_tv.linear_operator_from_shape(beta3d.shape)
enettv = estimators.LogisticRegressionL1L2TV(l1, l2, tv, A,
                                             algorithm_params=dict(eps=1e-5))
yte_pred_enettv = enettv.fit(Xtr, ytr).predict(Xte)
_, recall_enettv, _, _ = \
    precision_recall_fscore_support(yte, yte_pred_enettv, average=None)


plot = plt.subplot(235)
utils.plots.map2d(enettv.beta.reshape(shape), plot,
           title="Enet+TV (%.2f, %.2f)" % tuple(recall_enettv))

plt.show()