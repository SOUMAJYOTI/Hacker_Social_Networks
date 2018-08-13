import pandas as pd
import pickle
import sklearn.metrics
import numpy as np
import datetime


if __name__ == "__main__":
    feat_predictDf = pickle.load(open('../../../data/results/05_23/SNA/malicious_email/tgap_7.pickle', 'rb'))

    print(feat_predictDf)