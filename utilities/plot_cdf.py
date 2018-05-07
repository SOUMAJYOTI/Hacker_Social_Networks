import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

num_bins = 20

data_df = pd.read_pickle('../data/DW_data/features/feat_combine/features_Delta_T0_Mar16-Aug17.pickle')

for feat in data_df.columns:
    if feat == 'date':
        continue

    feat_list = data_df[feat].tolist()

    counts, bin_edges = np.histogram(feat_list, bins=num_bins, normed=True)
    cdf = np.cumsum (counts)
    plt.plot (bin_edges[1:], cdf/cdf[-1])
    plt.show()