import pandas as pd
import datetime as dt
import operator
import pickle
import numpy as np
import datetime
import matplotlib.pyplot as plt
import matplotlib
from sklearn.decomposition import PCA
import os
from dateutil.relativedelta import relativedelta
import sklearn.metrics
import random
from random import shuffle
from sklearn import linear_model, ensemble
from sklearn.naive_bayes import GaussianNB
pd.set_option("display.precision", 2)

def main():
    trainStart_date = datetime.datetime.strptime('2016-9-01', '%Y-%m-%d')
    trainEnd_date = datetime.datetime.strptime('2017-05-01', '%Y-%m-%d')


    ''' Concatenate the features into a single dataframe'''
    # fileName_prefix = ['shortestPaths', 'conductance', 'commThreads']
    fileName_prefix = ['commThreads',]

    feat_df = pd.DataFrame()
    for fp in fileName_prefix:
        if feat_df.empty:
            feat_df = pd.read_pickle('../../data/DW_data/features/feat_combine/' + str(fp) + '_DeltaT_4_Sept16-Apr17_TP10.pickle')
        else:
            curr_df = pd.read_pickle('../../data/DW_data/features/feat_combine/' + str(fp) + '_DeltaT_4_Sept16-Apr17_TP10.pickle')
            feat_df = pd.merge(feat_df, curr_df, on=['date'])

    trainDf = feat_df[feat_df['date'] >= trainStart_date]
    trainDf = trainDf[trainDf['date'] < trainEnd_date]

    # print(trainDf)
    ''' Plot the features forum wise '''
    # features = ['shortestPaths', 'CondExperts', 'expertsThreads'
    features = ['expertsThreads', ]

    for feat in features:
        for idx_cpe in range(5):
            dir_save = '../../plots/dw_stats/feat_plot/feat_combine/' + str(feat) + '/'
            if not os.path.exists(dir_save):
                os.makedirs(dir_save)
            featStr = feat+'_CPE_R' + str(idx_cpe+1)
            plotDf_CPE = trainDf[(trainDf[featStr] != -1.) & (trainDf[featStr] != 0.)][featStr]
            # print(plotDf_CPE)
            median_value = np.median(np.array(list(plotDf_CPE)))
            if np.isnan(median_value):
                median_value = 0.
            trainDf[featStr] = trainDf[featStr].replace([-1.00, 0.0, 0.00], median_value)
            # plotDf_forumCPE = trainDf[trainDf['forum'] == f]
            #
            # # print(plotDf_forumCPE[featStr])
            # trainDf.plot(figsize=(12,8), x='date', y=featStr, color='black', linewidth=2)
            # plt.grid(True)
            # plt.xticks(size=20)
            # plt.yticks(size=20)
            # plt.xlabel('date', size=20)
            # plt.ylabel(feat, size=20)
            # plt.subplots_adjust(left=0.13, bottom=0.25, top=0.9)
            # # plt.show()
            # file_save = dir_save + 'CPE_R' + str(idx_cpe+1)
            # plt.savefig(file_save)
            # plt.close()

    pickle.dump(trainDf, open('../../data/DW_data/features/feat_combine/train_df_et_v12_12.pickle', 'wb'))
    # ''' Form the residual and normal time series for each feaure, for each CPE'''
    # subspace_df = pd.DataFrame()
    # for feat in features:
    #     for idx_cpe in range(5):
    #         featStr = feat + '_CPE_R' + str(idx_cpe + 1)
    #         trainDf_subspace  = trainModel(trainDf, featStr, forums)
    #         if subspace_df.empty:
    #             subspace_df = trainDf_subspace
    #         else:
    #             subspace_df = pd.merge(subspace_df, trainDf_subspace, on=['date'])
    #
    # pickle.dump(subspace_df, open('../../data/DW_data/features/subspace_df_v12_10.pickle', 'wb'))

    # print(subspace_df)
    # for column in subspace_df:
    #     if column == 'date':
    #         continue
    #     dir_save = '../../plots/dw_stats/subspace_plot/'
    #     if not os.path.exists(dir_save):
    #         os.makedirs(dir_save)
    #
    #     subspace_df.plot(figsize=(12,8), x='date', y=column, color='black', linewidth=2)
    #     plt.grid(True)
    #     plt.xticks(size=20)
    #     plt.yticks(size=20)
    #     plt.xlabel('date', size=20)
    #     plt.ylabel(column, size=20)
    #     # plt.title('Forum=' + str(f))
    #     plt.subplots_adjust(left=0.13, bottom=0.25, top=0.9)
    #     # plt.show()
    #     file_save = dir_save + column
    #     plt.savefig(file_save)
    #     plt.close()

if __name__ == "__main__":
    main()

