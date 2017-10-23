import numpy as np
import matplotlib.pyplot as plt
import pylab
import pickle
import os
import pandas as pd
import statsmodels.tsa.api as sta
import math
import statsmodels.tsa.stattools as sts
from datetime import datetime
import time
import math
import statistics as st
import statsmodels.stats.stattools as ssts
import seaborn
import itertools
from pandas.stats.api import ols
import statsmodels.api as sm
import operator

# Using statsmodels granger causality api.
# Test for co-integration between the time series is not implemented here.

def test_stationarity(timeseries):

    #Determing rolling statistics
    rolmean = pd.rolling_mean(timeseries, window=12)
    rolstd = pd.rolling_std(timeseries, window=12)

    #Plot rolling statistics:
    # plt.close()
    # orig = plt.plot(timeseries, color='blue',label='Original')
    # mean = plt.plot(rolmean, color='red', label='Rolling Mean')
    # std = plt.plot(rolstd, color='black', label = 'Rolling Std')
    # plt.legend(loc='best')
    # plt.title('Rolling Mean & Standard Deviation')
    # plt.show()

    #Perform Dickey-Fuller test:
    # print('Results of Dickey-Fuller Test:')
    dftest = sts.adfuller(timeseries, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
    # for key,value in dftest[4].items():
    #     dfoutput['Critical Value (%s)'%key] = value
    # print(dfoutput)
    return dfoutput[2]

def grangerCausality(feat_TS_gc):
    VAR_model = sta.VAR(feat_TS_gc)
    VAR_results = VAR_model.fit(5)
    # VAR_results.plot()
    # plt.show()

    # results = VAR_model.fit(maxlags=10, ic='aic')
    # print(feat_TS_gc)
    measures = ['conductance', 'conductance_experts', 'core', 'pagerank', 'outdegree']
    for idx_1 in range(len(measures)):
        for idx_2 in range(idx_1 + 1, len(measures)):
            measures_temp = []
            for k in range(idx_1, idx_2):
                measures_temp.append(measures[k])
            print(measures_temp)

            causality_results = VAR_results.test_causality('number_events', measures_temp, kind='wald', verbose=True)
            print(causality_results)

            # feat_TS = pd.concat([feat_TS, eventsTSCopy])
            # print(feat_TS)

if __name__ == '__main__':
    # eventsTS = pickle.load(open('../../data/Armstrong_data/eventsDF_df_days.pickle', 'rb'))
    # eventsTS = eventsTS.rename(columns={'date': 'date_event'})
    # eventsTS_first_last = eventsTS.iloc[[0, -1]]
    # start_date = eventsTS_first_last['date_event'].tolist()[0]
    # end_date = eventsTS_first_last['date_event'].tolist()[1]
    # eventsTSCopy = eventsTS['number_events']
    #
    # featTS_1 = pickle.load(open('../../data/DW_data/features_daysV1.0P1.pickle', 'rb'))
    # featTS_2 = pickle.load(open('../../data/DW_data/features_daysV1.0P2.pickle', 'rb'))
    #
    # featTS_2 = featTS_2.rename(columns={'conductance': 'conductance_experts', 'date': 'date_dup'})
    #
    # feat_TS = pd.concat([featTS_1, featTS_2], axis=1)
    # feat_TS = feat_TS.drop('date_dup', axis=1)
    # feat_TS = feat_TS[feat_TS['date'] >= start_date]
    # feat_TS = feat_TS[feat_TS['date'] <= end_date]
    # feat_TS['number_events'] = pd.Series(eventsTS['number_events'].tolist())
    #
    # feat_TS ['date'] = pd.to_datetime(feat_TS['date'], format='%Y-%m-%d %H:%M:%S')
    # feat_TS_gc = feat_TS.copy()
    # feat_TS_gc.index = feat_TS['date']
    # feat_TS_gc = feat_TS_gc.drop('date', axis=1)

    forums_cve_mentions = [88, 248, 133, 49, 62, 161, 84, 60, 104, 173, 250, 105, 147, 40, 197]
    vulData = pickle.load(open('../../data/DW_data/08_29/Vulnerabilities-sample_v2+.pickle', 'rb'))
    vulDataFiltered = vulData[vulData['forumID'].isin(forums_cve_mentions)]

    vulDataFiltered = vulDataFiltered[vulDataFiltered['postedDate'] > pd.to_datetime('2016-04-05', format='%Y-%m-%d')]
    vulDataFiltered = vulDataFiltered[vulDataFiltered['postedDate'] <= pd.to_datetime('2016-04-25', format='%Y-%m-%d')]

    df_cve_cpe = pd.read_csv('../../data/DW_data/09_15/CVE_CPE_groups.csv')
    print(df_cve_cpe)

    cveList_CPE = df_cve_cpe['cve'].tolist()
    CPE_list = df_cve_cpe['cluster_tag'].tolist()

    cve_cpe_dict = {}
    for idx in range(len(cveList_CPE)):
        cve_cpe_dict[cveList_CPE[idx]] = CPE_list[idx]

    CVEs = list(set(vulDataFiltered['vulnId']))
    cveCount = {}

    for idx, row in vulDataFiltered.iterrows():
        if row['vulnId'] not in cveCount:
            cveCount[row['vulnId']] = 0

        cveCount[row['vulnId']] += 1

    cveCount_sorted = sorted(cveCount.items(), key=operator.itemgetter(1), reverse=True)
    for cve, count in cveCount_sorted:
        if cve in cveList_CPE:
            print(cve, cve_cpe_dict[cve], count)
        else:
            print(cve, count)
    # print(cveCount_sorted)

    # print(vulDataFiltered)


