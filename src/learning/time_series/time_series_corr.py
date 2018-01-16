import time
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import datetime


def plot_correlation_matrices(corrMat):
    ''


def time_lagged_cross_correlation_ts(ts_1, ts_2, start_date, end_date, feat, tlag=1):
    '''
    This function computes the cross correlation between the two time series
    with a lag order of tlag.

    Ideally the cross correlation function between 2 TS
    is assymteric, therefore it is also needed to change the
    :param ts_1:
    :param ts_2:
    :param tlag: Time lag order between the two series
    :return:
    '''


    ''' The causal variable'''
    ts_1 = ts_1[ts_1['start_dates'] >= start_date.date()]
    ts_1 = ts_1[ts_1['start_dates'] < end_date.date()]

    max_lag_order = 8
    # print(ts_2)
    correlationMatrix = [0. for _ in range(max_lag_order)]
    for lag_idx in range(1, max_lag_order+1):
        ts_2_shift = ts_2.copy()
        ts_2_shift[feat] = ts_2_shift[feat].shift(periods=lag_idx)
        # print(ts_2_shift)

        # Filter the dataframes based on the input date ranges
        ts_2_shift = ts_2_shift[ts_2_shift['date'] >= start_date]
        ts_2_shift = ts_2_shift[ts_2_shift['date'] < end_date]

        # Convert the time series Dataframe into pandas series object of 1d TS
        ts_1_1d = ts_1.ix[:, 1]
        ts_2_1d = ts_2_shift.ix[:, 1]

        correlationMatrix[lag_idx-1] = np.correlate(ts_1_1d, ts_2_1d)[0]

    return np.array(correlationMatrix)


def main():
    ''

    # Load the time series data for measuring cross correlation
    featDf = pd.read_pickle('../../../data/DW_data/features/feat_combine/graph_stats_weekly_Delta_T0_Mar16-Sep17.pickle')
    cve_eventsDf = pd.read_pickle('../../../data/DW_data/CPE_events_corr_me.pickle')

    ts_1 = cve_eventsDf[['start_dates', 'number_attacks']]
    featDf['date'] = pd.to_datetime(featDf['date'])
    for feat in featDf.columns.values:
        if feat == 'date':
            continue
        ts_2 = featDf[['date', feat]]

        start_date = datetime.datetime.strptime('2016-12-01', '%Y-%m-%d')
        end_date = datetime.datetime.strptime('2017-07-01', '%Y-%m-%d')

        corrMat = time_lagged_cross_correlation_ts(ts_1, ts_2, start_date, end_date, feat)
        print('Feature: ', feat)
        print(corrMat)

if __name__ == "__main__":
    main()