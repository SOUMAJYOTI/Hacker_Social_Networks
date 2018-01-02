import time
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import datetime

def plot_correlation_matrices(corrMat):
    ''


def time_lagged_cross_correlation_ts(ts_1, ts_2, start_date, end_date, tlag=1):
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

    # Filter the dataframes based on the input date ranges
    ts_1 = ts_1[ts_1['start_dates'] >= start_date.date()]
    ts_1 = ts_1[ts_1['start_dates'] < end_date.date()]

    ts_2 = ts_2[ts_2['start_dates'] >= start_date.date()]
    ts_2 = ts_2[ts_2['start_dates'] < end_date.date()]


    # Convert the time series Dataframe into pandas series object of 1d TS
    ts_1_1d = ts_1.ix[:, 1]
    ts_2_1d = ts_2.ix[:, 1]

    max_lag_order = 6
    correlationMatrix = [[] for _ in range(max_lag_order)]
    for lag_idx in range(1, max_lag_order+1):
        ts_1_1d_shift = ts_1_1d.shift(periods=-lag_idx)
        ts_1_1d_shift = ts_1_1d_shift[:-lag_idx]
        ts_2_1d = ts_2_1d[lag_idx:]

        correlationMatrix[lag_idx-1] = np.correlate(ts_1_1d_shift, ts_2_1d)
        print(np.correlate(ts_1_1d_shift, ts_2_1d))
        # correlationMatrix[lag_idx-1] += ([0.] * lag_idx)

    return np.array(correlationMatrix)


def main():
    ''

    # Load the time series data for measuring cross correlation
    cve_eventsDf = pd.read_pickle('../../../data/DW_data/CPE_events_corr.pickle')

    ts_1 = cve_eventsDf[['start_dates', 'number_attacks']]
    ts_2 = cve_eventsDf[['start_dates', 'vuln_counts_sum']]

    start_date = datetime.datetime.strptime('2016-10-01', '%Y-%m-%d')
    end_date = datetime.datetime.strptime('2017-09-01', '%Y-%m-%d')

    corrMat = time_lagged_cross_correlation_ts(ts_1, ts_2, start_date, end_date)
    print(corrMat)

if __name__ == "__main__":
    main()