from matplotlib.dates import DateFormatter
import pandas as pd
import pickle
import numpy as np
import datetime
import matplotlib.pyplot as plt
import csv
import sys
import matplotlib.dates as mdates


class ArgsStruct:
    name = ''
    plot_data = False

maxInt = sys.maxsize
decrement = True

while decrement:
    # decrease the maxInt value by factor 10
    # as long as the OverflowError occurs.

    decrement = False
    try:
        csv.field_size_limit(maxInt)
    except OverflowError:
        maxInt = int(maxInt/10)
        decrement = True


def dateToString(date):
    yearNum = str(date.year)
    monthNum = str(date.month)
    dayNum =str(date.day)
    if len(monthNum)<2:
        monthNum = "0"+monthNum
    if len(dayNum)<2:
        dayNum= "0"+dayNum
    return yearNum+"-"+monthNum+"-"+dayNum


def weeklyCVE_anomaly_corr(eventsDf, anomalyVecDf, start_date, end_date):
    '''

    :param eventsDf: Armstrong data
    :param anomalyVecDf: Darkweb data
    :param start_date:
    :param end_date:
    :return:
    '''

    eventsDf['date'] = pd.to_datetime(eventsDf['date'])
    anomalyVecDf['date'] = pd.to_datetime(anomalyVecDf['date'])

    # DS Structures
    startDatesList = []
    endDatesList = []
    attacksList = []

    outputDf = pd.DataFrame()

    startWeek = start_date
    endWeek = startWeek + datetime.timedelta(days=7)

    while (endWeek < end_date):
        ''' For the armstrong data '''
        eventsCurr = eventsDf[eventsDf['date'] >= startWeek]
        eventsCurr = eventsCurr[eventsCurr['date'] < endWeek]
        total_count = pd.DataFrame(eventsCurr.groupby(['date']).sum())
        count_attacks = np.sum(total_count['count'].values)

        if count_attacks == 0:
            attacksCurr = 0
        else:
            attacksCurr = count_attacks

        startDatesList.append(startWeek)
        endDatesList.append(endWeek)
        attacksList.append(attacksCurr)

        startWeek = endWeek
        endWeek = startWeek + datetime.timedelta(days=7)

    outputDf['start_dates'] = startDatesList
    outputDf['end_dates'] = endDatesList
    outputDf['number_attacks'] = attacksList

    for feat in anomalyVecDf.columns.values:
        if feat == 'date':
            continue

        startWeek = start_date
        endWeek = startWeek + datetime.timedelta(days=7)

        anomalyCount = []
        ''' Store the weekly statistics of the anomalies '''
        while(endWeek < end_date):
            anomalyCountWeek = 0

            ''' For the darkweb data '''
            currWeekDf = anomalyVecDf[anomalyVecDf['date'] >= startWeek]
            currWeekDf = currWeekDf[currWeekDf['date'] < endWeek]

            for idx, row in currWeekDf.iterrows():
                if row[feat] == 1:
                    anomalyCountWeek += 1

            startWeek = endWeek
            endWeek = startWeek + datetime.timedelta(days=7)

            anomalyCount.append(anomalyCountWeek)

        outputDf[feat] = anomalyCount

    return outputDf


def anomaly_eventCorr(eventsDf, anomalyVecDf, start_date, end_date):
    '''

    :param eventsDf:
    :param anomalyVecDf:
    :param start_date:
    :param end_date:
    :return:
    '''

    histDays = 30

    eventsDf['date'] = pd.to_datetime(eventsDf['date'])
    eventsDf = eventsDf[eventsDf['date'] >= start_date]
    eventsDf = eventsDf[eventsDf['date'] < end_date]

    anomalyVecDf = anomalyVecDf[anomalyVecDf['date'] >= (start_date-datetime.timedelta(days=histDays))]
    anomalyVecDf = anomalyVecDf[anomalyVecDf['date'] < end_date]

    anomPriorEvent = {}

    ''' The following calculates the number of distinct days on which there was an attack '''
    num_Attacks = len(eventsDf[eventsDf['count'] > 0])

    for idx, row in eventsDf.iterrows():
        dateAttack = row['date']

        ''' Consider the following operations only if there is an attack on that day '''
        if row['count'] > 0:
            ''' Iterate over the history of anomalies in darkweb
                This is  where we see how the anomalies in features correlate with attacks
            '''
            for hd in range(1, histDays+1):
                dateAnom = dateAttack - datetime.timedelta(days=hd)

                ''' Iterate over the features inividually to check their occurrence
                    1. Check only the residual vectors
                    2. TODO: Take the union of the state and the residual vectors
                '''
                for feat in anomalyVecDf.columns.values:
                    if feat == 'date' or 'state' in feat: # Avoid the state vector flags
                        continue

                    if feat not in anomPriorEvent:
                        anomPriorEvent[feat] = [0 for _ in range(histDays)]
                    isAnom = (anomalyVecDf[anomalyVecDf['date'] == dateAnom][feat].values)[0]
                    # print(isAnom)
                    if isAnom == 1:
                        anomPriorEvent[feat][hd-1] += 1

    for feat in anomPriorEvent:
        for d in range(len(anomPriorEvent[feat])):
            anomPriorEvent[feat][d] /= num_Attacks

    return anomPriorEvent


def main():
    amEvents = pd.read_csv('../../data/Armstrong_data/amEvents_11_17.csv')
    amEvents_malware = amEvents[amEvents['type'] == 'malicious-email']

    anomalyDf = pd.read_pickle('../../data/DW_data/features/feat_forums/anomalyVec_Delta_T0_Mar16-Aug17_v1.pickle')
    subspaceDf = pd.read_pickle('../../data/DW_data/features/feat_forums/subspace_df_v01_05.pickle')
    # subspaceDf = subspaceDf[['date', 'CondExperts_res_vec']]
    # subspaceDf = subspaceDf[subspaceDf['date'] > '2017-01-01']
    # subspaceDf = subspaceDf[subspaceDf['date'] < '2017-02-01']
    # print(subspaceDf)
    #
    # exit()
    trainStart_date = datetime.datetime.strptime('2016-7-01', '%Y-%m-%d')
    trainEnd_date = datetime.datetime.strptime('2017-09-01', '%Y-%m-%d')

    outputDf = weeklyCVE_anomaly_corr(amEvents_malware, anomalyDf, trainStart_date, trainEnd_date)

    # anomPriorEvent = anomaly_eventCorr(amEvents_malware, anomalyDf, trainStart_date, trainEnd_date)

    # anomPriorEvent = anomaly_eventCorr(amEvents_malware, anomalyDf, trainStart_date, trainEnd_date)
    pickle.dump(outputDf, open('../../data/DW_data/anomalies_events_corr_v1.pickle', 'wb'))
    # anomPriorEvent = pd.read_pickle('../../data/DW_data/anomalies_events_corr.pickle')
    # for feat in anomPriorEvent:
    #     print(feat)
    #     print(anomPriorEvent[feat])


if __name__ == "__main__":
    main()

