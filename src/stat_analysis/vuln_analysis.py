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




def main():
    vulnInfo = pd.read_pickle('../../data/DW_data/new_Dw/Vulnerabilities_Armstrong.pickle')
    print(vulnInfo)
    trainStart_date = datetime.datetime.strptime('2016-10-01', '%Y-%m-%d')
    trainEnd_date = datetime.datetime.strptime('2017-09-01', '%Y-%m-%d')


if __name__ == "__main__":
    main()

