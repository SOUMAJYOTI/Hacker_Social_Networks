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


def weeklyCVE_stats(eventsDf,start_date, end_date):
    '''

    :param eventsDf: Armstrong data
    :param start_date:
    :param end_date:
    :return:
    '''


    # DS Structures
    startDatesList = []
    endDatesList = []
    weeklyCount = []

    outputDf = pd.DataFrame()

    startWeek = start_date
    endWeek = startWeek + datetime.timedelta(days=7)

    while (endWeek < end_date):
        ''' For the armstrong data '''
        currWeekCount = 0
        for idx, row in eventsDf.iterrows():
            for d in row['postedDate']:
                d = pd.to_datetime(d)
                if d >=startWeek and d < endWeek:
                    currWeekCount += 1
                    break

        weeklyCount.append(currWeekCount)
        startDatesList.append(startWeek)
        endDatesList.append(endWeek)
        startWeek = endWeek
        endWeek = startWeek + datetime.timedelta(days=7)

    outputDf['start_dates'] = startDatesList
    outputDf['end_dates'] = endDatesList
    outputDf['number_VulnMentions'] = weeklyCount
    return outputDf


def main():
    vulnInfo = pd.read_pickle('../../data/DW_data/new_Dw/Vulnerabilities_Armstrong.pickle')


    trainStart_date = datetime.datetime.strptime('2016-01-01', '%Y-%m-%d')
    trainEnd_date = datetime.datetime.strptime('2017-09-01', '%Y-%m-%d')

    pickle.dump(weeklyCVE_stats(vulnInfo, trainStart_date, trainEnd_date),
                open('../../data/DW_data/stats/weekly_vulnMentions.pickle', 'wb'))

    vulnWeeklyInfo = pd.read_pickle('../../data/DW_data/stats/weekly_vulnMentions.pickle')
    print(vulnWeeklyInfo)


if __name__ == "__main__":
    main()

