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
from sklearn import linear_model, ensemble
from sklearn.naive_bayes import GaussianNB
from random import shuffle

from sklearn.svm import SVC
from sklearn import tree
import csv
import sys
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


def weeklyCVE_event_corr(eventsDf, vulnInfo, cve_cpe_map, start_date, end_date):
    '''

    :param eventsDf:
    :param attackdf:
    :param vulnInfo:
    :param cve_cpe_map:
    :param start_date:
    :param end_date:
    :return:
    '''
    eventsDf['date'] = pd.to_datetime(eventsDf['date'])
    vulnInfo['posteddate'] = pd.to_datetime(vulnInfo['posteddate'])

    startWeek = start_date
    endWeek = startWeek + datetime.timedelta(days=7)

    # DS Structures
    startDatesList = []
    endDatesList = []
    vulnIdsList = []
    vulnCounstList = []
    CPEsList = []
    CPECountsList = []
    numAttacks = []

    while(endWeek < end_date):

        ''' For the darkweb data '''
        vulnsCurr = {}
        cpesCurr = {}
        # try:
        vulWeek = vulnInfo[vulnInfo['posteddate'] >= startWeek - datetime.timedelta(days=14)]
        vulWeek = vulWeek[vulWeek['posteddate'] < startWeek]

        for idx, row in vulWeek.iterrows():
            cve = row['vulnerabilityid']
            if cve not in vulnsCurr:
                vulnsCurr[cve] = 0
            vulnsCurr[cve] += 1

            if cve in cve_cpe_map:
                cpes = cve_cpe_map[cve]
                for cp in cpes:
                    if cp not in cpesCurr:
                        cpesCurr[cp] = 0
                    cpesCurr[cp] += 1

        # except:
        #     pass

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
        vulnIdsList.append(list(vulnsCurr.keys()))
        vulnCounstList.append(list(vulnsCurr.values()))
        CPEsList.append(list(cpesCurr.keys()))
        CPECountsList.append(list(cpesCurr.values()))
        numAttacks.append(attacksCurr)

        startWeek = endWeek
        endWeek = startWeek + datetime.timedelta(days=7)

    outputDf = pd.DataFrame()
    outputDf['start_dates'] = startDatesList
    outputDf['end_dates'] = endDatesList
    outputDf['vulnerabilities'] = vulnIdsList
    outputDf['vuln_counts'] = vulnCounstList
    outputDf['CPEs'] = CPEsList
    outputDf['CPE_counts'] = CPECountsList
    outputDf['number_attacks'] = numAttacks

    return outputDf


# def weeklyCVE_event_corr():


def loadVulnInfo(df):
    vuln_groups= df.groupby(['vulnerabilityid'])



def main():
    amEvents = pd.read_csv('../../data/Armstrong_data/amEvents_11_17.csv')
    amEvents_malware = amEvents[amEvents['type'] == 'malicious-email']
    vuln_df = pd.read_csv('../../data/DW_data/VulnInfo_11_17.csv', encoding='ISO-8859-1', engine='python')
    cve_cpe_map = pickle.load(open('../../data/DW_data/cve_cpe_map.pickle', 'rb'))

    trainStart_date = datetime.datetime.strptime('2016-10-01', '%Y-%m-%d')
    trainEnd_date = datetime.datetime.strptime('2017-05-01', '%Y-%m-%d')

    outputDf = weeklyCVE_event_corr(amEvents_malware, vuln_df, cve_cpe_map, trainStart_date, trainEnd_date)

    pickle.dump(outputDf, open('../../data/DW_data/CPE_events_corr.pickle', 'wb'))
if __name__ == "__main__":
    main()

