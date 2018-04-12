import pandas as pd
import pickle
import numpy as np
import datetime
import matplotlib.pyplot as plt
import csv
import sys
import matplotlib.dates as mdates
import operator
from collections import *

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


def weeklyCVE_event_corr(eventsDf, vulnInfo, cve_cpe_map, start_date, end_date):
    '''

    :param eventsDf:
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
        vulWeek = vulnInfo[vulnInfo['posteddate'] >= startWeek]
        vulWeek = vulWeek[vulWeek['posteddate'] < endWeek]

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


''' Find the top CVE/CPE groups over weeks '''
def getTopCVEs_byWeek(dataDf):
    for idx, row in dataDf.iterrows():
        vulns = row['vulnerabilities']
        vulnCounts = row['vuln_counts']

        cpes = row['CPEs']
        cpeCounts = row['CPE_counts']

        cveCount_List = {}
        cpeCount_List = {}
        for idx_vuln in range(len(vulns)):
            # if vulns[idx_vuln] == 'cve-2017-5638':
            #     print(row['start_dates'], vulnCounts[idx_vuln])
            cveCount_List[vulns[idx_vuln]] = vulnCounts[idx_vuln]
            cveCount_List_sorted = sorted(cveCount_List.items(), key=operator.itemgetter(1), reverse=True )

        for idx_cpe in range(len(cpes)):
            cpeCount_List[cpes[idx_cpe]] = cpeCounts[idx_cpe]

        cpeCount_List_sorted = sorted(cpeCount_List.items(), key=operator.itemgetter(1), reverse=True)


            # for v, c in cveCount_List_sorted:
            #     print(v, c)

        print(row['start_dates'], row['number_attacks'], cveCount_List_sorted,)# cpeCount_List_sorted[:5])
        # print(row['start_dates'], cpeCount_List_sorted[:8])


def cpeCountsByWeek(vulnInfo, cve_cpe_map, start_date, end_date):
    '''
    The purpose of the function is to count the CPEs by the weeks - CPE wise and time wise

    :return:
    '''
    startWeek = start_date
    endWeek = startWeek + datetime.timedelta(days=7)

    startDatesList = []
    endDatesList = []
    cpesCurr = defaultdict()

    cpeList = defaultdict(int)
    while (endWeek < end_date):
        cpesCurr[startWeek] = defaultdict(int)
        ''' For the darkweb data '''
        vulnsCurr = defaultdict(int)


        vulWeek = vulnInfo[vulnInfo['posteddate'] >= startWeek]
        vulWeek = vulWeek[vulWeek['posteddate'] < endWeek]

        for idx, row in vulWeek.iterrows():
            cve = row['vulnerabilityid']
            vulnsCurr[cve] += 1

            if cve in cve_cpe_map:
                cpes = cve_cpe_map[cve]
                for cp in cpes:
                    # take the main version
                    cp_mver = (cp.split('_'))[0]
                    cpeList[cp_mver] += 1
                    cpesCurr[startWeek][cp_mver] += 1

        startDatesList.append(startWeek)
        endDatesList.append(endWeek)

        startWeek = endWeek
        endWeek = startWeek + datetime.timedelta(days=7)

    ''' Form the weekly CPE df'''

    outputDf = pd.DataFrame()
    outputDf['start_dates'] = startDatesList
    outputDf['end_dates'] = endDatesList

    for cpe in cpeList:
        if cpeList[cpe] > 500:
            cpeCountWeek = []
            startWeek = start_date
            endWeek = startWeek + datetime.timedelta(days=7)
            while (endWeek < end_date):
                if cpe in cpesCurr[startWeek]:
                    cpeCountWeek.append(cpesCurr[startWeek][cpe])
                else:
                    cpeCountWeek.append(0)

                startWeek = endWeek
                endWeek = startWeek + datetime.timedelta(days=7)

            outputDf[cpe] = cpeCountWeek

    return outputDf


def analyzeResiduals(subspaceDf):
    featList = ['numUsers', 'numVulnerabilities', 'numThreads', 'expert_NonInteractions',
                'communityCount', 'shortestPaths', 'CondExperts']
    for feat in featList:
        residualValues = subspaceDf[feat + '_res_vec']
        print(feat, np.mean(residualValues), 10*np.mean(residualValues))


# FIrst select the few attack days of each event depending on the count of attacks
def select_attack_days(eventsDf):
    '''

    :param eventsDf:
    :return:
    '''

    for idx, row in eventsDf.iterrows():
        if row['number_attacks'] >= 5:
            print(row)

def main():
    start_date = datetime.datetime.strptime('2016-01-01', '%Y-%m-%d')
    end_date = datetime.datetime.strptime('2017-08-01', '%Y-%m-%d')

    cve_eventsDf = pd.read_pickle('../../data/DW_data/CPE_events_corr_em.pickle')
    cve_cpe_map = pd.read_pickle('../../data/DW_data/new_DW/cve_cpe_map_new.pickle')
    vuln_df = pd.read_csv('../../data/DW_data/VulnInfo_11_17.csv', encoding='ISO-8859-1', engine='python')
    vuln_df['posteddate'] =  pd.to_datetime(vuln_df['posteddate'])
    # print(vuln_df[:10])
    cpe_weekly_df = cpeCountsByWeek(vuln_df, cve_cpe_map, start_date, end_date )
    cpe_weekly_df.to_csv('../../data/DW_data/new_DW/cpe_Counts_plot_weekly.csv')



    # cve_eventsDf = cve_eventsDf[cve_eventsDf['start_dates'] >= trainStart_date]
    # cve_eventsDf = cve_eventsDf[cve_eventsDf['start_dates'] < trainEnd_date]

    # getTopCVEs_byWeek(cve_eventsDf)

    # subspace = pd.read_pickle('../../data/DW_data/features/feat_forums/subspace_df_v01_05.pickle')

    # analyzeResiduals(subspace)

    # select_attack_days(cve_eventsDf)

if __name__ == "__main__":
    main()

