import sys
sys.path.insert(0, '../network_analysis/')
sys.path.insert(0, '../load_data/')
sys.path.insert(0, '../stat_analysis/')


import load_data_api as ldap
import pandas as pd
import networkx as nx
import datetime as dt
import operator
import pickle
from sqlalchemy import create_engine
import numpy as np
import datetime
import createConnections as ccon
import load_dataDW as ldDW
import matplotlib.pyplot as plt


def dateToString(date):
    yearNum = str(date.year)
    monthNum = str(date.month)
    dayNum =str(date.day)
    if len(monthNum)<2:
        monthNum = "0"+monthNum
    if len(dayNum)<2:
        dayNum= "0"+dayNum
    return yearNum+"-"+monthNum+"-"+dayNum


def user_CVE_groups(cve_cpe_data, vul_data):
    usersCVE_map = {}
    CVE_usersMap = {}
    for cve in cve_cpe_data['cve']:
        vulnItems = vul_data[vul_data['vulnId'] == cve]
        users = vulnItems['users'].tolist()
        if cve not in CVE_usersMap:
            CVE_usersMap[cve] = []
        CVE_usersMap[cve].extend(users)

        if len(users) == 0:
            continue

        usersList = users[0]
        for u in usersList:
            # print(u)
            if u not in usersCVE_map:
                usersCVE_map[u] = []

            usersCVE_map[u].append(cve)

    return usersCVE_map, CVE_usersMap


def segmentEventDaily(eventsDf):
    eventsDf = eventsDf.sort('date', ascending=True)

    start_year = eventsDf['date'].iloc[0].year
    start_month = eventsDf['date'].iloc[0].month
    daysMonths = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    df_amEventsTS = pd.DataFrame()

    datesList = []
    numEventsList = []
    eventTypeList = []
    endpoint_malwareList = []
    malicious_destList = []
    while start_month <= 12:
        start_day = 1

        while True:
            eventType = []
            if start_day < 10:
                start_dayStr = str('0') + str(start_day)
            else:
                start_dayStr = str(start_day)

            if start_month < 10:
                start_monthStr = str('0') + str(start_month)
            else:
                start_monthStr = str(start_month)

            start_date = datetime.datetime.strptime(
                str(start_year) + '-' + start_monthStr + '-' + start_dayStr + ' 00:00:00', '%Y-%m-%d %H:%M:%S')

            end_date = datetime.datetime.strptime(str(start_year) + '-' + start_monthStr + '-' + start_dayStr + ' 23:59:00',
                                                  '%Y-%m-%d %H:%M:%S')

            events_currDay = eventsDf[eventsDf['date'] >= start_date]
            events_currDay = events_currDay[events_currDay['date'] < end_date]

            datesList.append(start_date)
            numEventsList.append(len(events_currDay))

            endpoint_malware = 0
            malicious_dest = 0

            for idx, row in events_currDay.iterrows():
                if row['event_type'] == 'endpoint-malware':
                    endpoint_malware += 1
                else:
                    malicious_dest += 1
                eventType.append(row['event_type'])

            eventTypeList.append(eventType)
            endpoint_malwareList.append(endpoint_malware)
            malicious_destList.append(malicious_dest)

            start_day += 1

            # Break condition
            if start_day > daysMonths[start_month-1]:
                break

        start_month += 1
        if start_month < 10:
            start_monthStr = str('0') + str(start_month)
        else:
            start_monthStr = str(start_month)

        # Break condition
        events_nextMonth = eventsDf[eventsDf['date'] >= \
                                  pd.to_datetime(str(start_year)+'-'+start_monthStr+'-01', format='%Y-%m-%d')]
        events_nextMonth = events_nextMonth[events_nextMonth['date'] <= \
                                    pd.to_datetime(str(start_year) + '-' + start_monthStr + '-' + str(daysMonths[start_month-1])
                                                   , format='%Y-%m-%d')]
        if len(events_nextMonth) == 0:
            break

    df_amEventsTS['date'] = datesList
    df_amEventsTS['number_events'] = numEventsList
    df_amEventsTS['malicious-dest'] = malicious_destList
    df_amEventsTS['endpoint-malware'] = endpoint_malwareList
    df_amEventsTS['event_types'] = eventTypeList


    return df_amEventsTS


def clusterDist(df):
    # print(df[:10])
    clusters = {}

    for idx, row in df.iterrows():
        if row['cluster_tag'] not in clusters:
            clusters[row['cluster_tag']] = 0

        clusters[row['cluster_tag']] += 1

    clustersSorted = sorted(clusters.items(), key=operator.itemgetter(1), reverse=True)[20:40]
    topClusters = {}
    for cl, val in clustersSorted:
        topClusters[cl] = val
    # print(clusters)
    return topClusters


def topCPEGroups(start_date, end_date, vulInfo, cveCPE, K):
    vulCurr = vulInfo[vulInfo['postedDate'] >= start_date]
    vulCurr = vulCurr[vulCurr['postedDate'] < end_date]

    vulnerab = vulCurr['vulnId']
    cveCPE_curr = cveCPE[cveCPE['cve'].isin(vulnerab)]
    topCPEs = {}
    for idx, row in cveCPE_curr.iterrows():
        clTag = row['cluster_tag']
        if clTag not in topCPEs:
            topCPEs[clTag] = 0

        topCPEs[clTag] += 1

    # print(topCPEs)
    if K==-1:
        K = len(topCPEs)
    topCPEs_sorted = sorted(topCPEs.items(), key=operator.itemgetter(1), reverse=True)[:K]
    topCPEsList = []
    for cpe, count in topCPEs_sorted:
        topCPEsList.append(cpe)

    topCVE = cveCPE[cveCPE['cluster_tag'].isin(topCPEsList)]

    return list(topCVE['cve'])


def getExperts(topCVE, CVE_userMap, usersGlobal):
    usersCVECount = {}
    for tc in topCVE:
        userCurrCVE = CVE_userMap[tc]
        if len(userCurrCVE) == 0:
            continue
        for u in list(set(userCurrCVE[0])):
            if u in usersGlobal:
                if u not in usersCVECount:
                    usersCVECount[u] = 0
                usersCVECount[u] += 1

    usersSorted = sorted(usersCVECount.items(), key=operator.itemgetter(1), reverse=True)
    mean_count = np.mean(np.array(list(usersCVECount.values())))
    threshold = 0

    usersCVECount_top = {}
    for u, count in usersSorted:
        if count >= threshold:
            usersCVECount_top[u] = count

    return usersCVECount_top


def getNextWeekDate(currDate):
    day = int(currDate[8:])

    day += 7

    if day > 29:
        return -1

    if day < 10:
        dayStr = str(0) + str(day)
    else:
        dayStr = str(day)

    return currDate[:7] + '-' + dayStr


if __name__ == "__main__":
    titles = pickle.load(open('../../data/DW_data/09_15/train/features/titles_weekly.pickle', 'rb'))
    forums_cve_mentions = [88, 248, 133, 49, 62, 161, 84, 60, 104, 173, 250, 105, 147, 40, 197]
    # engine = create_engine('postgresql://postgres:Impossible2@10.218.109.4:5432/cyber_events_pred')
    # query = "select vendor, product, cluster_tag, cve from  cve_cpegroups"
    # posts_df = pickle.load(open('../../data/DW_data/09_15/train/data/DW_data_selected_forums_Oct15-Mar16.pickle', 'rb'))
    vulData = pickle.load(open('../../data/DW_data/08_29/Vulnerabilities-sample_v2+.pickle', 'rb'))
    vulDataFiltered = vulData[vulData['forumID'].isin(forums_cve_mentions)]

    read_path = '../../data/Armstrong_data/eventsDF_v1.0-demo.csv'
    amEvents = pd.read_csv(read_path)
    amEvents['date'] = pd.to_datetime(amEvents['date'], format="%Y-%m-%d")
    amEvents = amEvents[amEvents['date'] < pd.to_datetime('2016-12-01')]

    df_ts = segmentEventDaily(amEvents)

    pickle.dump(df_ts, open('../../data/Armstrong_data/eventsDF_df_days.pickle', 'wb'))

    df_ts.plot(x='date', y='endpoint-malware')
    plt.grid()
    plt.xticks(size=20)
    plt.yticks(size=20)
    plt.xlabel('Date Time frame', size=20)
    plt.ylabel('Number of endpoint malware events', size=20)
    plt.subplots_adjust(left=0.13, bottom=0.15, top=0.9)

    plt.show()
