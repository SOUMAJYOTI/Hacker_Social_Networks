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

    while start_month <= 12:
        start_day = 1
        currIndex = 0
        daysList = []

        while True:
            eventType = []
            if start_day < 10:
                start_dayStr = str('0') + str(start_day)
            else:
                start_dayStr = str(start_day)
            start_date = datetime.datetime.strptime(
                str(start_year) + '-' + str(start_month) + '-' + start_dayStr + ' 00:00:00', '%Y-%m-%d %H:%M:%S')
            daysList.append(str(start_year) + '-' + str(start_month) + '-' + start_dayStr)

            end_date = datetime.datetime.strptime(str(start_year) + '-' + str(start_month) + '-' + start_dayStr + ' 23:59:00',
                                                  '%Y-%m-%d %H:%M:%S')

            events_currDay = eventsDf[eventsDf['DateTime'] >= start_date]
            events_currDay = events_currDay[events_currDay['DateTime'] < end_date]

            datesList.append(events_currDay)
            numEventsList.append(len(events_currDay))

            for idx, row in events_currDay.iterrows():
                eventType.append(row['event_type'])

            eventTypeList.append(eventType)

            start_day += 1
            if start_day > daysMonths[start_month]:
                break

        start_month += 1

        # Break condition
        events_currDay = eventsDf[eventsDf['DateTime'] >= start_date]
        if

    df_amEventsTS['date'] = datesList
    

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

    segmentEventDaily(amEvents)

    start_date = '2016-01-01'
    end_date = '2016-04-01'
    df_cve_cpe = pd.read_csv('../../data/DW_data/09_15/CVE_CPE_groups.csv')

    users_CVE_map, CVE_users_map = user_CVE_groups(df_cve_cpe, vulData)
    feat_experts, feat_topUsers, titlesList = \
        monthlyFeatureCompute(forums_cve_mentions, start_date, users_CVE_map, CVE_users_map, vulDataFiltered,
                              df_cve_cpe, amEvents, titles)
    # pickle.dump(feat_experts,
    #             open('../../data/DW_data/09_15/train/features/randomWalkProb_allExpertsKB_alltrainUsers.pickle', 'wb'))
    # pickle.dump(graphConductance_experts,
    #             open('../../data/DW_data/09_15/train/features/spath_top0.2KB_alltrainUsers.pickle', 'wb'))
    # pickle.dump(titlesList,
    #             open('../../data/DW_data/09_15/train/features/titles_weekly.pickle', 'wb'))

    # print(df[:10])
    # print(len(list(set(df['vendor']))))
    # print(len(list(set(df['product']))))
    # print(len(list(set(df['cluster_tag']))))

    # topCVE = topCPEGroups(start_date, end_date, vulDataFiltered, df )
    # users_CVE_map, CVE_users_map = user_CVE_groups(df, vulData)
    #
    # df_KB = ldDW.getDW_data_postgres(forums_cve_mentions, start_date, end_date)
    # threadidsKB = list(set(df_KB['topicid']))
    # KB_edges = ccon.storeEdges(df_KB, threadidsKB)
    #
    # networkKB, topUsersKB, usersCVE = splitUsers(users_CVE_map, KB_edges)
    # getRelUsers_inCPE(topCVE, CVE_users_map, list(networkKB.nodes()))
    # clustersDict = clusterDist(df)
    #
    # data = []
    # titlesList = []
    # for cl in clustersDict:
    #     data.append(clustersDict[cl])
    #     titlesList.append(cl)

    # plot_bars(data, titlesList)


    # exit()

    # df = pd.read_sql_query(query, con=engine)
    # df.to_csv('../../data/DW_data/09_15/CVE_CPE_groups.csv')
    # print("Number of cluster tags: ", len(list(set(df['cluster_tag']))))


    # print(len(users_CVE_map))

    # dw_user_edges_train = pickle.load(
    #     open('../../data/DW_data/09_15/train/edges/user_edges_selected_forums_Oct15-Mar16.pickle', 'rb'))
    #
    # network, topUsers, usersCVE = splitUsers(users_CVE_map, dw_user_edges_train)
    #
    # # cpe_groups = df['cluster_tag']
    # # print(cpe_groups)
    # # cveUsers(df)
    #
    # # results_df.to_csv('../../data/DW_data/08_20/DW_data_selected_forums_Jul16.csv')
    #
    # posts_train = pickle.load(open('../../data/DW_data/09_15/train/data/DW_data_selected_forums_Apr16.pickle', 'rb'))
    # df_edgesWeekly = segmentPostsWeek(posts_train, network)
    # computeWeeklyConductance(network, df_edgesWeekly, usersCVE)
