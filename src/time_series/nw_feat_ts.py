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
import userAnalysis as usAn
import numpy as np
import networkx.algorithms.cuts as nxCut
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
    users_CVEMap = {}
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
            if u not in users_CVEMap:
                users_CVEMap[u] = []

            users_CVEMap[u].append(cve)

    return users_CVEMap, CVE_usersMap


def getCVEinTopCPE_Groups(start_date, end_date, vulInfo, cveCPE, K):
    """

    :param start_date:
    :param end_date:
    :param vulInfo:
    :param cveCPE:
    :param K:
    :return: CVEs in top CPE groups
    """
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

    usersCVECount_topCPE = {}
    for u, count in usersSorted:
        if count >= threshold:
            usersCVECount_topCPE[u] = count

    return usersCVECount_topCPE


def countConversations(start_date, end_date, forums):
    start_year = 2016
    start_month = 4

    daysMonths = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    df_postsTS = pd.DataFrame()

    datesList = []
    numPostsList = []
    uidsList = []
    uidsCount = []

    while start_month <= 12:
        # print("Start Date:", start_date )
        postsDf = ldDW.getDW_data_postgres(forums, start_date, end_date)
        postsDf['DateTime'] = postsDf['posteddate'].map(str) + ' ' + postsDf['postedtime'].map(str)
        postsDf['DateTime'] = postsDf['DateTime'].apply(lambda x:
                                                        datetime.datetime.strptime(x, '%Y-%m-%d %H:%M:%S'))
        postsDf = postsDf.sort('DateTime', ascending=True)

        start_day = 1
        while True:
            usersTempList = []
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

            end_date = datetime.datetime.strptime(
                str(start_year) + '-' + start_monthStr + '-' + start_dayStr + ' 23:59:00',
                '%Y-%m-%d %H:%M:%S')

            posts_currDay = postsDf[postsDf['DateTime'] >= start_date]
            posts_currDay = posts_currDay[posts_currDay['DateTime'] < end_date]

            datesList.append(start_date)
            numPostsList.append(len(posts_currDay))

            for idx, row in posts_currDay.iterrows():
                usersTempList.append(row['uid'])

            uidsList.append(usersTempList)
            uidsCount.append(len(list(set(usersTempList))))
            start_day += 1

            # Break condition
            if start_day > daysMonths[start_month - 1]:
                break

        start_month += 1
        if start_month > 12:
            break
        if start_month < 10:
            start_monthStr = str('0') + str(start_month)
        else:
            start_monthStr = str(start_month)
        start_date = str(start_year) + '-' + start_monthStr + '-01'
        end_date = str(start_year) + '-' + start_monthStr + '-' + str(
                                                daysMonths[start_month - 1])

    df_postsTS['date'] = datesList
    df_postsTS['number_posts'] = numPostsList
    df_postsTS['users'] = uidsList
    df_postsTS['number_users'] = uidsCount

    return df_postsTS


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

            for idx, row in events_currDay.iterrows():
                eventType.append(row['event_type'])

            eventTypeList.append(eventType)

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
    df_amEventsTS['event_types'] = eventTypeList

    return df_amEventsTS


def Conductance(network, userG1, userG2):
    conductanceVal = nxCut.conductance(network, userG1, userG2)

    return conductanceVal


def computeFeatureTimeSeries(start_date, end_date, forums, cve_cpeData, vulnData, postsDailyDf, allPosts):
    KB_gap = 3
    titlesList = []
    feat_topUsers = []
    feat_experts = []
    newUsersWeeklyPercList = []
    conductanceList = []

    # Map of users with CVE to user and user to CVE
    users_CVEMap, CVE_usersMap = user_CVE_groups(cve_cpeData, vulnData)

    for idx in range(0, 6, 3):
        # KB network formation
        start_month = int(start_date[5:7]) + idx
        end_month = start_month + KB_gap
        if end_month > 12:
            end_month = 12

        if start_month < 10:
            start_monthStr = str('0') + str(start_month)
        else:
            start_monthStr = str(start_month)

        if end_month < 10:
            end_monthStr = str('0') + str(end_month)
        else:
            end_monthStr = str(end_month)

        start_dateCurr = start_date[:5] + start_monthStr + start_date[7:]
        if end_month == 12:
            end_dateCurr = start_date[:5] + 31 + start_date[7:]
        else:
            end_dateCurr = start_date[:5] + end_monthStr + start_date[7:]
        print("KB info: ")
        print("Start date: ", start_dateCurr, " ,End date: ", end_dateCurr)
        df_KB = allPosts[allPosts['posteddate'] >= start_dateCurr]
        df_KB = df_KB[df_KB['posteddate'] < end_dateCurr]
        # df_KB = ldDW.getDW_data_postgres(forums, start_dateCurr, end_dateCurr)
        threadidsKB = list(set(df_KB['topicid']))
        KB_edges = ccon.storeEdges(df_KB, threadidsKB)

        # Get the users who have mentioned CVEs in their posts
        usersCVE = list(users_CVEMap.keys())

        # networkKB, topUsersKB, usersCVE = splitUsers(users_CVEMap, KB_edges)

        # Find the experts in the KB
        # User Group
        # topCVE = getCVEinTopCPE_Groups(start_dateCurr, end_dateCurr, vulData, cve_cpeData, K=20)
        # expertsDict = getExperts(topCVE, CVE_usersMap, list(networkKB.nodes()))

        # Training network formation starts from here
        train_start_month = end_month
        train_end_month = train_start_month + KB_gap
        if train_start_month < 10:
            train_start_monthStr = str('0') + str(train_start_month)
        else:
            train_start_monthStr = str(train_start_month)

        if train_end_month < 10:
            train_end_monthStr = str('0') + str(train_end_month)
        else:
            train_end_monthStr = str(train_end_month)

        train_start_date = start_date[:5] + train_start_monthStr + start_date[7:]
        train_end_date = start_date[:5] + train_end_monthStr + start_date[7:]

        print("Training data info: ")
        print("Start date: ", train_start_date, " ,End_date: ", train_end_date)

        postsDailyDf_curr = postsDailyDf[postsDailyDf['date'] >= train_start_date]
        postsDailyDf_curr = postsDailyDf_curr[postsDailyDf_curr['date'] < train_end_date]

        # print(postsDailyDf_curr)
        dates_curr = list(postsDailyDf_curr['date'])
        mergeEdges = KB_edges.copy()
        for idx_date in range(len(dates_curr)):
            print("Computing for date:", dates_curr[idx_date])
            date_start_curr = str(dates_curr[idx_date])[:10]
            date_end_curr = str(dates_curr[idx_date+1])[:10]

            # print(date_start_curr, date_end_curr)
            df_daily = allPosts[allPosts['posteddate'] >= date_start_curr]
            df_daily = df_daily[df_daily['posteddate'] < date_end_curr]
            # df_daily = ldDW.getDW_data_postgres(forums, date_start_curr, date_end_curr)

            topics_curr = list(set(df_daily['topicid']))
            dailyEdges = ccon.storeEdges(df_daily, topics_curr)

            if len(dailyEdges) == 0:
                conductanceList.append(0)
                continue

            users_curr = list(set(dailyEdges['source']).intersection(set(dailyEdges['target'])))
            users_curr = [str(int(i)) for i in users_curr]
            # print(df_dailyEdges)
            # print("Merging edges...")
            mergeEgdes = ccon.network_merge(mergeEdges, dailyEdges) # Update the mergedEdges every day
            # df_KB = pd.concat([df_KB, df_daily])  # This is the new KB Dataframe with merged posts

            G = nx.DiGraph()
            G.add_edges_from(mergeEgdes)

            conductanceList.append(Conductance(G, usersCVE, users_curr))

    postsDailyDf['conductance'] = conductanceList
    return postsDailyDf

if __name__ == "__main__":
    titles = pickle.load(open('../../data/DW_data/09_15/train/features/titles_weekly.pickle', 'rb'))
    forums_cve_mentions = [88, 248, 133, 49, 62, 161, 84, 60, 104, 173, 250, 105, 147, 40, 197]
    vulData = pickle.load(open('../../data/DW_data/08_29/Vulnerabilities-sample_v2+.pickle', 'rb'))
    vulDataFiltered = vulData[vulData['forumID'].isin(forums_cve_mentions)]

    read_path = '../../data/Armstrong_data/eventsDF_v1.0-demo.csv'
    amEvents = pd.read_csv(read_path)

    df_cve_cpe = pd.read_csv('../../data/DW_data/09_15/CVE_CPE_groups.csv')

    start_date = '2016-01-01'
    end_date = '2016-12-01'

    # df_postsTS = countConversations(start_date, end_date, forums_cve_mentions)
    # print(df_postsTS)
    # pickle.dump(df_postsTS, open('../../data/DW_data/posts_daysV1.0.pickle', 'wb'))

    allPosts = pickle.load(open('../../data/DW_data/09_15/DW_data_selected_forums_2016.pickle', 'rb'))
    allPosts['posteddate'] = allPosts['posteddate'].map(str)
    df_postsTS = pickle.load(open('../../data/DW_data/posts_daysV1.0.pickle', 'rb'))
    df_postsTS = computeFeatureTimeSeries(start_date, end_date, forums_cve_mentions, df_cve_cpe, vulData, df_postsTS, allPosts)
    pickle.dump(df_postsTS, open('../../data/DW_data/posts_daysV2.0.pickle', 'wb'))

    # df_postsTS.plot(x='date', y='number_posts')
    # plt.grid()
    # plt.xticks(size=20)
    # plt.yticks(size=20)
    # plt.xlabel('Date Time frame', size=20)
    # plt.ylabel('Number of posts', size=20)
    # plt.subplots_adjust(left=0.13, bottom=0.15, top=0.9)
    #
    # plt.show()
    # plt.close()
    #
    # df_postsTS.plot(x='date', y='number_users')
    # plt.grid()
    # plt.xticks(size=20)
    # plt.yticks(size=20)
    # plt.xlabel('Date Time frame', size=20)
    # plt.ylabel('Number of Users', size=20)
    # plt.subplots_adjust(left=0.13, bottom=0.15, top=0.9)
    #
    # plt.show()
