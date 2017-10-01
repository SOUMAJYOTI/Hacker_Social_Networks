import sys
sys.path.insert(0, '../network_analysis/')
sys.path.insert(0, '../load_data/')


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


def conductance(network, group_1,group_2 ):
    return conductance(network, group_1, group_2, weight=None)

def splitUsers(usersCVE_map, df_edges):
    trainUsersGlobal = list(set(df_edges['source']).union(set(df_edges['target'])))
    usersCVE = list(usersCVE_map.keys())
    trainUsersGlobal = [str(int(i)) for i in trainUsersGlobal]
    commonUsers = set(usersCVE).intersection(set(trainUsersGlobal))
    print(len(trainUsersGlobal))

    # compute the top users in the network of the last 6 months
    # print('Creating network....')
    nw_edges = usAn.store_edges(df_edges)
    edgeCounts = usAn.edgeCountPairs(df_edges)
    # print(np.mean(np.array(list(edgeCounts.values()))))
    # usAn.plot_hist(list(edgeCounts.values()), numBins=20, xLabel = 'Number of edges b/w node pairs', yLabel='Count')

    # Remove edges with count less than a threshold
    filteredEdges = []
    for src, tgt in nw_edges:
        # if edgeCounts[str(src)+'_'+str(tgt)] <= 3.:
        #     continue
        filteredEdges.append((str(int(src)), str(int(tgt))))

    # print("Number of edges before: ", len(edgeCounts))
    # print("Number of edges after: ", len(filteredEdges))
    network = nx.DiGraph()
    network.add_edges_from(filteredEdges)

    centValues = usAn.computeCentrality(network, 'OutDegree')
    sortedUsersByCent = sorted(centValues.items(), key=operator.itemgetter(1), reverse=True)
    K = int(0.2 * len(sortedUsersByCent)) # THIS IS IMPORTANT --- PERCENT OF ALL USERS AS TOP COUNT
    topUsers = []
    for u, val in sortedUsersByCent:
        topUsers.append(u)

    commExperts = set(usersCVE).intersection(set(topUsers))
    # print(len(list(commExperts)))

    # Form the two groups of nodes
    # Remove any common nodes between the two groups
    topUsers = list(set(topUsers).difference(commExperts))

    return network, topUsers, usersCVE


def segmentPostsWeek(posts, G):
    posts['DateTime'] = posts['posteddate'].map(str) + ' ' + posts['postedtime'].map(str)
    posts['DateTime'] = posts['DateTime'].apply(lambda x:
                                                    datetime.datetime.strptime(x, '%Y-%m-%d %H:%M:%S'))
    posts = posts.sort('DateTime', ascending=True)

    allUsers = list(G.nodes())
    newUsers = list(set(posts['uid']))
    newUsers = [str(int(i)) for i in newUsers]

    existingUsers = list(set(allUsers).intersection(newUsers))
    # print(len(newUsers), len(existingUsers))

    # print(posts[:10])
    # Form the network for each week
    start_year = posts['DateTime'].iloc[0].year
    start_month = posts['DateTime'].iloc[0].month

    start_day = 1
    currIndex = 0
    dfEges_WeeklyList = []
    daysMonths = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    numDaysCurrMonth = daysMonths[start_month-1]

    while True:
        if start_day < 10:
            start_dayStr = str('0') + str(start_day)
        else:
            start_dayStr = str(start_day)
        start_date = datetime.datetime.strptime(str(start_year)+'-'+str(start_month)+'-'+start_dayStr+' 00:00:00', '%Y-%m-%d %H:%M:%S')

        end_day = start_day + 7
        if end_day > numDaysCurrMonth:
            end_day = numDaysCurrMonth

        if end_day < 10:
            end_dayStr = str('0') + str(end_day)
        else:
            end_dayStr = str(end_day)
        end_date = datetime.datetime.strptime(str(start_year) + '-' + str(start_month) + '-' + end_dayStr + ' 23:59:00',
                                                '%Y-%m-%d %H:%M:%S')

        posts_currWeek = posts[posts['DateTime'] >= start_date]
        posts_currWeek = posts_currWeek[posts_currWeek['DateTime'] < end_date]

        topics = list(set(posts_currWeek['topicid']))
        df_edgesCurrWeek = ccon.storeEdges(posts_currWeek, topics)
        dfEges_WeeklyList.append(df_edgesCurrWeek)

        currIndex += len(posts_currWeek)
        start_day = end_day
        if start_day >= numDaysCurrMonth:
            break

    return dfEges_WeeklyList


def computeWeeklyConductance(networkKB, df_edgesWeekly, userGroup):
    graphConductanceDist = []
    networkNew = networkKB.copy()
    for w in range(len(df_edgesWeekly)):
        if len(df_edgesWeekly[w]) == 0:
            graphConductanceDist.append(0)

        currEdgeList = []
        userListWeek = []
        for i, r in df_edgesWeekly[w].iterrows():
            src = str(int(r['source']))
            tgt = str(int(r['target']))

            userListWeek.append(src)
            userListWeek.append(tgt)
            if (src, tgt) not in currEdgeList:
                currEdgeList.append((src, tgt))

        userListWeek = list(set(userListWeek))

        networkNew.add_edges_from(currEdgeList)
        userListWeek = list(set(userListWeek).difference(set(userGroup))) # remove the experts from the new users

        conductanceDist = nxCut.conductance(networkNew, userGroup, userListWeek)
        graphConductanceDist.append(conductanceDist)

    return graphConductanceDist

def monthlyFeatureCompute(forums, start_date, usersCVE):
    '''
    One of the main issues here is the automation of the rolling basis dates of KB and training data

    :param forums:
    :param start_date:
    :param usersCVE:
    :return:
    '''
    KB_gap = 3
    start_month = int(start_date[5:7])
    end_month = start_month + KB_gap
    if end_month < 10:
        end_monthStr = str('0') + str(end_month)
    else:
        end_monthStr = str(end_month)

    end_date = start_date[:5] + end_monthStr + start_date[7:]
    df_KB = ldDW.getDW_data_postgres(forums, start_date, end_date)

    threadids = list(set(df_KB['topicid']))
    KB_edges = ccon.storeEdges(df_KB, threadids)

    network, topUsers, usersCVE = splitUsers(users_CVE_map, KB_edges)

    train_start_month = end_month + 1
    train_end_month = train_start_month + 1
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

    df_train = ldDW.getDW_data_postgres(forums, train_start_date, train_end_date)
    threadidsTrain = list(set(df_train['topicid']))
    train_edges = ccon.storeEdges(df_train, threadidsTrain)

    df_edgesWeekly = segmentPostsWeek(posts_train, network)
    computeWeeklyConductance(network, df_edgesWeekly, usersCVE)



if __name__ == "__main__":
    forums_cve_mentions = [88, 248, 133, 49, 62, 161, 84, 60, 104, 173, 250, 105, 147, 40, 197]
    # engine = create_engine('postgresql://postgres:Impossible2@10.218.109.4:5432/cyber_events_pred')
    # query = "select vendor, product, cluster_tag, cve from  cve_cpegroups"
    posts_df = pickle.load(open('../../data/DW_data/09_15/train/data/DW_data_selected_forums_Oct15-Mar16.pickle', 'rb'))
    vulData = pickle.load(open('../../data/DW_data/08_29/Vulnerabilities-sample_v2+.pickle', 'rb'))

    start_date = '2016-01-01'
    monthlyFeatureCompute(start_date)
    exit()

    # df = pd.read_sql_query(query, con=engine)
    # df.to_csv('../../data/DW_data/09_15/CVE_CPE_groups.csv')
    df = pd.read_csv('../../data/DW_data/09_15/CVE_CPE_groups.csv')
    print("Number of cluster tags: ", len(list(set(df['cluster_tag']))))

    users_CVE_map, CVE_users_map = user_CVE_groups(df, vulData)
    # print(len(users_CVE_map))

    dw_user_edges_train = pickle.load(
        open('../../data/DW_data/09_15/train/edges/user_edges_selected_forums_Oct15-Mar16.pickle', 'rb'))

    network, topUsers, usersCVE = splitUsers(users_CVE_map, dw_user_edges_train)

    # cpe_groups = df['cluster_tag']
    # print(cpe_groups)
    # cveUsers(df)

    # results_df.to_csv('../../data/DW_data/08_20/DW_data_selected_forums_Jul16.csv')

    posts_train = pickle.load(open('../../data/DW_data/09_15/train/data/DW_data_selected_forums_Apr16.pickle', 'rb'))
    df_edgesWeekly = segmentPostsWeek(posts_train, network)
    computeWeeklyConductance(network, df_edgesWeekly, usersCVE)
