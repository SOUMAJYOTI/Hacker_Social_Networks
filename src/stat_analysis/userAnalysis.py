import pandas as pd
import pickle
import operator
import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
from sqlalchemy import create_engine


def binarySearch(alist, item):
    first = 0
    last = len(alist)-1
    found = False

    while first<=last and not found:
        midpoint = (first + last)//2
        if alist[midpoint] == item:
            found = True
        else:
            if item < alist[midpoint]:
                last = midpoint-1
            else:
                first = midpoint+1

    return found


def store_neighbors(network_df):
    network_nbrList = {}
    for idx, row in network_df.iterrows():
        src = row['source']
        tgt = row['target']
        rtTime = pd.to_datetime(row['date'])

        if src not in network_nbrList:
            network_nbrList[src] = []

        if tgt not in network_nbrList[src]:
            network_nbrList[src].append(tgt)

    return network_nbrList

def edgeCountPairs(network_df):
    edgeCount = {}
    for idx, row in network_df.iterrows():
        src = row['source']
        tgt = row['target']

        pairUsers = str(src) + '_' + str(tgt)
        if pairUsers not in edgeCount:
            edgeCount[pairUsers] = 0

        edgeCount[pairUsers] += 1

    return edgeCount

def store_edges(network_df):
    network_edgeList = []
    for idx, row in network_df.iterrows():
        src = row['source']
        tgt = row['target']

        if not binarySearch(network_edgeList, (src, tgt)):
            network_edgeList.append((src, tgt))

    return network_edgeList


def plot_hist(data, numBins, xLabel='', yLabel='', titleName=''):
    plt.figure()
    n, bins, patches = plt.hist(data, bins=numBins, facecolor='g')
    plt.xlabel(xLabel, size=25)
    plt.ylabel(yLabel, size=25)
    plt.title(titleName, size=25)
    plt.grid(True)
    plt.tick_params('x', labelsize=20)
    plt.tick_params('y', labelsize=20)
    plt.show()


def relevantInfUsers(centDict, newUsers):
    countRel = 0
    for uid in newUsers:
        if uid in centDict.keys():
            countRel += 1

    return countRel


def topKUsers(centDict, K):
    sortedDict = sorted(centDict.items(), key=operator.itemgetter(1), reverse=True)[:K]
    newCentDict = {}
    for key, v in sortedDict:
        newCentDict[key] = v

    return newCentDict


def computeCentrality(network, arg):
    if arg == "InDegree":
        cent = nx.in_degree_centrality(network)

    if arg == "OutDegree":
        cent = nx.out_degree_centrality(network)

    if arg == "Pagerank":
        cent = nx.pagerank(network)

    if arg == "core":
        cent = nx.core_number(network)

    if arg == 'neighbors':
        cent = nx.degree(network)

    return cent


def topThreadsUsers(data_df, topicIdsList, limit):
    '''
    :param data_df:
    :param topicIdsList:
    :param limit: percent of topics
    :return:
    '''
    topTopicsByCount = {}
    for tl in topicIdsList:
        topTopicsByCount[tl] = len(data_df[data_df['topicid'] == tl])

    sortedTopics = sorted(topTopicsByCount.items(), key=operator.itemgetter(1), reverse=True)

    topUsers = []
    for k, v in sortedTopics:
        users = data_df[data_df['topicid'] == k]['uid']
        topUsers.extend(users)

    return list(set(topUsers))


def getVulnMentUsers(data_df):
    userCVE = {}
    userlist = []
    for i, r in data_df.iterrows():
        for u in r['users']:
            if u not in userCVE:
                userCVE[u] = []

            userCVE[u].append(r['vulnId'])
            if u not in userlist:
                userlist.append(u)

    return userCVE, userlist

# def getTopUsersWithVulMent():

def commUsers(train, test):
    users_train = set(train['source']).union(set(train['target']))
    users_test = set(test['source']).union(set(test['target']))

    commUsersList = list(users_train.intersection(users_test))
    print("Total train users: ", len(list(users_train)))
    print("Users common: ", len(commUsersList))
    print("New users: ", len(users_train.difference(users_test)))



def plot_DegDist(data, title=''):
    sorted_X = sorted(set(data.values()))
    Y = list(data.values())
    distributionX = [Y.count(x) for x in sorted_X]
    plt.figure()
    plt.loglog(sorted_X, distributionX, 'ro', basex=2, basey=2)
    # plt.xlim([])
    plt.xlabel('Out-Degree (Neighbors)', size=35)
    plt.ylabel('Number of nodes', size=30)
    plt.xticks(size=30)
    plt.yticks(size=30)
    plt.title(title, size=25)
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    startDate = "2010-07-01"
    endDate = "2016-08-01"

    # Load the data
    # forums to be considered
    dw_user_edges_train = pickle.load(open('../../data/DW_data/09_15/train/edges/user_edges_selected_forums_Oct15-Mar16.pickle', 'rb'))
    dw_user_edges_test = pickle.load(open('../../data/DW_data/09_15/test/edges/user_edges_selected_forums_Apr16.pickle', 'rb'))

    # print(dw_user_edges_test[:10])
    # commUsers(dw_user_edges_train, dw_user_edges_test)

    # posts_df = pickle.load(open('../../data/DW_data/09_15/DW_data_selected_forums_Jan-Mar16.pickle', 'rb'))
    #
    # results_df = pd.DataFrame()
    #
    # threadids = list(set(posts_df['topicid']))
    # dw_user_edges = dw_user_edges[dw_user_edges['topicid'].isin(threadids)]
    # print(len(dw_user_edges))

    # topicsId_list = list(set(posts_df['topicid'].tolist()))
    # dw_user_edges = dw_user_edges[dw_user_edges['topicid'].isin(topicsId_list)]

    print('Creating network....')
    nw_edges = store_edges(dw_user_edges_train)
    network = nx.DiGraph()
    network.add_edges_from(nw_edges)

    print("Computing centralities....")
    c = computeCentrality(network, 'neighbors')
    plot_DegDist(c)
    exit()
    # pickle.dump(c, open('../../data/DW
    # _data/09_15/centralities/outDeg_Jan-Mar2016.pickle', 'wb'))

    c = pickle.load(open('../../data/DW_data/09_15/centralities/outDeg_Jan-Mar2016.pickle', 'rb'))
    # 2. Find top users by centrality
    perc = 0.2
    k = int(perc * len(c))
    topKUid = list(topKUsers(c, k).keys())

    # start_date = pd.to_datetime('2016-04-01')
    # end_date = pd.to_datetime('2016-06-01')
    # print("Load Vul Data...")
    # vulData = pickle.load(open("../../data/DW_data/08_29/Vulnerabilities-sample_v1+.pickle", 'rb'))
    # vulData['postedDate'] = pd.to_datetime(vulData['postedDate'])
    # vulData = vulData[vulData['postedDate'] > start_date]
    # vulData = vulData[vulData['postedDate'] < endDate]
    # pickle.dump(vulData, open('../../data/DW_data/VulnData/Vulnerabilities_Apr-May2016.pickle', 'wb'))

    # userCVE, userList = getVulnMentUsers(vulData)
    #
    # print("Total number of users: ", len(c))
    # print("Total number of top K users: ", len(topKUid))
    #
    # print("Number of top users with CVE mentions: ")
    # print(len(list(set(topKUid).intersection(set(userList)))))
    # print("Number of users with CVE mentions: ")
    # print(len(list(set(list(c.keys())).intersection(set(userList)))))




    # print(len(list(set(posts_df['uid']))))
    # pr = pickle.load(open('../../data/Network_stats/core_Jan-Mar2016.pickle', 'rb'))
    # print(pr)

    # 2. Find the relevant popular users to the history
    # posts_df_new = pd.read_csv('../../data/DW_data/08_20/DW_data_selected_forums_Jul16.csv')
    # for p in [0.1, 0.2, 0.3, 0.4]:
    #     k = int(p* len(pr))
    #
    #     topKUid = list(topKUsers(pr, k).keys())
    #     newThreadUsers = list(set(posts_df_new['uid'].tolist()))
    #
    #     commonUsers = list(set(topKUid).intersection(set(newThreadUsers)))
    #     print(p, len(commonUsers)/len(newThreadUsers))
