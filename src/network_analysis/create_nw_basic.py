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


def plot_histLine(data, xLabel='', yLabel='', titleName=''):
    in_values = sorted(set(data))
    list_degree_values = list(data)
    in_hist = [list_degree_values.count(x) for x in in_values]

    plt.figure()
    plt.loglog(in_values, in_hist, basex=2, basey=2)
    # plt.xlim([1, 2 ** 14])
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


if __name__ == "__main__":
    startDate = "2010-07-01"
    endDate = "2016-08-01"

    results_df = pd.DataFrame()

    dw_user_edges = pickle.load(open('../../data/Mohammed/DW_user_edges_DataFrame_June15-June16.pickle', 'rb'))
    posts_df = pickle.load(open('../../data/DW_data/08_29/DW_data_selected_forums_Jan-Mar16.pickle', 'rb'))

    threadids = list(set(posts_df['topicid']))
    dw_user_edges = dw_user_edges[dw_user_edges['topicid'].isin(threadids)]
    # print(len(dw_user_edges))

    # topicsId_list = list(set(posts_df['topicid'].tolist()))
    # dw_user_edges = dw_user_edges[dw_user_edges['topicid'].isin(topicsId_list)]

    # print('Creating network....')
    nw_edges = store_edges(dw_user_edges)
    network = nx.DiGraph()
    network.add_edges_from(nw_edges)

    for n in network:
        if n == 5:
            print('hello')

