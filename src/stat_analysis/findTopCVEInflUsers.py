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
    start_date = pd.to_datetime('2016-01-01')
    end_date = pd.to_datetime('2016-03-01')

    results_df = pd.DataFrame()

    dw_user_edges = pickle.load(open('../../data/Mohammed/DW_user_edges_DataFrame_June15-June16.pickle', 'rb'))

    dw_user_edges['date'] = pd.to_datetime(dw_user_edges['date'])
    dw_user_edges = dw_user_edges[dw_user_edges['date'] > start_date]
    dw_user_edges = dw_user_edges[dw_user_edges['date'] < end_date]

    src = set(dw_user_edges['source'])
    tgt = set(dw_user_edges['target'])

    users = list(src.union(tgt))
    users = [str(users[i]) for i in range(len(users))]
    vulData = pickle.load(open('../../data/DW_data/VulnData/Vulnerabilities_Apr-May2016.pickle', 'rb'))

    userCVE, userList = getVulnMentUsers(vulData)

    print(len(userList))
    print(len(list(set(userList).intersection(set(users)))))
