import pandas as pd
import pickle
import operator
import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
from sqlalchemy import create_engine
import queue
import datetime
import time


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


def replyTimeDist(nwData, topics):
    diff_data = []
    for topicid in topics:
        threads = nwData[nwData['topicid'] == topicid]
        threads.is_copy=False
        threads['DateTime'] = threads['posteddate'].map(str) + ' ' + threads['postedtime'].map(str)
        threads['DateTime'] = threads['DateTime'].apply(lambda x:
                                    datetime.datetime.strptime(x, '%Y-%m-%d %H:%M:%S'))
        threads = threads.sort('DateTime', ascending=True)

        # print(threads)
        count_rows = 0
        for i, r in threads.iterrows():
            if count_rows == 0:
                last_time = r['DateTime']
                count_rows += 1
                continue

            d1_ts = time.mktime(last_time.timetuple())
            d2_ts = time.mktime(r['DateTime'].timetuple())
            diff = int(int(d2_ts - d1_ts) / 60)
            count_rows += 1
            last_time = r['DateTime']

            diff_data.append(diff)

    return diff_data


def storeEdges(nwData, topics):
    source = []
    target = []
    tid = []
    postTime = []
    forumIds = []

    # print(nwData)
    for topicid in topics:
        threads = nwData[nwData['topicid'] == topicid]
        threads.is_copy=False
        threads['DateTime'] = threads['posteddate'].map(str) + ' ' + threads['postedtime'].map(str)
        threads['DateTime'] = threads['DateTime'].apply(lambda x:
                                                        datetime.datetime.strptime(x, '%Y-%m-%d %H:%M:%S'))
        threads = threads.sort_values(['DateTime'], ascending=True)

        userWindow = [] # keeps track of the previous users in the dynamic window
        lastTime  = 0 # keeps track of the previous post time for the current user
        meanTimeWindow = [] # keeps list of differences in replies
        edgesList = [] # to keep track of duplicate interactions
        for i, r in threads.iterrows():
            if len(userWindow) == 0:
                pass
            else:
                d1_ts = time.mktime(lastTime.timetuple())
                d2_ts = time.mktime(r['DateTime'].timetuple())
                diff = int(int(d2_ts - d1_ts) / 60)  # difference in minutes

                if len(meanTimeWindow) > 5:
                    meanDiffWindow = np.mean(meanTimeWindow)
                    if diff > meanDiffWindow:
                        while True:
                            remELem = userWindow.pop(0)
                            remTime = meanTimeWindow.pop(0)
                            meanDiffWindow = np.mean(meanTimeWindow) # calculate new mean after removal
                            if diff > meanDiffWindow and len(meanTimeWindow) > 1: # maintain  atleast one element in window
                                continue
                            else:
                                break

                for pu in range(len(userWindow)):
                    prevUser = userWindow[pu]
                    if prevUser == r['uid']:
                        continue
                    if (prevUser, r['uid']) in edgesList:
                        continue
                    source.append(prevUser)
                    target.append(r['uid'])
                    tid.append(r['topicid'])
                    postTime.append(r['DateTime'])
                    forumIds.append(r['forumsid'])
                    edgesList.append((prevUser, r['uid']))

                meanTimeWindow.append(diff)

            userWindow.append(r['uid'])
            lastTime = r['DateTime']

    df_edges = pd.DataFrame()
    df_edges['source'] = source
    df_edges['target'] = target
    df_edges['Topic'] = tid
    df_edges['diffTime'] = postTime
    df_edges['Forum'] = forumIds

    return df_edges

def network_merge(network_df1, network_df2):
    networkMergeEdges = []
    for i, r in network_df1.iterrows():
        s = str(int(r['source']))
        t = str(int(r['target']))

        networkMergeEdges.append((s, t))

    for i, r in network_df2.iterrows():
        s = str(int(r['source']))
        t = str(int(r['target']))

        networkMergeEdges.append((s, t))

    return list(set(networkMergeEdges))

# if __name__ == "__main__":
#     # startDate = "2010-04-01"
#     # endDate = "2016-05-01"
#
#     results_df = pd.DataFrame()
#     posts_df = pickle.load(open('../../data/DW_data/09_15/DW_data_selected_forums_Oct15-Mar16.pickle', 'rb'))
#
#     threadids = list(set(posts_df['topicid']))
#
#     # diff_data = replyTimeDist(posts_df, threadids)
#     df_edges = storeEdges(posts_df, threadids)
#     print(len(df_edges))
#     pickle.dump(df_edges, open('../../data/DW_data/09_15/user_edges_selected_forums_Oct15-Mar16.pickle', 'wb'))