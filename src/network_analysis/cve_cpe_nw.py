import sys
sys.path.insert(0, '../network_analysis/')
sys.path.insert(0, '../load_data/')

import pandas as pd
import networkx as nx
import datetime as dt
import operator
import pickle
from sqlalchemy import create_engine
import numpy as np
import datetime
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


def shortest_pathLengths(network, sourceNodes, targetNodes):
    shortest_pathDist = []
    for s in sourceNodes:
        for t in targetNodes:
            try:
                shortest_pathDist.append(nx.shortest_path_length(network,source=s,target=t))
                # print(nx.shortest_path_length(network,source=s,target=t))
            except:
                continue # or some high value

    return shortest_pathDist


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


def plot_bars(data, xTicks, xLabels='', yLabels=''):
    hfont = {'fontname': 'Arial'}
    ind = np.arange(len(data))
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111)

    width=0.35
    rects1 = ax.bar(ind, data, width,
                    color='#0000ff')  # axes and labels
    ax.set_xlim(-width, len(ind) + width)
    # ax.set_ylim(87, 95)
    ax.set_ylabel(yLabels, size=30, **hfont)
    ax.set_xlabel(xLabels, size=30, **hfont)
    ax.set_xticks(ind)
    xtickNames = ax.set_xticklabels(xTicks, **hfont)
    plt.setp(xtickNames, rotation=45, fontsize=5)
    plt.grid(True)
    plt.xticks(size=20)
    plt.yticks(size=20)
    # plt.subplots_adjust(left=0.13, bottom=0.30, top=0.9)
    plt.subplots_adjust(left=0.13, bottom=0.25, top=0.9)
    ## add a legend
    # ax.legend( (rects1[0], ('Men', 'Women') )

    plt.show()
    plt.close()


def preprocessProb(network):
    transition_probs = {}
    userEdgeCount = {}
    for src, tgt in network.edges():
        if src not in transition_probs:
            transition_probs[src] = {}
            userEdgeCount[src] = 0
        if tgt not in transition_probs[src]:
            transition_probs[src][tgt] = 0
        if tgt not in transition_probs:
            transition_probs[tgt] = {}
            userEdgeCount[tgt] = 0
        transition_probs[src][tgt] += 1

        userEdgeCount[src] += 1

    for src in transition_probs:
        for tgt in transition_probs[src]:
            transition_probs[src][tgt] /= userEdgeCount[src]

    return transition_probs


def random_walk(network, source_node, expertGroup):
    transition_matrix = preprocessProb(network)
    lengthWalk = 50
    countPositive = 0
    totalCount = 0
    for idx in range(20):
        traversedNodes = []
        for l in range(lengthWalk):
            # nextNbrs = network.neighbors(source_node)
            transitionNext = transition_matrix[source_node]
            print(transitionNext)

            nodesChoices = []
            probChoices = []
            for nbr in transitionNext:
                if nbr in traversedNodes:
                    continue
                traversedNodes.append(nbr)
                nodesChoices.append(nbr)
                probChoices.append(transitionNext[nbr])

            if len(nodesChoices) == 0:
                source_node = np.random.choice(list(network.nodes()))[0]
                continue
            nextNode = np.random.choice(nodesChoices, 1, p=probChoices)[0]

            if nextNode in expertGroup:
                countPositive += 1
                break
            source_node = nextNode

        totalCount += 1

    return countPositive/totalCount


def computeCondProbAttackWeekly(networkKB, df_edgesWeekly, expertGroup, amEvents):
    probDist = []
    for w in range(len(df_edgesWeekly)):
        # print(w)
        if len(df_edgesWeekly[w]) == 0:
            probDist.append([])
            continue

        currProbList= []

        for i, r in df_edgesWeekly[w].iterrows():
            src = str(int(r['source']))
            tgt = str(int(r['target']))

            if (src, tgt) not in networkKB.edges():
                networkKB.add_edge(src, tgt)

            curr_Date = str(r['diffTime'])[:10]
            if curr_Date == -1:
                continue
            end_date = getNextWeekDate(curr_Date)

            print(curr_Date, end_date)
            # print(list(amEvents['date'])[0], type(list(amEvents['date'])[0]))
            # exit()
            amEventsNextWeek = amEvents[amEvents['date'] > curr_Date]
            amEventsNextWeek = amEventsNextWeek[amEventsNextWeek['date'] <= end_date]

            if len(amEventsNextWeek) == 0:
                continue

            print(len(amEventsNextWeek))
            probPositive = random_walk(networkKB, src, expertGroup)
            currProbList.append(probPositive)

        probDist.append(currProbList)
        print(currProbList)
    return probDist


def monthlyFeatureCompute(forums, start_date, users_CVE_map, CVE_users_map, vulData, cveCPE_data, amEvents, titles):
    '''
    One of the main issues here is the automation of the rolling basis dates of KB and training data

    :param forums:
    :param start_date:
    :param usersCVE:
    :return:
    '''
    KB_gap = 3
    titlesList = []
    feat_topUsers = []
    feat_experts = []
    # week_number = 14
    newUsersWeeklyPercList = []
    for idx in range(6):
        # KB network formation
        start_month = int(start_date[5:7]) + idx
        end_month = start_month + KB_gap
        if start_month < 10:
            start_monthStr = str('0') + str(start_month)
        else:
            start_monthStr = str(start_month)

        if end_month < 10:
            end_monthStr = str('0') + str(end_month)
        else:
            end_monthStr = str(end_month)

        start_dateCurr = start_date[:5] + start_monthStr + start_date[7:]
        end_dateCurr = start_date[:5] + end_monthStr + start_date[7:]
        print("KB info: ")
        print("Start date: ", start_dateCurr, " ,End date: ", end_dateCurr)
        df_KB = ldDW.getDW_data_postgres(forums, start_dateCurr, end_dateCurr)
        threadidsKB = list(set(df_KB['topicid']))
        KB_edges = ccon.storeEdges(df_KB, threadidsKB)

        networkKB, topUsersKB, usersCVE = splitUsers(users_CVE_map, KB_edges)

        # Find the experts in the KB
        topCVE = topCPEGroups(start_dateCurr, end_dateCurr, vulData, cveCPE_data, K=20)
        expertsDict = getExperts(topCVE, CVE_users_map, list(networkKB.nodes()))

        # Training network formation starts from here
        train_start_month = end_month
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

        print("Training data info: ")
        print("Start date: ", train_start_date, " ,End_date: ", train_end_date)

        df_train = ldDW.getDW_data_postgres(forums, train_start_date, train_end_date)
        train_edgesWeekly, expertUsersListWeekly, newUsersWeeklyPerc, weekList = \
            segmentPostsWeek(df_train, networkKB, cveCPE_data, vulData, CVE_users_map)

        newUsersWeeklyPercList.extend(newUsersWeeklyPerc)


        # Graph conductance
        # gcExperts = computeWeeklyConductance(networkKB, train_edgesWeekly, expertUsersListWeekly, list(expertsDict.keys()))
        # gc_Top = computeWeeklyConductance(networkKB, train_edgesWeekly, expertUsersListWeekly, topUsersKB)

        # Shortest path
        # spExperts = computeWeeklyShortestPath(networkKB, train_edgesWeekly, expertUsersListWeekly,
        #                                      list(expertsDict.keys()))
        # spTop = computeWeeklyShortestPath(networkKB, train_edgesWeekly, expertUsersListWeekly, topUsersKB)

        # Random Walk probability
        # probDistList = computeCondProbAttackWeekly(networkKB, train_edgesWeekly,
        #                                            list(expertsDict.keys()), amEvents)


        '''
        *****************************
        '''
        # feat_experts.extend(probDistList)
        # feat_topUsers.extend([])

        # print(len(gcExperts), len(gc_Top))
        # titlesList.extend(weekList)
        # for wnum in range(len(gc_Top)):
        #     title = start_date[:5] + ', ' + str(week_number + wnum)
        #     titlesList.append(title)

        # week_number += len(gc_Top)
    plot_bars(newUsersWeeklyPercList, titles, 'Date Time (start of week)', '%age of new users')
    return feat_experts, feat_topUsers, titlesList

