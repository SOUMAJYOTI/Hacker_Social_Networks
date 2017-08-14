import pandas as pd
import pickle
import operator
import matplotlib.pyplot as plt
import numpy as np
import networkx as nx

# TODO: Temporal evaluation of the ego network formation


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

        if (src, tgt) not in network_edgeList:
            network_edgeList.append((src, tgt))

    return network_edgeList


def mostReplyUsers(network_df, percentUsers):
    users_replyCount = {}

    for idx, row in network_df.iterrows():
        src = row['source']
        # tgt = row['target']

        if src not in users_replyCount:
            users_replyCount[src] = 0

        users_replyCount[src] += 1

    # Sort the users based on the count of the replies
    usersCount = len(users_replyCount)
    topUsers = sorted(users_replyCount.items(), key=operator.itemgetter(1), reverse=True)[int((percentUsers-0.1)*usersCount):int(usersCount*percentUsers)]

    countValList = []
    usersList = []
    for item, val in topUsers:
        usersList.append(item)
        countValList.append(val)

    return countValList, usersList


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


def createEgoNetwork(user, userNbrList, global_edge_list):
    # Consider the ego network
    alters = userNbrList[user]
    egoNw_Edges = []

    # Consider all pairs of edges possible among the alters
    for i in range(len(alters)-1):
        u_i = alters[i]
        for j in range(i+1, len(alters)):
            u_j = alters[j]
            if binarySearch(global_edge_list, (u_i, u_j)):
                egoNw_Edges.append((u_i, u_j))

            if binarySearch(global_edge_list, (u_j, u_i)):
                egoNw_Edges.append((u_j, u_i))

    return egoNw_Edges


def computeNwDistance(Users, userNbrList, global_edge_list):
    shortest_path_length = [[] for _ in range(1000)]
    for u in Users:
        # Consider the ego network
        alters = userNbrList[u]
        egoNw_Edges = []

        countUser = 0
        nbrs = userNbrList[u]
        userGraph = nx.DiGraph()

        for i in range(len(alters) - 1):
            u_i = alters[i]
            for j in range(i + 1, len(alters)):
                u_j = alters[j]
                if binarySearch(global_edge_list, (u_i, u_j)):
                        userGraph.add_node(u_i)
                        if u_j not in userGraph.nodes():
                            countUser += 1
                            userGraph.add_node(u_j)
                            userGraph.add_edge(*(u_i, u_j))
                            shortest_path_length[countUser].append(nx.average_shortest_path_length(userGraph))
                        else:
                            countUser += 1
                            if u_j not in userGraph.nodes():
                                userGraph.add_node(u_j)
                            userGraph.add_edge(*(u_i, u_j))
                            shortest_path_length[countUser].append(nx.average_shortest_path_length(userGraph))

                if binarySearch(global_edge_list, (u_j, u_i)):
                    if u_i not in userGraph.nodes():
                        userGraph.add_node(u_i)
                        if u_j not in userGraph.nodes():
                            userGraph.add_node(u_j)
                        userGraph.add_edge(*(u_j, u_i))
                        shortest_path_length[countUser].append(nx.average_shortest_path_length(userGraph))
                    else:
                        if u_j not in userGraph.nodes():
                            userGraph.add_node(u_j)
                        userGraph.add_edge(*(u_j, u_i))
                        shortest_path_length[countUser].append(nx.average_shortest_path_length(userGraph))
        countUser += 1

    return shortest_path_length


if __name__ == "__main__":
    # Load stored and preprocessed data
    network_edges = pickle.load(open('../../data/Network_stats/user_edges_Jan16_Mar16_forums_top5_CVE.pickle', 'rb'))
    network_userNbrs = pickle.load(open('../../data/Network_stats/user_nbrs_Jan16_Mar16_forums_top5_CVE.pickle', 'rb'))
    network_data = pickle.load(open('../../data/Mohammed/DW_user_edges_DataFrame_June15-June16.pickle', 'rb'))

    network_data['date'] = pd.to_datetime(network_data['date'])
    nw_data_time_slice = network_data[network_data['date'] > pd.to_datetime('2016-01-01')]
    nw_data_time_slice = nw_data_time_slice[nw_data_time_slice['date'] < pd.to_datetime('2016-03-31')]

    forums_cve_mentions = [259, 229, 88, 176, 159, 38]

    nw_forums_filtered = pd.DataFrame()
    for f in forums_cve_mentions:
        df_forum = nw_data_time_slice[nw_data_time_slice['forumid'] == f]
        nw_forums_filtered = pd.concat([nw_forums_filtered, df_forum], axis=0)

    # Stats 1: User out-degree distribution
    topUserPercent = [0.1]


    # for tup in topUserPercent:
    #     data, usersList = mostReplyUsers(nw_forums_filtered, tup)
    #     plot_hist(data, 20, xLabel='# Out-neighbors', yLabel='# Users', titleName='Top ' + str((tup-0.1)*100) + '%  to ' +
    #                                                 str(tup*100) + ' % Users by out-degree')

    # Stats 2: Store the edges of the network for faster computation
    # network_edges = store_edges(nw_forums_filtered)
    # pickle.dump(network_edges, open('../../data/Network_stats/user_edges_Jan16_Mar16_forums_top5_CVE.pickle', 'wb'))
    # print('Done')

    # Stats 3: Store the out-neighbors of the network users for faster computation
    # network_userNbrs = store_neighbors(nw_forums_filtered)
    # pickle.dump(network_userNbrs, open('../../data/Network_stats/user_nbrs_Jan16_Mar16_forums_top5_CVE.pickle', 'wb'))
    # print('Done')

    # Stats 4: Number of links in ego network
    # ego_edgesCount = []
    # countUser = 0
    # print('Total users: ', len(network_userNbrs))
    # for ego in network_userNbrs:
    #     print('Ego number: ', countUser)
    #     ego_edges = createEgoNetwork(ego, network_userNbrs, network_edges)
    #     ego_edgesCount.append(len(ego_edges))
    #     countUser += 1
    #
    # plot_histLine(ego_edgesCount, xLabel='# nodes', yLabel='# links in ego network', titleName='Top 10% users by out-degree')

    # Stats 5: Average shortest distance with user addition
    # shortest_distances = computeDistance(list(network_userNbrs.keys()), network_edges)
    # for idx in range(len(shortest_distances)):
    #     print(np.mean(np.array(shortest_distances[idx])))
