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


def splitUsers(usersCVE_map, df_edges):
    trainUsersGlobal = list(set(df_edges['source']).union(set(df_edges['target'])))
    usersCVE = list(usersCVE_map.keys())
    trainUsersGlobal = [str(int(i)) for i in trainUsersGlobal]
    commonUsers = set(usersCVE).intersection(set(trainUsersGlobal))
    # print(len(trainUsersGlobal))

    # compute the top users in the network of the last 6 months
    # print('Creating network....')
    nw_edges , nw_edges_multiple = usAn.store_edges(df_edges)
    # edgeCounts = usAn.edgeCountPairs(df_edges)
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
    for u, val in sortedUsersByCent[:K]:
        topUsers.append(u)

    commExperts = set(usersCVE).intersection(set(topUsers))
    # print(len(list(commExperts)))

    # Form the two groups of nodes
    # Remove any common nodes between the two groups
    topUsers = list(set(topUsers).difference(commExperts))

    return network, topUsers, usersCVE


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


def segmentPostsWeek(posts, G, cveCPE_data, vulData, CVE_users_map):
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
    weeksList = []
    expertUsersListWeekly = []

    while True:
        if start_day < 10:
            start_dayStr = str('0') + str(start_day)
        else:
            start_dayStr = str(start_day)
        start_date = datetime.datetime.strptime(str(start_year)+'-'+str(start_month)+'-'+start_dayStr+' 00:00:00', '%Y-%m-%d %H:%M:%S')
        weeksList.append(str(start_year)+'-'+str(start_month)+'-'+start_dayStr)

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
        usersCurrWeek = list(set(df_edgesCurrWeek['source']).union(set(df_edgesCurrWeek['target'])))

        # Find the experts in the training data of the week
        start_dateStr = str(start_year)+'-'+str(start_month)+'-'+start_dayStr
        end_dateStr = str(start_year) + '-' + str(start_month) + '-' + end_dayStr
        topCVETrain = topCPEGroups(start_dateStr, end_dateStr, vulData, cveCPE_data, K=-1)
        expertsDictTrain = getExperts(topCVETrain, CVE_users_map, usersCurrWeek)

        expertUsersListWeekly.append(expertsDictTrain)

        currIndex += len(posts_currWeek)
        start_day = end_day
        if start_day >= 29:
            break

    return dfEges_WeeklyList, expertUsersListWeekly, weeksList


def computeWeeklyConductance(networkKB, df_edgesWeekly, trainExpertsWeekly, userGroup):
    graphConductanceDist = []
    networkNew = networkKB.copy()
    for w in range(len(df_edgesWeekly)):
        if len(df_edgesWeekly[w]) == 0:
            graphConductanceDist.append(0)

        currEdgeList = []
        userListWeek = []
        trainExperts = trainExpertsWeekly[w] # expert users
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

        conductanceDist = nxCut.conductance(networkNew, userGroup, trainExperts)
        graphConductanceDist.append(conductanceDist)

    return graphConductanceDist


def computeWeeklyShortestPath(networkKB, df_edgesWeekly, trainExpertsWeekly, expertGroup):
    shortestPathDist = []
    networkNew = networkKB.copy()
    for w in range(len(df_edgesWeekly)):
        # print(w)
        if len(df_edgesWeekly[w]) == 0:
            shortestPathDist.append([])

        currEdgeList = []
        userListWeek = []
        trainExperts = trainExpertsWeekly[w] # expert users
        for i, r in df_edgesWeekly[w].iterrows():
            src = str(int(r['source']))
            tgt = str(int(r['target']))

            userListWeek.append(src)
            userListWeek.append(tgt)
            if (src, tgt) not in currEdgeList:
                currEdgeList.append((src, tgt))

        userListWeek = list(set(userListWeek))
        networkNew.add_edges_from(currEdgeList)

        shortestPathDist.append(shortest_pathLengths(networkNew, userListWeek, expertGroup))

    return shortestPathDist


def monthlyFeatureCompute(forums, start_date, users_CVE_map, CVE_users_map, vulData, cveCPE_data):
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
        train_edgesWeekly, expertUsersListWeekly, weekList = \
            segmentPostsWeek(df_train, networkKB, cveCPE_data, vulData, CVE_users_map)

        # Graph conductance
        # gcExperts = computeWeeklyConductance(networkKB, train_edgesWeekly, expertUsersListWeekly, list(expertsDict.keys()))
        # gc_Top = computeWeeklyConductance(networkKB, train_edgesWeekly, expertUsersListWeekly, topUsersKB)

        # Shortest path
        # spExperts = computeWeeklyShortestPath(networkKB, train_edgesWeekly, expertUsersListWeekly,
        #                                      list(expertsDict.keys()))
        # spTop = computeWeeklyShortestPath(networkKB, train_edgesWeekly, expertUsersListWeekly, topUsersKB)

        '''
        *****************************
        '''
        feat_experts.extend([])
        feat_topUsers.extend([])

        # print(len(gcExperts), len(gc_Top))
        titlesList.extend(weekList)
        # for wnum in range(len(gc_Top)):
        #     title = start_date[:5] + ', ' + str(week_number + wnum)
        #     titlesList.append(title)

        # week_number += len(gc_Top)

    return feat_experts, feat_topUsers, titlesList


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


def plot_bars(data, titles):
    width=0.35
    ind = np.arange(len(data))  # the x locations for the groups
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111)
    ## the bars
    # rects1 = ax.bar(ind, data_mean, width,
    #                 color='#C0C0C0')
    rects1 = ax.bar(ind, data, width,
                    color='#0000ff')  # axes and labels
    ax.set_xlim(-width, len(ind) + width)
    # ax.set_ylim(0, 200)
    ax.set_ylabel('Number of CVEs', size=40)
    # ax.set_xlabel('Date frame (start of each week)', size=30)
    ax.set_title('Distribution of CVEs within CPE groups', size=30)
    xTickMarks = titles
    ax.set_xticks(ind)
    xtickNames = ax.set_xticklabels(xTickMarks)
    plt.setp(xtickNames, rotation=45, fontsize=5, ha='right')
    plt.grid(True)
    plt.xticks(size=25)
    plt.yticks(size=25)
    plt.subplots_adjust(left=0.13, bottom=0.50, top=0.9)
    ## add a legend
    # ax.legend( (rects1[0], ('Men', 'Women') )

    plt.show()


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


def preprocessProb(network):
    transition_probs = {}
    userEdgeCount = {}
    for src, tgt in network:
        if src not in transition_probs:
            transition_probs[src] = {}
            userEdgeCount[src] = 0
        if tgt not in transition_probs[src]:
            transition_probs[src][tgt] = 0
        transition_probs[src][tgt] += 1

        userEdgeCount[src] += 1

    for src in transition_probs:
        for tgt in transition_probs[src]:
            transition_probs[src][tgt] /= userEdgeCount[src]

    return transition_probs


def random_walk(network, source_node, expertGroup):
    transition_matrix = preprocessProb(network)
    lengthWalk = 10
    countPositive = 0
    totalCount = 0
    for idx in range(50):
        traversedNodes = []
        for l in range(lengthWalk):
            nextNbrs = network.neighbors(source_node)
            transitionNext = transition_matrix[source_node]

            nodesChoices = []
            probChoices = []
            for nbr in transitionNext:
                if nbr in traversedNodes:
                    continue
                nodesChoices.append(nbr)
                probChoices.append(transition_matrix[source_node][nbr])

            nextNode = np.random.choice(nodesChoices, 1, p=probChoices)

            if nextNode in expertGroup:
                countPositive += 1
                break
        totalCount += 1

    return countPositive/totalCount


def computeCondProbAttackWeekly(networkKB, df_edgesWeekly, expertGroup):
    probDist = []
    for w in range(len(df_edgesWeekly)):
        # print(w)
        if len(df_edgesWeekly[w]) == 0:
            probDist.append([])

        currEdgeList = []
        userListWeek = []
        for i, r in df_edgesWeekly[w].iterrows():
            src = str(int(r['source']))
            tgt = str(int(r['target']))

            if (src, tgt) not in networkKB.edges():
                networkKB.add_edge((src, tgt))

            probPositive = random_walk(networkKB, src, expertGroup)



if __name__ == "__main__":
    forums_cve_mentions = [88, 248, 133, 49, 62, 161, 84, 60, 104, 173, 250, 105, 147, 40, 197]
    # engine = create_engine('postgresql://postgres:Impossible2@10.218.109.4:5432/cyber_events_pred')
    # query = "select vendor, product, cluster_tag, cve from  cve_cpegroups"
    # posts_df = pickle.load(open('../../data/DW_data/09_15/train/data/DW_data_selected_forums_Oct15-Mar16.pickle', 'rb'))
    vulData = pickle.load(open('../../data/DW_data/08_29/Vulnerabilities-sample_v2+.pickle', 'rb'))
    vulDataFiltered = vulData[vulData['forumID'].isin(forums_cve_mentions)]

    start_date = '2016-01-01'
    end_date = '2016-04-01'
    df_cve_cpe = pd.read_csv('../../data/DW_data/09_15/CVE_CPE_groups.csv')

    users_CVE_map, CVE_users_map = user_CVE_groups(df_cve_cpe, vulData)
    graphConductance_experts, graphConductance_topUsers, titlesList = \
        monthlyFeatureCompute(forums_cve_mentions, start_date, users_CVE_map, CVE_users_map, vulDataFiltered,
                              df_cve_cpe)
    # pickle.dump(graphConductance_experts,
    #             open('../../data/DW_data/09_15/train/features/spath_allExpertsKB_alltrainUsers.pickle', 'wb'))
    # pickle.dump(graphConductance_experts,
    #             open('../../data/DW_data/09_15/train/features/spath_top0.2KB_alltrainUsers.pickle', 'wb'))
    pickle.dump(titlesList,
                open('../../data/DW_data/09_15/train/features/titles_weekly.pickle', 'wb'))

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
