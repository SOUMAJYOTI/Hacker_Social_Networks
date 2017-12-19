import pandas as pd
import networkx as nx
import operator
import pickle
# import userAnalysis as usAn
import numpy as np
import networkx.algorithms.cuts as nxCut
import datetime
import src.network_analysis.createConnections as ccon
import matplotlib.pyplot as plt
from dateutil.relativedelta import relativedelta
import multiprocessing
import gc
import re
from src.network_analysis.features import *
import time

# data = pd.read_pickle('../../data/DW_data/user_interStats_DeltaT_4_Sept16-Apr17_TP10.pickle')
# print(data)
# exit()

# Global storage structures used over all pool processing
forums = [88, 248, 133, 49, 62, 161, 84, 60, 104, 173, 250, 105, 147, 40, 197, 220
        , 179, 219, 265, 98, 150, 121, 35, 214, 266, 89, 71, 146, 107, 64,
                           218, 135, 257, 243, 211, 236, 229, 259, 176, 159, 38]

# forums = [88, ]
vulnInfo = pd.read_pickle('../../data/DW_data/09_15/Vulnerabilities-sample_v2+.pickle')
cve_cpe_DF = pd.read_csv('../../data/DW_data/cve_cpe_mapDF.csv')
cve_cpe_map = pickle.load(open('../../data/DW_data/cve_cpe_map.pickle', 'rb'))

# Map of users with CVE to user and user to CVE
users_CVEMap, CVE_usersMap = pickle.load(open('../../data/DW_data/users_CVE_map.pickle', 'rb'))


allPosts = pd.read_pickle('../../data/DW_data/dw_database_data_2016-17.pickle')
KB_edges_DS = pd.read_pickle('../../data/DW_data/KB_edges_df.pickle')
allPosts['forumsid'] = allPosts['forumsid'].astype(int)
allPosts['topicid'] = allPosts['topicid'].astype(int)
allPosts['postsid'] = allPosts['postsid'].astype(int)
allPosts['uid'] = allPosts['uid'].astype(int)


def dateToString(date):
    yearNum = str(date.year)
    monthNum = str(date.month)
    dayNum =str(date.day)
    if len(monthNum)<2:
        monthNum = "0"+monthNum
    if len(dayNum)<2:
        dayNum= "0"+dayNum
    return yearNum+"-"+monthNum+"-"+dayNum


def hasVersion(inputString):
    return re.findall(r'\.', inputString)


def topCPEGroups(start_date, end_date, K):
    '''

    :param start_date:
    :param end_date:
    :param K: -1 if all CPEs to be returned, else the top K
    :return: the top CPE groups
    '''

    # 1. List all the vulnerabilties within the time frame input
    # 2. Then group all the CVE/vulnerab by their cluster tags
    # 3. Return the top CPE/cluster tags

    # 1.
    vulCurr = vulnInfo[vulnInfo['postedDate'] >= start_date]
    vulCurr = vulCurr[vulCurr['postedDate'] < end_date]

    # 2.
    vulnerab = vulCurr['vulnId']
    cveCPE_curr = cve_cpe_DF[cve_cpe_DF['cve'].isin(vulnerab)]
    topCPEs = {}
    # print(cveCPE_curr)
    for idx, row in cveCPE_curr.iterrows():
        ''' MOdify the cluster tags to remove versions '''
        cluster_tags = str(row['cluster']).split(' | ')
        cluster_custom = []
        cluster_custom.append(cluster_tags[0])
        # for idx_cl in range(1, min(2, len(cluster_tags))): #### Change this
        #     ctag = cluster_tags[idx_cl]
        #     if ctag in cluster_custom:
        #         continue
        #
        #     ver = hasVersion(ctag)
        #     if len(ver) >= 2:
        #         continue
        #
        #     cluster_custom.append(ctag)

        cluster_final = ''
        for idx_cc in range(len(cluster_custom)):
            cluster_final += cluster_custom[idx_cc] + ' '

        cluster_final = " ".join(cluster_final.split())
        if cluster_final not in topCPEs:
            topCPEs[cluster_final] = 0

        topCPEs[cluster_final] += 1

    # 3.
    # print(topCPEs)
    if K==-1:
        K = len(topCPEs)
    topCPEs_sorted = sorted(topCPEs.items(), key=operator.itemgetter(1), reverse=True)[:K]

    topCPEsList = []
    for cpe, count in topCPEs_sorted:
        topCPEsList.append(cpe)

    # topCVE = cve_cpe_map[cve_cpe_map['cluster_tag'].isin(topCPEsList)]

    # return list(topCVE['cve'])
    return topCPEsList


def getExperts(users):
    '''

    :param users:
    :return:

    This has to be corrected - experts should have CVEs in/history of the KB timeframe - not have CVEs in any time frame

    '''
    experts = []
    users_with_CVEs = list(users_CVEMap.keys())
    for u in users:
        if u in users_with_CVEs:
            experts.append(u)

    return experts


def getExpertsCPE(users, CPE):
    # 1. For each user, gather all his CVEs
    # 2. For each of his CVEs, find whether the CPE of that CVE is in the input CPE group
    # 3. If yes to 2, add those users

    # 1.
    usersCPE = []
    for u in users:
        currUserCVE = users_CVEMap[u]
        # 2.
        for cve in currUserCVE:
            if CPE in cve_cpe_map[cve]:
                usersCPE.append(u)
                break

    return usersCPE


def computeExpertsTime(df, experts):
    expertsLastTime = {}
    for idx, row in df.iterrows():
        src = str(int(row['source']))
        tgt = str(int(row['target']))

        if src in experts:
            if src not in expertsLastTime:
                expertsLastTime[src] = row['diffTime']

        if tgt in experts:
            if tgt not in expertsLastTime:
                expertsLastTime[tgt] = 0.

            expertsLastTime[tgt] = row['diffTime']

    return expertsLastTime


def computeFeatureTimeSeries_Users(start_date, end_date):
    numUsersList = []
    numThreadsList = []
    numNewUsersList = []
    numNewInterExp_ExpList = []
    numNewInterExp_NonExpList = []
    numInterExp_ExpList = []
    numInterExp_NonExpList = []
    timeDiff_expList = []

    datesList = []

    # KB timeframe 3 months prior to start of daily training data
    KBStartDate = start_date - relativedelta(months=3)

    KB_edges = KB_edges_DS[KBStartDate]
    KB_users = list(set(KB_edges['source']).intersection(set(KB_edges['target'])))
    KB_users = [str(int(i)) for i in KB_users]
    experts = getExperts(KB_users)

    expertsLastTime = computeExpertsTime(KB_edges, experts)

    currStartDate = start_date
    currEndDate = start_date + datetime.timedelta(days=1)
    # print(" Month: ", currStartDate.date())

    # Store the KB pairs - edges
    KB_pairs = []
    for idx_KB, row_KB in KB_edges.iterrows():
        src = str(int(row_KB['source']))
        tgt = str(int(row_KB['target']))
        if (src, tgt) not in KB_pairs:
            KB_pairs.append((src, tgt))

    ''' Feature computation starts '''
    while currEndDate < end_date:
        print("Date ", currStartDate.date(), )

        '''' Consider forums posts for current day '''
        df_currDay = allPosts[allPosts['posteddate'] > (currStartDate.date() - datetime.timedelta(days=2))]
        df_currDay = df_currDay[df_currDay['posteddate'] <= currEndDate.date()]
        df_currDay = df_currDay[df_currDay['forumsid'].isin(forums)]
        threadidsCurrDay = list(set(df_currDay['topicid']))
        currDay_edges = ccon.storeEdges(df_currDay, threadidsCurrDay)

        numInterExp_Exp = 0
        numNewInterExp_Exp = 0
        numInterExp_NonExp = 0
        numNewInterExp_NonExp = 0
        timeDiff = 0
        countInst = 0
        maxTimeDiff = 0
        for idx_curr, row_curr in currDay_edges.iterrows():
            src = str(int(row_curr['source']))
            tgt = str(int(row_curr['target']))

            if src in experts and tgt in experts:
                numInterExp_Exp += 1
                if (src, tgt) not in KB_pairs:
                    numNewInterExp_Exp += 1

            if (src in experts and tgt not in experts) or (src not in experts and tgt in experts):
                numInterExp_NonExp += 1
                if (src, tgt) not in KB_pairs:
                    numNewInterExp_NonExp += 1

            # Time difference between the experts and the user
            t_user = row_curr['diffTime']
            d1_ts = time.mktime(t_user.timetuple())
            if src in experts and src in expertsLastTime:
                t_exp = expertsLastTime[src]
                d2_ts = time.mktime(t_exp.timetuple())

                diff = (d1_ts - d2_ts)

                if diff > 0:
                    timeDiff += diff
                    countInst += 1

                if diff > maxTimeDiff:
                    maxTimeDiff = diff

        if countInst == 0:
            timeDiff = 0
        else:
            timeDiff /= countInst

        ''' Normalize it '''
        if maxTimeDiff != 0:
            timeDiff /= maxTimeDiff

        ''' Update te last seen time for experts '''
        # The update should be done only after \Delta t days,
        # for idx_curr, row_curr in currDay_edges.iterrows():
        #     tgt = str(int(row_curr['target']))
        #
        #     if tgt in experts:
        #         expertsLastTime[tgt] = row_curr['diffTime']

        numInterExp_ExpList.append(numInterExp_Exp)
        numNewInterExp_ExpList.append(numNewInterExp_Exp)
        numInterExp_NonExpList.append(numInterExp_NonExp)
        numNewInterExp_NonExpList.append(numNewInterExp_NonExp)
        timeDiff_expList.append(timeDiff)
        numThreadsList.append(len(threadidsCurrDay))

        users_curr = list(set(currDay_edges['source']).intersection(set(currDay_edges['target'])))
        numUsersList.append(len(users_curr))

        users_curr = [str(int(i)) for i in users_curr]
        numNewUsersList.append(len(list(set(users_curr).difference(set(KB_users)))))

        datesList.append(currStartDate)
        currStartDate = currStartDate + datetime.timedelta(days=1)
        currEndDate = currStartDate + datetime.timedelta(days=1)

    ''' Put the data into a dataframe '''
    feat_df = pd.DataFrame()
    feat_df['date'] = datesList
    feat_df['numThreads'] = numThreadsList
    feat_df['numUsers'] = numUsersList
    feat_df['numNewUsers'] = numNewUsersList
    feat_df['expertsInteractions'] = numInterExp_ExpList
    feat_df['expertsNewInteractions'] = numNewInterExp_ExpList
    feat_df['expert_NonInteractions'] = numInterExp_NonExpList
    feat_df['expert_NonNewInteractions'] = numNewInterExp_NonExpList
    feat_df['expertReplyTimeDiff'] = timeDiff_expList

    return feat_df


def computeFeatureTimeSeries_Graph(start_date, end_date,):
    condExpertsList = {}
    commThreadsList = {}
    shortPathList = {}
    commutePathList = {}
    communityCountList = {}

    datesList = []
    featDF = pd.DataFrame()

    # KB timeframe 3 months prior to start of daily training data
    KBStartDate = start_date - relativedelta(months=3)

    # while currEndDate < end_date:
    KB_edges = KB_edges_DS[KBStartDate]
    KB_users = list(set(KB_edges['source']).intersection(set(KB_edges['target'])))
    KB_users = [str(int(i)) for i in KB_users]
    experts = getExperts(KB_users)

    topCPEs = topCPEGroups(KBStartDate, start_date, K=10)

    currStartDate = start_date
    currEndDate = start_date + datetime.timedelta(days=1)
    # print(" Month: ", currStartDate.date())

    G_KB = nx.DiGraph()
    G_KB.add_edges_from(KB_edges)

    lapl_mat = (nx.laplacian_matrix(G_KB.to_undirected())).todense()
    # print(lapl_mat.shape)
    print('Computing pseudo lapl')
    pseudo_lapl_mat = np.linalg.pinv(lapl_mat)  # Compute the pseudo-inverse of the graph laplacian

    # Store the KB pairs
    KB_pairs = []
    for idx_KB, row_KB in KB_edges.iterrows():
        src = str(row_KB['source'])
        tgt = str(row_KB['target'])
        if(src, tgt) not in KB_pairs:
            KB_pairs.append((src, tgt))

    while currEndDate < end_date:
        print("Date ", currStartDate.date(), )
        # users_currDay = postsDailyDf[postsDailyDf['date'] == currStartDate]
        # Compute the feature for each forum separately

        # users_currDay_forum = users_currDay[users_currDay['forum'] == float(f)]

        '''' Consider forums posts for current day '''
        df_currDay = allPosts[allPosts['posteddate'] > (currStartDate.date() - datetime.timedelta(days=2))]
        df_currDay = df_currDay[df_currDay['posteddate'] <= currEndDate.date()]
        df_currDay = df_currDay[df_currDay['forumsid'].isin(forums)]
        threadidsCurrDay = list(set(df_currDay['topicid']))
        currDay_edges = ccon.storeEdges(df_currDay, threadidsCurrDay)

        # If there are no edges, no feature computation
        if len(currDay_edges) == 0:
            # condList.append(0)
            for cpe_count in range(len(topCPEs)):
                key = 'CPE_R' + str(cpe_count)
                if key not in condExpertsList:
                    condExpertsList[key] = []
                    commThreadsList[key] = []
                    shortPathList[key] = []
                    commutePathList[key] = []
                    communityCountList[key] = []
                condExpertsList[key].append(0)
                commThreadsList[key].append(0)
                shortPathList[key].append(0)
                commutePathList[key].append(0)
                communityCountList[key].append(0)

            # prList.append(0)
            # degList.append(0)
            datesList.append(currStartDate)
            currStartDate = currStartDate + datetime.timedelta(days=1)
            currEndDate = currStartDate + datetime.timedelta(days=1)
            continue

        users_curr = list(set(currDay_edges['source']).intersection(set(currDay_edges['target'])))

        # If there are no users, no feature computation
        if len(users_curr) == 0:
            # condList.append(0)
            for cpe_count in range(len(topCPEs)):
                key = 'CPE_R' + str(cpe_count)
                if key not in condExpertsList:
                    condExpertsList[key] = []
                    commThreadsList[key] = []
                    shortPathList[key] = []
                    commutePathList[key] = []
                    communityCountList[key] = []
                condExpertsList[key].append(0)
                commThreadsList[key].append(0)
                shortPathList[key].append(0)
                commutePathList[key].append(0)
                communityCountList[key].append(0)

            # prList.append(0)
            # degList.append(0)
            datesList.append(currStartDate)
            currStartDate = currStartDate + datetime.timedelta(days=1)
            currEndDate = currStartDate + datetime.timedelta(days=1)
            continue

        users_curr = [str(int(i)) for i in users_curr]

        # print("Merging edges...")
        mergeEgdes = ccon.network_merge(KB_edges.copy(), currDay_edges)  # Update the mergedEdges every day


        G = nx.DiGraph()
        G.add_edges_from(mergeEgdes)

        countCPE = 1
        for cpe in topCPEs:
            # print("Forum:", f, " Month: ", currStartDate.date(), cpe)
            expertUsersCPE = getExpertsCPE(experts, cpe)
            key = 'CPE_R' + str(countCPE)
            if key not in condExpertsList:
                condExpertsList[key] = []
                commThreadsList[key] = []
                shortPathList[key] = []
                commutePathList[key] = []
                communityCountList[key] = []

            ''' Store the computed features by CPE'''
            # condExpertsList[key].append(Conductance(G, expertUsersCPE, users_curr))
            # commThreadsList[key].append(threadCommon(df_currDay, expertUsersCPE))
            # shortPathList[key].append(shortestPaths(G, expertUsersCPE, users_curr))
            # communityCountList[key].append(community_detect(G, expertUsersCPE, users_curr))
            commutePathList[key].append(commuteTime(G, pseudo_lapl_mat, expertUsersCPE, users_curr))
            # print(commuteTime(G, expertUsersCPE, users_curr))

            countCPE += 1

        datesList.append(currStartDate)

        currStartDate = currStartDate + datetime.timedelta(days=1)
        currEndDate = currStartDate + datetime.timedelta(days=1)

    featDF['date'] = datesList

    for k in range(10):
        try:
            # featDF['shortestPaths_CPE_R' + str(k+1)] = shortPathList['CPE_R' + str(k+1)]
            featDF['commuteTime_CPE_R' + str(k+1)] = commutePathList['CPE_R' + str(k+1)]
            # featDF['communityCount_CPE_R' + str(k + 1)] = communityCountList['CPE_R' + str(k + 1)]
            # featDF['expertsThreads_CPE_R' + str(k+1)] = commThreadsList['CPE_R' + str(k+1)]
            # featDF['CondExperts_CPE_R' + str(k+1)] = condExpertsList['CPE_R' + str(k+1)]
        except:
            # featDF['shortestPaths_CPE_R' + str(k + 1)] = 0.0
            featDF['communityCount_CPE_R' + str(k + 1)] = 0.0
            # featDF['commuteTime_CPE_R' + str(k+1)] = 0.0
            # featDF['expertsThreads_CPE_R' + str(k + 1)] = 0.0
            # featDF['CondExperts_CPE_R' + str(k + 1)] = 0.0

    return featDF


def computeFeatureTimeSeriesSeparateForums(start_date, end_date,):
    # titlesList = []
    # feat_topUsers = []
    # feat_experts = []
    # newUsersWeeklyPercList = []

    condList = []
    condExpertsList = []
    prList = []
    degList = []

    datesList = []
    forumsList = []
    featDF = pd.DataFrame()

    # KB timeframe 3 months prior to start of daily training data
    KBStartDate = start_date - relativedelta(months=3)

    # while currEndDate < end_date:
    KB_edges = KB_edges_DS[KBStartDate]
    KB_users = list(set(KB_edges['source']).intersection(set(KB_edges['target'])))
    KB_users = [str(int(i)) for i in KB_users]
    experts = getExperts(KB_users)

    for f in forums:
        currStartDate = start_date
        currEndDate = start_date + datetime.timedelta(days=1)
        print("Forum:", f, " Month: ", currStartDate.date())

        while currEndDate < end_date:
            # users_currDay = postsDailyDf[postsDailyDf['date'] == currStartDate]
            # Compute the feature for each forum separately

            # users_currDay_forum = users_currDay[users_currDay['forum'] == float(f)]

            '''' Consider forums posts for current forum for current day'''
            df_currDay = allPosts[allPosts['posteddate'] >= (currStartDate.date() - datetime.timedelta(days=4))]
            df_currDay = df_currDay[df_currDay['posteddate'] < currEndDate.date()]
            df_currDay = df_currDay[df_currDay['forumsid'] == f]
            threadidsCurrDay = list(set(df_currDay['topicid']))
            currDay_edges = ccon.storeEdges(df_currDay, threadidsCurrDay)

            # If there are no edges, no feature computation
            if len(currDay_edges) == 0:
                condList.append(0)
                condExpertsList.append(0)
                prList.append(0)
                degList.append(0)
                forumsList.append(f)
                datesList.append(currStartDate)
                currStartDate = currStartDate + datetime.timedelta(days=1)
                currEndDate = currStartDate + datetime.timedelta(days=1)
                continue

            users_curr = list(set(currDay_edges['source']).intersection(set(currDay_edges['target'])))
            # If there are no users, no feature computation
            if len(users_curr) == 0:
                condList.append(0)
                condExpertsList.append(0)
                prList.append(0)
                degList.append(0)
                forumsList.append(f)
                datesList.append(currStartDate)
                currStartDate = currStartDate + datetime.timedelta(days=1)
                currEndDate = currStartDate + datetime.timedelta(days=1)
                continue

            users_curr = [str(int(i)) for i in users_curr]

            # print("Merging edges...")
            mergeEgdes = ccon.network_merge(KB_edges.copy(), currDay_edges)  # Update the mergedEdges every day

            G = nx.DiGraph()
            G.add_edges_from(mergeEgdes)

            condList.append(Conductance(G, KB_users, users_curr))
            condExpertsList.append(Conductance(G, experts, users_curr))
            prList.append(centralities(G, 'PageRank', users_curr))
            degList.append(centralities(G, 'OutDegree', users_curr))
            forumsList.append(f)
            datesList.append(currStartDate)

            currStartDate = currStartDate + datetime.timedelta(days=1)
            currEndDate = currStartDate + datetime.timedelta(days=1)

    featDF['date'] = datesList
    featDF['forum'] = forumsList
    featDF['conductance'] = condList
    featDF['conductanceExperts'] = condExpertsList
    featDF['pagerank'] = prList
    featDF['degree'] = degList

    return featDF


def computeFeatureTimeSeriesForumsCPE(start_date, end_date):
    condList = []
    condExpertsList = {}
    commThreadsList = {}
    shortPathList = {}
    commutePathList = {}
    communityCountList = {}
    prList = []
    degList = []

    datesList = []
    forumsList = []
    featDF = pd.DataFrame()

    # KB timeframe 3 months prior to start of daily training data
    KBStartDate = start_date - relativedelta(months=3)

    # while currEndDate < end_date:
    KB_edges = KB_edges_DS[KBStartDate]
    KB_users = list(set(KB_edges['source']).intersection(set(KB_edges['target'])))
    KB_users = [str(int(i)) for i in KB_users]
    experts = getExperts(KB_users)

    topCPEs = topCPEGroups(KBStartDate, start_date, K=10)

    for f in forums:
        currStartDate = start_date
        currEndDate = start_date + datetime.timedelta(days=1)
        print("Forum:", f, " Month: ", currStartDate.date())

        while currEndDate < end_date:
            # print("Forum:", f, " Month: ", currStartDate.date(), )
            # users_currDay = postsDailyDf[postsDailyDf['date'] == currStartDate]
            # Compute the feature for each forum separately

            # users_currDay_forum = users_currDay[users_currDay['forum'] == float(f)]

            '''' Consider forums posts for current forum for current day '''
            df_currDay = allPosts[allPosts['posteddate'] >= (currStartDate.date() - datetime.timedelta(days=4))]
            df_currDay = df_currDay[df_currDay['posteddate'] < currEndDate.date()]
            df_currDay = df_currDay[df_currDay['forumsid'] == f]
            threadidsCurrDay = list(set(df_currDay['topicid']))
            currDay_edges = ccon.storeEdges(df_currDay, threadidsCurrDay)

            # If there are no edges, no feature computation
            if len(currDay_edges) == 0:
                # condList.append(0)
                for cpe_count in range(len(topCPEs)):
                    key = 'CPE_R' + str(cpe_count)
                    if key not in condExpertsList:
                        condExpertsList[key] = []
                        commThreadsList[key] = []
                        shortPathList[key] = []
                        commutePathList[key] = []
                        communityCountList[key] = []
                    condExpertsList[key].append(0)
                    commThreadsList[key].append(0)
                    shortPathList[key].append(0)
                    commutePathList[key].append(0)
                    communityCountList[key].append(0)

                # prList.append(0)
                # degList.append(0)
                forumsList.append(f)
                datesList.append(currStartDate)
                currStartDate = currStartDate + datetime.timedelta(days=1)
                currEndDate = currStartDate + datetime.timedelta(days=1)
                continue

            users_curr = list(set(currDay_edges['source']).intersection(set(currDay_edges['target'])))
            # If there are no users, no feature computation
            if len(users_curr) == 0:
                # condList.append(0)
                for cpe_count in range(len(topCPEs)):
                    key = 'CPE_R' + str(cpe_count)
                    if key not in condExpertsList:
                        condExpertsList[key] = []
                        commThreadsList[key] = []
                        shortPathList[key] = []
                        commutePathList[key] = []
                        communityCountList[key] = []
                    condExpertsList[key].append(0)
                    commThreadsList[key].append(0)
                    shortPathList[key].append(0)
                    commutePathList[key].append(0)
                    communityCountList[key].append(0)

                # prList.append(0)
                # degList.append(0)
                forumsList.append(f)
                datesList.append(currStartDate)
                currStartDate = currStartDate + datetime.timedelta(days=1)
                currEndDate = currStartDate + datetime.timedelta(days=1)
                continue

            users_curr = [str(int(i)) for i in users_curr]

            # print("Merging edges...")
            mergeEgdes = ccon.network_merge(KB_edges.copy(), currDay_edges)  # Update the mergedEdges every day

            G = nx.DiGraph()
            G.add_edges_from(mergeEgdes)

            lapl_mat = (nx.laplacian_matrix(G.to_undirected())).todense()
            # print(lapl_mat.shape)
            # print('Computing pseudo lapl')
            # pseudo_lapl_mat = np.linalg.pinv(lapl_mat)  # Compute the pseudo-inverse of the graph laplacian
            # print('done..')

            countCPE = 1
            for cpe in topCPEs:
                # print("Forum:", f, " Month: ", currStartDate.date(), cpe)
                expertUsersCPE = getExpertsCPE(experts, cpe)
                key = 'CPE_R' + str(countCPE)
                if key not in condExpertsList:
                    condExpertsList[key] = []
                    commThreadsList[key] = []
                    shortPathList[key] = []
                    commutePathList[key] = []
                    communityCountList[key] = []

                ''' Store the computed features by CPE'''
                # condExpertsList[key].append(Conductance(G, expertUsersCPE, users_curr))
                # commThreadsList[key].append(threadCommon(df_currDay, expertUsersCPE))
                shortPathList[key].append(shortestPaths(G, expertUsersCPE, users_curr))
                # communityCountList[key].append(community_detect(G, expertUsersCPE, users_curr))
                # commutePathList[key].append(commuteTime(G, pseudo_lapl_mat, expertUsersCPE, users_curr))
                # print(commuteTime(G, expertUsersCPE, users_curr))

                countCPE += 1

            forumsList.append(f)
            datesList.append(currStartDate)

            currStartDate = currStartDate + datetime.timedelta(days=1)
            currEndDate = currStartDate + datetime.timedelta(days=1)

    featDF['date'] = datesList
    featDF['forum'] = forumsList

    for k in range(10):
        try:
            featDF['shortestPaths_CPE_R' + str(k+1)] = shortPathList['CPE_R' + str(k+1)]
            # featDF['commuteTime_CPE_R' + str(k+1)] = commutePathList['CPE_R' + str(k+1)]
            # featDF['communityCount_CPE_R' + str(k+1)] = communityCountList['CPE_R' + str(k+1)]
            # featDF['expertsThreads_CPE_R' + str(k+1)] = commThreadsList['CPE_R' + str(k+1)]
            # featDF['CondExperts_CPE_R' + str(k+1)] = condExpertsList['CPE_R' + str(k+1)]
        except:
            featDF['shortestPaths_CPE_R' + str(k + 1)] = 0.0
            # featDF['communityCount_CPE_R' + str(k+1)] = 0.0
            # featDF['commuteTime_CPE_R' + str(k+1)] = 0.0
            # featDF['expertsThreads_CPE_R' + str(k + 1)] = 0.0
            # featDF['CondExperts_CPE_R' + str(k + 1)] = 0.0


    return featDF


def main():
    gc.collect()
    # posts = pickle.load(open('../../data/Dw_data/posts_days_forumsV1.0.pickle', 'rb'))

    start_date = datetime.datetime.strptime('09-01-2016', '%m-%d-%Y')
    end_date = datetime.datetime.strptime('05-01-2017', '%m-%d-%Y')

    # df_posts = countConversations(start_date, end_date, forums_cve_mentions)
    # pickle.dump(df_posts, open('../../data/DW_data/posts_days_forumsV2.0.pickle', 'wb'))

    # featDf = computeFeatureTimeSeries(start_date, end_date)

    numProcessors = 4
    pool = multiprocessing.Pool(numProcessors)

    currEndDate = start_date + relativedelta(months=1)
    print("Loading data...")

    tasks = []
    while (currEndDate <= end_date):
        # print(start_date, currEndDate)
        tasks.append((start_date, currEndDate))
        start_date += relativedelta(months=1)
        currEndDate = start_date + relativedelta(months=1)

    results = pool.starmap_async(computeFeatureTimeSeries_Graph, tasks)
    pool.close()
    pool.join()

    feat_data = results.get()
    df_list = []
    for df_idx in range(len(feat_data)):
        df = feat_data[df_idx]
        df_list.append(df)

    feat_data_all = pd.concat(df_list)
    # print(feat_data_all)
    # pickle.dump(feat_data_all, open('../../data/DW_data/conductance_DeltaT_4_Sept16-Apr17_TP10.pickle', 'wb'))
    # pickle.dump(feat_data_all, open('../../data/DW_data/commThreads_DeltaT_4_Sept16-Apr17_TP10.pickle', 'wb'))
    # pickle.dump(feat_data_all, open('../../data/DW_data/commuteTime_DeltaT_4_Sept16-Apr17_TP10.pickle', 'wb'))
    pickle.dump(feat_data_all, open('../../data/DW_data/communityCount_DeltaT_4_Sept16-Apr17_TP10.pickle', 'wb'))
    # feat_data_all = pickle.load(open('../../data/DW_data/feature_df_Sept16-Apr17.pickle', 'rb'))
    # feat_data_all = feat_data_all.reset_index(drop=True)
    # pickle.dump(feat_data_all, open('../../data/DW_data/user_interStats_DeltaT_4_Sept16-Apr17_TP10.pickle', 'wb'))


if __name__ == "__main__":
   main()