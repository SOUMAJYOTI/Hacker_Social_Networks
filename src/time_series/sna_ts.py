import pandas as pd
import networkx as nx
import operator
import pickle
# import userAnalysis as usAn
import numpy as np
import datetime
import src.network_analysis.createConnections as ccon
import matplotlib.pyplot as plt
from dateutil.relativedelta import relativedelta
import multiprocessing
import gc
import re
from src.network_analysis.features import *
import time

# data = pd.read_pickle('../../data/DW_data/new_DW/Vulnerabilities_Armstrong.pickle')
# print(data[:20])
# exit()

# Global storage structures used over all pool processing
# Top forums by count in the Armstrong timeframe > 1000 posts
forums = [129, 6, 112, 77, 69, 178, 31, 134, 193, 56, 201, 250, 13,
          205, 194, 110, 121, 233, 23, 232, 44, 29, 97, 204, 82, 155,
          48, 93, 45, 126, 174, 117, 41, 248, 177, 135, 22, 172, 189,
          14, 137, 231, 91, 55, 192, 245, 234, 199, 7, 184, 43, 183, 57]

forums_features = [41, 129, 6, 112, 77, 69, 178, 31, 134, 193, 56, 201, 250, 13, 205, 194, 110, 121]

# forums = [88, ]
vulnInfo = pd.read_pickle('../../data/DW_data/new_DW/Vulnerabilities_Armstrong.pickle')
cve_cpe_DF = pd.read_csv('../../data/DW_data/new_DW/cve_cpe_mapDF_new.csv')
cve_cpe_map = pickle.load(open('../../data/DW_data/new_DW/cve_cpe_map_new.pickle', 'rb'))

# Map of users with CVE to user and user to CVE
users_CVEMap, CVE_usersMap = pickle.load(open('../../data/DW_data/new_DW/users_CVE_map_new.pickle', 'rb'))


allPosts = pd.read_pickle('../../data/DW_data/new_DW/dw_database_dataframe_2016-17_new.pickle')
KB_edges_DS = pd.read_pickle('../../data/DW_data/new_DW/KB_edges_df_new.pickle')

# start_date = datetime.datetime.strptime('08-01-2016', '%m-%d-%Y')
# KB_edges_DS[start_date].to_csv('../../data/DW_data/new_DW/KB_edges_df_new.csv')
#
# exit()

allPosts['forumsid'] = allPosts['forumsid'].astype(int)
allPosts['topicid'] = allPosts['topicid'].astype(str)
allPosts['postsid'] = allPosts['postsid'].astype(str)
allPosts['uid'] = allPosts['uid'].astype(str)


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



def getUsersCVES():
    '''

    :param users:
    :return:

    return users with CVEs
    '''
    users_with_CVEs = list(users_CVEMap.keys())

    return users_with_CVEs


def computeFeatureTimeSeries_SNA(start_date, end_date,):

    '''

    Compute the features daily for sna on reply network - centralities

    :param start_date:
    :param end_date:
    :return:
    '''

    pr = []
    deg = []
    bw = []

    pr_cve = []
    eg_cve = []
    bw_cve = []

    datesList = []
    featDF = pd.DataFrame()

    users_CVEs = getUsersCVES()

    currStartDate = start_date
    currEndDate = start_date + datetime.timedelta(days=1)

    while currEndDate <= end_date:
        print("Date: ", currStartDate)
        '''' Consider forums posts for current day '''
        df_currDay = allPosts[allPosts['posteddate'] >= (currStartDate.date())]
        df_currDay = df_currDay[df_currDay['posteddate'] < currEndDate.date()]
        df_currDay = df_currDay[df_currDay['forumsid'].isin(forums)]

        ''' Create the network

         POINTS TO NOTE:

         (1) THE NETWORKS ARE DISCONNECTED ACROSS FORUMS
         (2) SO TOP-K METRICS MEANS THAT IT WOULD RETRIEVE TOP K ACROSS ALL DISCONNECTED FORUMS ON THAT DAY

         THE IDEAL CASE - HAVE ALL FORUMS GRAPH WITH GLOBAL USERS IDS AND THEN COMPUTE TOP-K

         **** WOULD NOT RECOMMEND: TOP-K FROM ALL FORUMS SEPARATELY AND THEN AVERAGE ******

         k can be set to 50 users
         '''
        k = 50

        currDay_edges = ccon.createNetwork(df_currDay)
        G = nx.DiGraph()
        G.add_edges_from(currDay_edges)

        # If there are no edges, no feature computation
        if len(currDay_edges) == 0:
            pr.append(0)
            deg.append(0)
            bw.append(0)

            datesList.append(currStartDate)
            currStartDate = currStartDate + datetime.timedelta(days=1)
            currEndDate = currStartDate + datetime.timedelta(days=1)
            continue

        users_curr = list(set(currDay_edges['source']).union(set(currDay_edges['target'])))

        # If there are no users, no feature computation
        if len(users_curr) == 0:
            pr.append(0)
            deg.append(0)
            bw.append(0)

            datesList.append(currStartDate)
            currStartDate = currStartDate + datetime.timedelta(days=1)
            currEndDate = currStartDate + datetime.timedelta(days=1)
            continue

        # users_curr = [str(i) for i in users_curr]

        ''' Store the computed features '''


        pr.append(top_k_feat(G, 'PageRank', k))
        print(pr)

        # condExpertsList.append(Conductance(G, experts, users_curr))
        # commThreadsList.append(threadCommon(df_currDay, experts))
        # shortPathList.append(shortestPaths(G, experts, users_curr, weighted=weighted))
        # print(shortPathList)
        # communityCountList.append(approximate_community_detect(G, comm_partition, comm_experts, KB_users, experts, users_curr ))
        # commutePathList.append(commuteTime(pseudo_lapl_mat, nodeIndexMap, experts, users_curr))
        # print(commutePathList)
        # print(commuteTime(G, expertUsersCPE, users_curr))

        datesList.append(currStartDate)
        currStartDate = currStartDate + datetime.timedelta(days=1)
        currEndDate = currStartDate + datetime.timedelta(days=1)

    featDF['date'] = datesList
    # featDF['shortestPaths'] = shortPathList
    # featDF['commuteTime'] = commutePathList
    # featDF['communityCount'] = communityCountList
    # featDF['expertsThreads'] = commThreadsList
    # featDF['CondExperts'] = condExpertsList

    return featDF


def main():
    gc.collect()
    # posts = pickle.load(open('../../data/Dw_data/posts_days_forumsV1.0.pickle', 'rb'))

    start_date = datetime.datetime.strptime('03-01-2016', '%m-%d-%Y')
    end_date = datetime.datetime.strptime('09-01-2017', '%m-%d-%Y')

    # df_posts = countConversations(start_date, end_date, forums_cve_mentions)
    # pickle.dump(df_posts, open('../../data/DW_data/posts_days_forumsV2.0.pickle', 'wb'))

    # featDf = computeFeatureTimeSeries(start_date, end_date)

    numProcessors = 5
    pool = multiprocessing.Pool(numProcessors)

    currEndDate = start_date + relativedelta(months=1)
    print("Loading data...")

    tasks = []
    while (currEndDate <= end_date):
        # print(start_date, currEndDate)
        tasks.append((start_date, currEndDate))
        start_date += relativedelta(months=1)
        currEndDate = start_date + relativedelta(months=1)

    results = pool.starmap_async(computeFeatureTimeSeries_SNA, tasks)
    pool.close()
    pool.join()

    feat_data = results.get()
    df_list = []
    for df_idx in range(len(feat_data)):
        df = feat_data[df_idx]
        df_list.append(df)

    feat_data_all = pd.concat(df_list)

    # print(feat_data_all)
    # pickle.dump(df_list, open('../../data/DW_data/features/usersDegDistribution.pickle', 'wb'))
    # pickle.dump(feat_data_all, open('../../data/DW_data/features/feat_combine/weightedShortestPaths_Delta_T0_Mar16-Apr17.pickle', 'wb'))
    pickle.dump(feat_data_all, open('../../data/DW_data/SNA_Mar16-Apr17_TP50.pickle', 'wb'))



if __name__ == "__main__":
   main()