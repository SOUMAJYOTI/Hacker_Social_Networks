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
    deg_cve = []
    bw_cve = []

    datesList = []
    featDF = pd.DataFrame()

    users_CVEs = getUsersCVES() # Get the list of users with CVEs

    currStartDate = start_date
    currEndDate = start_date + datetime.timedelta(days=1)

    while currEndDate <= end_date:
        print("Date: ", currStartDate)
        '''' Consider forums posts for current day '''
        df_currDay = allPosts[allPosts['posteddate'] >= (currStartDate.date())]
        df_currDay = df_currDay[df_currDay['posteddate'] < currEndDate.date()]
        df_currDay = df_currDay[df_currDay['forumsid'].isin(forums)]

        threadidsCurrDay = list(set(df_currDay['topicid']))
        currDay_edges_df = ccon.storeEdges(df_currDay, threadidsCurrDay)


        ''' Create the network

         POINTS TO NOTE:

         (1) THE NETWORKS ARE DISCONNECTED ACROSS FORUMS
         (2) SO TOP-K METRICS MEANS THAT IT WOULD RETRIEVE TOP K ACROSS ALL DISCONNECTED FORUMS ON THAT DAY

         THE IDEAL CASE - HAVE ALL FORUMS GRAPH WITH GLOBAL USERS IDS AND THEN COMPUTE TOP-K

         **** WOULD NOT RECOMMEND: TOP-K FROM ALL FORUMS SEPARATELY AND THEN AVERAGE ******

         k can be set to 50 users
         '''
        k = 50

        G = nx.DiGraph()
        G.add_edges_from(ccon.createNetwork(currDay_edges_df))

        # If there are no edges, no feature computation
        if len(currDay_edges_df) == 0:
            pr.append(0)
            deg.append(0)
            bw.append(0)

            datesList.append(currStartDate)
            currStartDate = currStartDate + datetime.timedelta(days=1)
            currEndDate = currStartDate + datetime.timedelta(days=1)
            continue

        users_curr = list(set(currDay_edges_df['source']).union(set(currDay_edges_df['target'])))

        # If there are no users, no feature computation
        if len(users_curr) == 0:
            pr.append(0)
            deg.append(0)
            bw.append(0)

            datesList.append(currStartDate)
            currStartDate = currStartDate + datetime.timedelta(days=1)
            currEndDate = currStartDate + datetime.timedelta(days=1)
            continue


        ''' Store the computed features '''
        pr.append(top_k_feat(G, 'PageRank', k))
        bw.append(top_k_feat(G, 'Betweenness', k))
        deg.append(top_k_feat(G, 'OutDegree', k))

        pr_cve.append(top_k_cve_feat(G, 'PageRank', k, users_CVEs))
        bw_cve.append(top_k_cve_feat(G, 'Betweenness', k, users_CVEs))
        deg_cve.append(top_k_cve_feat(G, 'OutDegree', k, users_CVEs))


        datesList.append(currStartDate)
        currStartDate = currStartDate + datetime.timedelta(days=1)
        currEndDate = currStartDate + datetime.timedelta(days=1)

    featDF['date'] = datesList
    featDF['pagerank'] = pr
    featDF['betweenness'] = bw
    featDF['outdegree'] = deg
    featDF['pagerank_cve'] = pr_cve
    featDF['betweenness_cve'] = bw_cve
    featDF['outdegree_cve'] = deg_cve

    return featDF


def main():
    gc.collect()

    start_date = datetime.datetime.strptime('03-01-2016', '%m-%d-%Y')
    end_date = datetime.datetime.strptime('09-01-2017', '%m-%d-%Y')


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

    pickle.dump(feat_data_all, open('../../data/DW_data/SNA_Mar16-Apr17_TP50.pickle', 'wb'))


if __name__ == "__main__":
   main()