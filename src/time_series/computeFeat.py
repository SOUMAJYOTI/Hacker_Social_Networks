# import load_data_api as ldap
import pandas as pd
import networkx as nx
import datetime as dt
import operator
import pickle
from sqlalchemy import create_engine
# import userAnalysis as usAn
import numpy as np
import networkx.algorithms.cuts as nxCut
import datetime
import src.network_analysis.createConnections as ccon
import src.load_data.load_dataDW as ldDW
import matplotlib.pyplot as plt
from dateutil.relativedelta import relativedelta
import multiprocessing
import threading
import gc

# Global storage structures used over all pool processing
forums = [88, 248, 133, 49, 62, 161, 84, 60, 104, 173, 250, 105, 147, 40, 197, 220
        , 179, 219, 265, 98, 150, 121, 35, 214, 266, 89, 71, 146, 107, 64,
                           218, 135, 257, 243, 211, 236, 229, 259, 176, 159, 38]

# forums = [88, 40]
# vulnInfo = pickle.load(open('../../data/DW_data/09_15/Vulnerabilities-sample_v2+.pickle', 'rb'))
# cve_cpe_map = pd.read_csv('../../data/DW_data/cve_cpe_map.csv')
# Map of users with CVE to user and user to CVE
users_CVEMap, CVE_usersMap = pickle.load(open('../../data/DW_data/users_CVE_map.pickle', 'rb'))


allPosts = pickle.load(open('../../data/DW_data/dw_database_data_2016-17.pickle', 'rb'))
KB_edges_DS = pickle.load(open('../../data/DW_data/KB_edges_df.pickle', 'rb'))
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


def Conductance(network, userG1, userG2):
    conductanceVal = nxCut.conductance(network, userG1, userG2)

    return conductanceVal


def centralities(network, arg, users):
    cent = {}
    if arg == "InDegree":
        cent = nx.in_degree_centrality(network)

    if arg == "OutDegree":
        cent = nx.out_degree_centrality(network)

    if arg == "PageRank":
        cent = nx.pagerank(network)

    if arg == "Core":
        cent = nx.core_number(network)

    cent_sum = 0.
    for u in users:
        cent_sum += cent[u]

    return cent_sum / len(users)


def getExperts(users):
    '''

    :param users:
    :return:

    This has to be corrected - experts should have CVEs in the KB timeframe - not have CVEs in any time frame

    '''
    experts = []
    users_with_CVEs = list(users_CVEMap.keys())
    for u in users:
        if u in users_with_CVEs:
            experts.append(u)

    return experts


def computeFeatureTimeSeries(start_date, end_date,):
    # titlesList = []
    # feat_topUsers = []
    # feat_experts = []
    # newUsersWeeklyPercList = []

    condList = []
    condExpertsList = []
    prList = []
    degList = []

    datesList = []
    featDF = pd.DataFrame()

    # KB timeframe 3 months prior to start of daily training data
    KBStartDate = start_date - relativedelta(months=3)

    # while currEndDate < end_date:
    KB_edges = KB_edges_DS[KBStartDate]
    KB_users = list(set(KB_edges['source']).intersection(set(KB_edges['target'])))
    KB_users = [str(int(i)) for i in KB_users]
    experts = getExperts(KB_users)

    currStartDate = start_date
    currEndDate = start_date + datetime.timedelta(days=1)
    print( "Month: ", currStartDate.date())

    while currEndDate < end_date:
        # users_currDay = postsDailyDf[postsDailyDf['date'] == currStartDate]
        # Compute the feature for each forum separately

        # users_currDay_forum = users_currDay[users_currDay['forum'] == float(f)]

        '''' Consider forums posts for current forum for current day'''
        df_currDay = allPosts[allPosts['posteddate'] >= (currStartDate.date() - datetime.timedelta(days=2))]
        df_currDay = df_currDay[df_currDay['posteddate'] < currEndDate.date()]
        threadidsCurrDay = list(set(df_currDay['topicid']))
        currDay_edges = ccon.storeEdges(df_currDay, threadidsCurrDay)

        # If there are no edges, no feature computation
        if len(currDay_edges) == 0:
            condList.append(0)
            condExpertsList.append(0)
            prList.append(0)
            degList.append(0)
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
        datesList.append(currStartDate)

        currStartDate = currStartDate + datetime.timedelta(days=1)
        currEndDate = currStartDate + datetime.timedelta(days=1)

    featDF['date'] = datesList
    featDF['conductance'] = condList
    featDF['conductanceExperts'] = condExpertsList
    featDF['pagerank'] = prList
    featDF['degree'] = degList

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
            df_currDay = allPosts[allPosts['posteddate'] >= (currStartDate.date() - datetime.timedelta(days=2))]
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


def main():
    gc.collect()
    posts = pickle.load(open('../../data/Dw_data/posts_days_forumsV1.0.pickle', 'rb'))

    start_date = datetime.datetime.strptime('09-01-2016', '%m-%d-%Y')
    end_date = datetime.datetime.strptime('05-01-2017', '%m-%d-%Y')

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

    results = pool.starmap_async(computeFeatureTimeSeries, tasks)
    pool.close()
    pool.join()

    feat_data = results.get()
    df_list = []
    for df_idx in range(len(feat_data)):
        df = feat_data[df_idx]
        df_list.append(df)

    feat_data_all = pd.concat(df_list)
    pickle.dump(feat_data_all, open('../../data/DW_data/feature_df_combined_Sept16-Apr17.pickle', 'wb'))
    # feat_data_all = pickle.load(open('../../data/DW_data/feature_df_Sept16-Apr17.pickle', 'rb'))
    # feat_data_all = feat_data_all.reset_index(drop=True)
    # print(feat_data_all)

if __name__ == "__main__":
   main()