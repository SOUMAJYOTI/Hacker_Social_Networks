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


# Global storage structures used over all pool processing
forums = [ 88, 248, 133, 49, 62, 161, 84, 60, 104, 173, 250, 105, 147, 40, 197, 220
        , 179, 219, 265, 98, 150, 121, 35, 214, 266, 89, 71, 146, 107, 64,
                           218, 135, 257, 243, 211, 236, 229, 259, 176, 159, 38]
# vulnInfo = pickle.load(open('../../data/DW_data/09_15/Vulnerabilities-sample_v2+.pickle', 'rb'))
# cve_cpe_map = pd.read_csv('../../data/DW_data/cve_cpe_map.csv')
# pickle.dump((users_CVEMap, CVE_usersMap_filtered), open('../../data/DW_data/users_CVE_map.pickle', 'wb'))

allPosts = pickle.load(open('../../data/DW_data/dw_database_data_2016-17.pickle', 'rb'))
KB_edges_DS = pickle.load(open('../../data/DW_data/KB_edges_df.pickle', 'rb'))



def dateToString(date):
    yearNum = str(date.year)
    monthNum = str(date.month)
    dayNum =str(date.day)
    if len(monthNum)<2:
        monthNum = "0"+monthNum
    if len(dayNum)<2:
        dayNum= "0"+dayNum
    return yearNum+"-"+monthNum+"-"+dayNum


def getCVEinTopCPE_Groups(start_date, end_date, vulInfo, cveCPE, K):
    """

    :param start_date:
    :param end_date:
    :param vulInfo:
    :param cveCPE:
    :param K:
    :return: CVEs in top CPE groups
    """
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


def getExperts(topCVE, CVE_userMap):
    usersCVECount = {}
    for tc in topCVE:
        userCurrCVE = CVE_userMap[tc]
        if len(userCurrCVE) == 0:
            continue
        for u in list(set(userCurrCVE[0])):
            # if u in usersGlobal:
            if u not in usersCVECount:
                usersCVECount[u] = 0
            usersCVECount[u] += 1

    usersSorted = sorted(usersCVECount.items(), key=operator.itemgetter(1), reverse=True)
    mean_count = np.mean(np.array(list(usersCVECount.values())))
    threshold = 0

    usersCVECount_topCPE = {}
    for u, count in usersSorted:
        if count >= threshold:
            usersCVECount_topCPE[u] = count

    return usersCVECount_topCPE


def Conductance(network, userG1, userG2):
    conductanceVal = nxCut.conductance(network, userG1, userG2)

    return conductanceVal


def centralities(network, arg, users):
    cent = {}
    if arg == "InDegree":
        cent = nx.in_degree_centrality(network)

    if arg == "OutDegree":
        cent = nx.out_degree_centrality(network)

    if arg == "Pagerank":
        cent = nx.pagerank(network)

    if arg == "core":
        cent = nx.core_number(network)

    cent_sum = 0.
    for u in users:
        cent_sum += cent[u]

    return cent_sum / len(users)


def computeFeatureTimeSeries(start_date, end_date,):
    # titlesList = []
    # feat_topUsers = []
    # feat_experts = []
    # newUsersWeeklyPercList = []

    datesList = []
    forumsList = []
    featValList = []
    featDF = pd.DataFrame()

    # KB timeframe 3 months prior to start of daily training data
    KBStartDate = start_date - relativedelta(months=3)

    # Map of users with CVE to user and user to CVE
    # users_CVEMap, CVE_usersMap = user_CVE_groups(cve_cpeData, vulnData)

    # while currEndDate < end_date:
    KB_edges = KB_edges_DS[KBStartDate]
    KB_users = list(set(KB_edges['source']).intersection(set(KB_edges['target'])))

    currStartDate = start_date
    currEndDate = start_date + datetime.timedelta(days=1)

    for f in forums:
        while currEndDate < end_date:
            # users_currDay = postsDailyDf[postsDailyDf['date'] == currStartDate]
            # Compute the feature for each forum separately

            # users_currDay_forum = users_currDay[users_currDay['forum'] == float(f)]

            df_currDay = allPosts[allPosts['posteddate'] >= currStartDate.date()]
            df_currDay = df_currDay[df_currDay['posteddate'] < currEndDate.date()]
            threadidsCurrDay = list(set(df_currDay['topicid']))
            currDay_edges = ccon.storeEdges(df_currDay, threadidsCurrDay)

            # If there are no edges, no feature computation
            if len(currDay_edges) == 0:
                featValList.append(0)
                currStartDate = currStartDate + datetime.timedelta(days=1)
                currEndDate = currStartDate + datetime.timedelta(days=1)
                continue

            users_curr = list(set(currDay_edges['source']).intersection(set(currDay_edges['target'])))
            # If there are no users, no feature computation
            if len(users_curr) == 0:
                featValList.append(0)
                currStartDate = currStartDate + datetime.timedelta(days=1)
                currEndDate = currStartDate + datetime.timedelta(days=1)
                continue

            users_curr = [str(int(i)) for i in users_curr]

            # print("Merging edges...")
            mergeEgdes = ccon.network_merge(KB_edges.copy(), currDay_edges)  # Update the mergedEdges every day

            G = nx.DiGraph()
            G.add_edges_from(mergeEgdes)

            featValList.append(Conductance(G, KB_users, users_curr))
            forumsList.append(f)
            datesList.append(currStartDate)

            currStartDate = currStartDate + datetime.timedelta(days=1)
            currEndDate = currStartDate + datetime.timedelta(days=1)

    featDF['date'] = datesList
    featDF['forum'] = forumsList
    featDF['conductance'] = featValList

    return featDF

if __name__ == "__main__":

    posts = pickle.load(open('../../data/Dw_data/posts_days_forumsV1.0.pickle', 'rb'))

    start_date = datetime.datetime.strptime('04-01-2016', '%m-%d-%Y')
    end_date = datetime.datetime.strptime('05-01-2016', '%m-%d-%Y')

    # df_posts = countConversations(start_date, end_date, forums_cve_mentions)
    # pickle.dump(df_posts, open('../../data/DW_data/posts_days_forumsV2.0.pickle', 'wb'))

    featDf = computeFeatureTimeSeries(start_date, end_date)

