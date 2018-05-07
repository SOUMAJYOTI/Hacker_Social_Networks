import numpy as np
import pandas as pd
import os
from src.network_analysis.features import *
import src.network_analysis.createConnections as ccon
import pickle
from dateutil.relativedelta import relativedelta
from datetime import *
import datetime

'''
The purpose of this file is to check whether the feature values are outliers between the different
nodes considering the discussions in  the Darkweb for a vulnerability cve.

'''

forums = [129, 6, 112, 77, 69, 178, 31, 134, 193, 56, 201, 250, 13,
          205, 194, 110, 121, 233, 23, 232, 44, 29, 97, 204, 82, 155,
          48, 93, 45, 126, 174, 117, 41, 248, 177, 135, 22, 172, 189,
          14, 137, 231, 91, 55, 192, 245, 234, 199, 7, 184, 43, 183, 57]

forums_features = [129, 6, 112, 77, 69, 178, 31, 134, 193, 56, 201, 250, 13, 205, 194, 110, 121]
vulnInfo = pd.read_pickle('../../data/DW_data/new_DW/Vulnerabilities_Armstrong.pickle')
cve_cpe_DF = pd.read_csv('../../data/DW_data/new_DW/cve_cpe_mapDF_new.csv')
cve_cpe_map = pickle.load(open('../../data/DW_data/new_DW/cve_cpe_map_new.pickle', 'rb'))

# Map of users with CVE to user and user to CVE
users_CVEMap, CVE_usersMap = pickle.load(open('../../data/DW_data/new_DW/users_CVE_map_new.pickle', 'rb'))


allPosts = pd.read_pickle('../../data/DW_data/new_DW/dw_database_dataframe_2016-17_new.pickle')
KB_edges_DS = pd.read_pickle('../../data/DW_data/new_DW/KB_edges_df_new.pickle')


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


def getExperts(users, G_KB, deg_thresh):
    '''

    :param users:
    :return:

    This has to be corrected - experts should have CVEs in/history of the KB timeframe - not have CVEs in any time frame
    Conditions for experts:

    (1) Experts should have posted CVEs in the KB time frame
    (2) Experts should have posted CVEs in top CPEs for the KB time frame
    (3) The in-degree of experts in the KB networks should cross a threshold ---> lot of
        users have potentially replied to the experts

    For now, the conditions (1) and (3) are implemented
    '''
    experts = []
    users_with_CVEs = list(users_CVEMap.keys())
    for u in users:
        if u in users_with_CVEs:
            experts.append(u)

    experts_filter = []
    for exp in experts:
        if G_KB.in_degree(exp) >= deg_thresh:
            experts_filter.append(exp)

    return experts_filter


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


def findExperts_inKB(KBStartDate):
    # ----- Temporary modification to adjust the training data availability ----!!!!
    if KBStartDate < datetime.datetime.strptime('01-01-2016', '%m-%d-%Y'):
        KBStartDate = datetime.datetime.strptime('01-01-2016', '%m-%d-%Y')

    KB_edges = KB_edges_DS[KBStartDate]
    KB_edges = KB_edges[KB_edges['Forum'].isin(forums)]  ##### Forums Filter
    KB_users = list(set(KB_edges['source']).union(set(KB_edges['target'])))
    KB_users = [str(i) for i in KB_users]

    # print("hello")
    # Store the KB pairs - edges
    KB_pairs = []
    for idx_KB, row_KB in KB_edges.iterrows():
        src = str(row_KB['source'])
        tgt = str(row_KB['target'])
        # if (src, tgt) not in KB_pairs:
        KB_pairs.append((src, tgt))

    G_KB = nx.DiGraph()
    G_KB.add_edges_from(list(set(KB_pairs)))

    deg_thresh = 10  # based on empirical distribution
    experts = getExperts(KB_users, G_KB, deg_thresh)

    return experts


def computeFeat(start_date,):
    '''
    One example: find the paths between the post of the vulnerability mention
    and the experts - if the post user is an expert himself, then better

    For example, graph conductance: Try to show it is above the median of the graph conductance

    The first task is to see the shortest paths between the experts and the vulnerability mention users

    @param start_date:
    :return:
    '''

    start_month = start_date - timedelta(days=(start_date.days-1))
    datesList = []

    # KB timeframe 3 months prior to start of daily training data for the current month
    KBStartDate = start_month - relativedelta(months=3)

    experts = findExperts_inKB(start_date)

    print("Date ", start_date.date(), )
    # print("Number of users: ", len(KB_users), ", Number of experts: ", len(experts))

    # expertsLastTime = computeExpertsTime(KB_edges, experts)

    currStartDate = start_date
    currEndDate = start_date + datetime.timedelta(days=7)
    # print(" Month: ", currStartDate.date())


def selectVulnWeeks(df):
    '''

    :param df:
    :return:
    '''

    ''' Sort the df by posted date '''
    df = df.sort_values(by=['posteddate'])

    ''' Get the users who have posted in this cve '''
    uids = df['uid'].tolist()

    ''' Get the experts from the KB for this month '''
    start_date = df.iloc[0]['posteddate']
    start_date = datetime.datetime.strptime(start_date, '%Y-%m-%d')

    start_month = start_date - timedelta(days=(start_date.day-1))
    start_month = start_month.date()

    KBStartDate = start_month - relativedelta(months=3)

    KBStartDate = datetime.datetime(KBStartDate.year, KBStartDate.month, KBStartDate.day)

    experts = findExperts_inKB(KBStartDate)

    # return list(set(experts).intersection(set(uids)))

    return df.iloc[0]['uid']


def selectPostsUsers(uid, start_time, end_time):
    '''

    Find posts by specific users in a timeframe
    :param uid:
    :return:
    '''

    user_posts = allPosts[allPosts['uid'] == uid]

    print(user_posts)


def load_data(df):
    '''

    :param df:
    :return:
    '''


if __name__ == "__main__":
    feat_df = pd.read_pickle('../../data/DW_data/features/feat_combine/features_Delta_T0_Mar16-Aug17.pickle')
    vuln_df = pd.read_csv('../../data/DW_data/cve-2016-4117.csv')


    load_data(feat_df)

    user = selectVulnWeeks(vuln_df)

    selectPostsUsers(user, '', '')

    feat_df = pd.read_pickle('../../data/DW_data/features/feat_forums/features_Delta_T0_Mar16-Aug17.pickle')


