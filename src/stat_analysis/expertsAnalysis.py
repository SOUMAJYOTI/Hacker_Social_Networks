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

# data = pd.read_pickle('../../data/DW_data/new_DW/Vulnerabilities_Armstrong.pickle')
# print(data[:20])
# exit()

# Global storage structures used over all pool processing
# Top forums by count in the Armstrong timeframe > 1000 posts
forums = [129, 6, 112, 77, 69, 178, 31, 134, 193, 56, 201, 250, 13,
          205, 194, 110, 121, 233, 23, 232, 44, 29, 97, 204, 82, 155,
          48, 93, 45, 126, 174, 117, 41, 248, 177, 135, 22, 172, 189,
          14, 137, 231, 91, 55, 192, 245, 234, 199, 7, 184, 43, 183, 57]

forums_features = [129, 6, 112, 77, 69, 178, 31, 134, 193, 56, 201, 250, 13, 205, 194, 110, 121]

# forums = [88, ]
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


def expertDegDistr(start_date):
    KBStartDate = start_date

    KB_edges = KB_edges_DS[KBStartDate]
    KB_users = list(set(KB_edges['source']).intersection(set(KB_edges['target'])))
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

    experts = getExperts(KB_users, G_KB, 10) # deg_thresh = 1, means no restriction on in-degree

    degList = getDegDist(G_KB, experts)
    # degList = getDegDist(G_KB, KB_users)

    return degList


def computeCentralities(start_date):
    KBStartDate = start_date
    KB_edges = KB_edges_DS[KBStartDate]
    KB_users = list(set(KB_edges['source']).intersection(set(KB_edges['target'])))
    KB_users = [str(i) for i in KB_users]

    # Store the KB pairs - edges
    KB_pairs = []
    for idx_KB, row_KB in KB_edges.iterrows():
        src = str(row_KB['source'])
        tgt = str(row_KB['target'])
        # if (src, tgt) not in KB_pairs:
        KB_pairs.append((src, tgt))

    G_KB = nx.DiGraph()
    G_KB.add_edges_from(list(set(KB_pairs)))

    experts = getExperts(KB_users, G_KB, 1)  # deg_thresh = 1, means no restriction on in-degree

    print('hello')

    degList_exp = getDegDist(G_KB, experts)
    print('hello')
    degList_nonExp = getDegDist(G_KB, list(set(KB_users).difference(set(experts))))

    return degList_exp, degList_nonExp


def computeUsersStats(start_date, end_date):
    print('Start date: ', start_date)

    return computeCentralities(start_date)

def main():
    gc.collect()

    # start_date = datetime.datetime.strptime('03-01-2016', '%m-%d-%Y')
    # end_date = datetime.datetime.strptime('09-10-2017', '%m-%d-%Y')
    #
    # numProcessors = 5
    # pool = multiprocessing.Pool(numProcessors)
    #
    # currEndDate = start_date + relativedelta(months=3)
    # print("Loading data...")
    #
    # tasks = []
    # while (currEndDate <= end_date):
    #     # print(start_date, currEndDate)
    #     tasks.append((start_date, currEndDate))
    #     start_date += relativedelta(months=3)
    #     currEndDate = start_date + relativedelta(months=3)
    #
    # results = pool.starmap_async(computeUsersStats, tasks)
    # pool.close()
    # pool.join()
    #
    # feat_data = results.get()
    # feat_list_exp = []
    # feat_list_nonexp = []
    # for f_idx in range(len(feat_data)):
    #     feat_exp, feat_nonexp = feat_data[f_idx]
    #     feat_list_exp.extend(feat_exp)
    #     feat_list_nonexp.extend(feat_nonexp)
    #
    # pickle.dump([feat_list_exp, feat_list_nonexp], open('../../data/DW_data/stats/experts_outDegree.pickle', 'wb'))
    [feat_list_exp, feat_list_nonexp] = pickle.load(open('../../data/DW_data/stats/experts_outDegree.pickle', 'rb'))
    fig = plt.figure(1, figsize=(10, 8))

    # Create an axes instance
    ax = fig.add_subplot(111)

    bp = ax.boxplot([feat_list_exp, feat_list_nonexp], patch_artist=True)

    for box in bp['boxes']:
        # change outline color
        box.set(color='#000000', linewidth=2)
        # change fill color
        box.set(facecolor='#D3D3D3')

        ## change color and linewidth of the whiskers
        # for whisker in bp['whiskers']:
        #     whisker.set(color='#7570b3', linewidth=2)

        ## change color and linewidth of the caps
        # for cap in bp['caps']:
        #     cap.set(color='#7570b3', linewidth=2)

        ## change color and linewidth of the medians
    for median in bp['medians']:
        median.set(color='#000000', linewidth=4)

        ## change the style of fliers and their fill
    for flier in bp['fliers']:
        flier.set(marker='o', color='#e7298a', alpha=0.5)

    ax.set_ylim([0, 30])

    third_quartile = [item.get_ydata()[0] for item in bp['whiskers']]
    third_quartile = max(third_quartile)


    # set axes limits and labels
    # xlim(0, pos_index)
    # ylim(0, pos_index)

    # ax.set_xticklabels(x_labels)
    # ax.set_xticks(x_tickPos)

    plt.tick_params('y', labelsize=20)
    plt.tick_params('x', labelsize=20)
    # ax.set_yticks(fontsize=15)
    # plt.title(title, size=25)

    plt.grid(True)
    plt.xticks([1, 2], ['Experts', 'Non Experts'], size=30)
    # plt.ylim([0,30])
    # plt.xlabel('Number of days prior to date of event', fontsize=30)
    plt.ylabel('Out Degree', fontsize=30)
    plt.show()
    # plt.savefig(plot_dir + feat + '_' + title + '.png')
    # plt.close()

if __name__ == "__main__":
   main()