import pandas as pd
from dateutil.relativedelta import relativedelta
import pickle
import datetime
import src.network_analysis.createConnections as ccon
import src.load_data.load_dataDW as ldDW
import operator
import re
import csv
import numpy as np
import sys
maxInt = sys.maxsize
decrement = True

while decrement:
    # decrease the maxInt value by factor 10
    # as long as the OverflowError occurs.

    decrement = False
    try:
        csv.field_size_limit(maxInt)
    except OverflowError:
        maxInt = int(maxInt/10)
        decrement = True


def dateToString(date):
    yearNum = str(date.year)
    monthNum = str(date.month)
    dayNum = str(date.day)
    if len(monthNum) < 2:
        monthNum = "0" + monthNum
    if len(dayNum) < 2:
        dayNum = "0" + dayNum
    return yearNum + "-" + monthNum + "-" + dayNum


def user_CVE_groups(vul_data):
    users_CVEMap = {}
    CVE_usersMap = {}
    for idx, row in vul_data.iterrows():
        if row['vulnId'] not in CVE_usersMap:
            CVE_usersMap[row['vulnId']] = []
        for uid in row['users']:
            if uid not in users_CVEMap:
                users_CVEMap[uid] = []

            if type(uid) is float and np.isnan(uid):
                continue
            # print(uid, row['vulnId'])
            users_CVEMap[uid].append(row['vulnId'])
            CVE_usersMap[row['vulnId']].append(uid)

    return users_CVEMap, CVE_usersMap


def countConversations(start_date, end_date, forums):
    df_postsTS = pd.DataFrame()

    datesList = []
    forumsList = []
    numPostsList = []
    uidsList = []
    uidsCount = []

    for f in forums:
        currStartDate = start_date
        currEndDate = start_date + datetime.timedelta(days=1)
        print("Forum: ", f)
        postsDf = ldDW.getDW_data_postgres(forums_list=[f], start_date=start_date.strftime("%Y-%m-%d"),
                                           end_date=end_date.strftime("%Y-%m-%d"))
        postsDf['posteddate'] = pd.to_datetime(postsDf['posteddate'])
        while currEndDate < end_date:
            # print("Start Date:", currStartDate )
            usersTempList = []

            posts_currDay = postsDf[postsDf['posteddate'] >= currStartDate]
            posts_currDay = posts_currDay[posts_currDay['posteddate'] < currEndDate]

            forumsList.append(f)
            datesList.append(currStartDate)
            numPostsList.append(len(posts_currDay))

            for idx, row in posts_currDay.iterrows():
                usersTempList.append(row['uid'])

            uidsList.append(usersTempList)
            uidsCount.append(len(list(set(usersTempList))))
            currStartDate = currStartDate + datetime.timedelta(days=1)
            currEndDate = currEndDate + datetime.timedelta(days=1)

    df_postsTS['forum'] = forumsList
    df_postsTS['date'] = datesList
    df_postsTS['number_posts'] = numPostsList
    df_postsTS['users'] = uidsList
    df_postsTS['number_users'] = uidsCount

    return df_postsTS


def topCPEGroups(start_date, end_date, vulInfo, cve_cpe_df, K):
    currStartDate = start_date
    currEndDate = start_date + relativedelta(months=3)

    while currStartDate < end_date:
        print(currStartDate, currEndDate)
        vulCurr = vulInfo[vulInfo['postedDate'] >= currStartDate.date()]
        vulCurr = vulCurr[vulCurr['postedDate'] < currEndDate.date()]

        vulnerab = vulCurr['vulnId']
        cve_cpe_curr = cve_cpe_df[cve_cpe_df['cve'].isin(vulnerab)]
        topCPEs = {}
        for idx, row in cve_cpe_curr.iterrows():
            clTag = row['cluster']
            if clTag not in topCPEs:
                topCPEs[clTag] = 0

            topCPEs[clTag] += 1

        # topCPEs_sorted = sorted(topCPEs.items(), key=operator.itemgetter(1), reverse=True)
        if K==-1:
            K = len(topCPEs)
        topCPEs_sorted = sorted(topCPEs.items(), key=operator.itemgetter(1), reverse=True)[:K]
        topCPEsList = []
        for cpe, count in topCPEs_sorted:
            topCPEsList.append(cpe)

        topCVE = cveCPE[cveCPE['cluster_tag'].isin(topCPEsList)]
        currStartDate += relativedelta(months=1)
        currEndDate = currStartDate + relativedelta(months=3)
    #
    # return list(topCVE['cve'])


def computeKBnetwork(start_date, end_date, allPosts):
    currStartDate = start_date
    currEndDate = start_date + relativedelta(months=3)

    KB_edges_Data = {}
    while currStartDate < end_date:
        print(currStartDate, currEndDate)
        df_KB = allPosts[allPosts['posteddate'] >= currStartDate.date()]
        df_KB = df_KB[df_KB['posteddate'] < currEndDate.date()]
        threadidsKB = list(set(df_KB['topicid']))
        KB_edges = ccon.storeEdges(df_KB, threadidsKB)
        KB_edges_Data[currStartDate] = KB_edges

        print(len(df_KB), len(KB_edges))
        currStartDate += relativedelta(months=1)
        currEndDate = currStartDate + relativedelta(months=3)

    return KB_edges_Data


def hasNumbers(inputString):
    return re.findall(r'\d', inputString)

def hasVersion(inputString):
    return re.findall(r'\.', inputString)

def hasLetters(inputString):
    return bool(re.search('[a-zA-Z]', inputString))
    # return any(c.isalpha() for c in inputString)

def store_cve_cpe_map(cve_cpe_df):
    cve_cpe_map = {}
    clusters_id = []
    for idx, row in cve_cpe_df.iterrows():
        ''' Filter the clusters '''
        cluster_tags = str(row['cluster']).split(' | ')
        cluster_custom = []
        cluster_custom.append(cluster_tags[0])
        for idx_cl in range(1, len(cluster_tags)):
            ctag = cluster_tags[idx_cl]
            if ctag in cluster_custom:
                continue

            ver = hasVersion(ctag)
            if len(ver) >= 2:
                continue

            cluster_custom.append(ctag)

        cluster_final = ''
        for idx_cc in range(len(cluster_custom)):
            cluster_final += cluster_custom[idx_cc] + ' '

        cluster_final = " ".join(cluster_final.split())
        if cluster_final not in clusters_id:
            clusters_id.append(cluster_final)

        if row['cve'] not in cve_cpe_map:
            cve_cpe_map[row['cve']] = []

        cve_cpe_map[row['cve']].append(cluster_final)

    # print(len(cve_cpe_df), len(list(set(list(cve_cpe_df['cluster'])))))
    # exit()
    return cve_cpe_map


def processVulnInfo(vuln_df, start_date, end_date):
    '''
    This function processes and filters relevant columns from the SDK/API information
    obtained from getdetailedVulnInfo()

    The start_date and end_date is to filter the vulnerabilities within the Armstrong event range
    :param vuln_df:
    :return:
    '''

    postedDateList = []
    vulnIdList = []
    indicatorList = []
    marketIdList = []
    forumIdList = []
    numForumsList = []
    itemNameList = []
    numUsersList = []
    usersList = []

    for cve, g in vuln_df.groupby(by='vulnerabilityid'):
        postedDates = []
        forums = []
        indicators = []
        marketiDs = []
        itemNames = []
        users = []
        for idx, row in g.iterrows():
            pDate = pd.to_datetime(row['posteddate'])

            if pDate >= start_date and pDate < end_date:
                postedDates.append(row['posteddate'])
                forums.append(row['forumsid'])
                indicators.append(row['indicator'])
                marketiDs.append(row['marketplaceid'])
                itemNames.append(row['itemdescription'])
                users.append(row['uid'])

        if len(postedDates) == 0: # empty df
            continue
        vulnIdList.append(cve)
        postedDateList.append(postedDates)
        indicatorList.append(indicators)
        marketIdList.append(marketiDs)
        forumIdList.append(forums)
        numForumsList.append(len(list(set(forums))))
        itemNameList.append(itemNames)
        numUsersList.append(len(list(set(users))))
        usersList.append(users)

    vuln_df_filtered = pd.DataFrame()
    vuln_df_filtered['postedDate'] = postedDateList
    vuln_df_filtered['vulnId'] = vulnIdList
    vuln_df_filtered['indicator'] = indicatorList
    vuln_df_filtered['marketId'] = marketIdList
    vuln_df_filtered['forumID'] = forumIdList
    vuln_df_filtered['numForums'] = numForumsList
    vuln_df_filtered['itemName'] = itemNameList
    vuln_df_filtered['users'] = usersList
    vuln_df_filtered['numUsers'] = numUsersList

    return vuln_df_filtered


if __name__ == "__main__":
    # forums_cve_mentions = [88, 248, 133, 49, 62, 161, 84, 60, 104, 173, 250, 105, 147, 40, 197, 220
    #     , 179, 219, 265, 98, 150, 121, 35, 214, 266, 89, 71, 146, 107, 64,
    #                        218, 135, 257, 243, 211, 236, 229, 259, 176, 159, 38]
    # posts = pd.read_pickle('../../data/Dw_data/posts_days_forumsV1.0.pickle')

    start_date = datetime.datetime.strptime('01-01-2016', '%m-%d-%Y')
    end_date = datetime.datetime.strptime('10-01-2017', '%m-%d-%Y')

    # df_posts = countConversations(start_date, end_date, forums_cve_mentions)
    # pickle.dump(df_posts, open('../../data/DW_data/posts_days_forumsV2.0.pickle', 'wb'))
    # vuln_old = pd.read_pickle('../../data/DW_data/CPE_events_corr.pickle')
    # print(vuln_old[:100])
    # exit()
    vuln_df = pd.read_csv('../../data/DW_data/VulnInfo_11_17.csv', encoding='ISO-8859-1', engine='python')

    # print(vulnInfo[:10])
    vuln_df_filter = processVulnInfo(vuln_df, start_date, end_date)
    print(vuln_df_filter)
    pickle.dump(vuln_df_filter, open('../../data/DW_data/Vulnerabilities_Armstrong.pickle', 'wb'))
    # exit()
    #
    # cve_cpe_df =  pd.read_csv('../../data/DW_data/new_DW/cve_cpe_mapDF_new.csv')
    # cve_cpe_map = store_cve_cpe_map(cve_cpe_df)
    #
    # pickle.dump(cve_cpe_map, open('../../data/DW_data/new_DW/cve_cpe_map_new.pickle', 'wb') )

    # topCPEGroups(start_date, end_date, vulnInfo, cve_cpe_df, -1)
    # cve_cpe_map = pd.read_csv('../../data/DW_data/cve_cpe_mapDF.csv')
    # print(cve_cpe_map[:10])
    # vulnInfo = pd.read_pickle('../../data/DW_data/Vulnerabilities_Armstrong.pickle')
    # print(vulnInfo[:10])
    # users_CVEMap, CVE_usersMap = user_CVE_groups(vulnInfo)
    #
    # CVE_usersMap_filtered = {}
    # for cve in CVE_usersMap:
    #     if CVE_usersMap[cve] == 0:
    #         continue
    #     CVE_usersMap_filtered[cve] = CVE_usersMap[cve]
    #
    # # print(CVE_usersMap_filtered)
    # pickle.dump((users_CVEMap, CVE_usersMap), open('../../data/DW_data/new_DW/users_CVE_map_new.pickle', 'wb'))

    # df_posts = pd.read_pickle('../../data/DW_data/new_DW/dw_database_dataframe_2016-17_new.pickle')
    # print(df_posts[:10])
    # KB_edgesDf = computeKBnetwork(start_date, end_date, df_posts)
    # pickle.dump(KB_edgesDf, open('../../data/DW_data/new_DW/KB_edges_df_new.pickle', 'wb'))

    # print(KB_edgesDf)

