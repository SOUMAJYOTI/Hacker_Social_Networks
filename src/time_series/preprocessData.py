import pandas as pd
from dateutil.relativedelta import relativedelta
import pickle
import datetime
import src.network_analysis.createConnections as ccon
import src.load_data.load_dataDW as ldDW
import operator

def dateToString(date):
    yearNum = str(date.year)
    monthNum = str(date.month)
    dayNum = str(date.day)
    if len(monthNum) < 2:
        monthNum = "0" + monthNum
    if len(dayNum) < 2:
        dayNum = "0" + dayNum
    return yearNum + "-" + monthNum + "-" + dayNum


def user_CVE_groups(cve_cpe_data, vul_data):
    users_CVEMap = {}
    CVE_usersMap = {}
    for cve in cve_cpe_data['cve']:
        vulnItems = vul_data[vul_data['vulnId'] == cve]
        users = vulnItems['users'].tolist()
        if cve not in CVE_usersMap:
            CVE_usersMap[cve] = []

        if len(users) == 0:
            continue

        CVE_usersMap[cve].extend(users[0])
        usersList = users[0]  ##
        for u in usersList:
            # print(u)
            if u not in users_CVEMap:
                users_CVEMap[u] = []

            users_CVEMap[u].append(cve)

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

def store_cve_cpe_map(cve_cpe_df):
    cve_cpe_map = {}
    for idx, row in cve_cpe_df.iterrows():
        if row['cve'] not in cve_cpe_map:
            cve_cpe_map[row['cve']] = []

        cve_cpe_map[row['cve']].append(row['cluster'])
    return cve_cpe_map

if __name__ == "__main__":
    forums_cve_mentions = [88, 248, 133, 49, 62, 161, 84, 60, 104, 173, 250, 105, 147, 40, 197, 220
        , 179, 219, 265, 98, 150, 121, 35, 214, 266, 89, 71, 146, 107, 64,
                           218, 135, 257, 243, 211, 236, 229, 259, 176, 159, 38]
    posts = pickle.load(open('../../data/Dw_data/posts_days_forumsV1.0.pickle', 'rb'))

    start_date = datetime.datetime.strptime('01-01-2016', '%m-%d-%Y')
    end_date = datetime.datetime.strptime('07-01-2017', '%m-%d-%Y')

    # df_posts = countConversations(start_date, end_date, forums_cve_mentions)
    # pickle.dump(df_posts, open('../../data/DW_data/posts_days_forumsV2.0.pickle', 'wb'))

    vulnInfo = pickle.load(open('../../data/DW_data/09_15/Vulnerabilities-sample_v2+.pickle', 'rb'))

    cve_cpe_df =  pd.read_csv('../../data/DW_data/cve_cpe_map.csv')
    cve_cpe_map = store_cve_cpe_map(cve_cpe_df)

    pickle.dump(cve_cpe_map, open('../../data/DW_data/cve_cpe_map.pickle', 'wb') )

    # topCPEGroups(start_date, end_date, vulnInfo, cve_cpe_df, -1)
    exit()

    # users_CVEMap, CVE_usersMap = user_CVE_groups(cve_cpe_map, vulnInfo)

    # CVE_usersMap_filtered = {}
    # for cve in CVE_usersMap:
    #     if CVE_usersMap[cve] == 0:
    #         continue
    #     CVE_usersMap_filtered[cve] = CVE_usersMap[cve]
    #
    # pickle.dump((users_CVEMap, CVE_usersMap_filtered), open('../../data/DW_data/users_CVE_map.pickle', 'wb'))

    df_posts = pickle.load(open('../../data/DW_data/dw_database_data_2016-17.pickle', 'rb'))
    # df_posts = ldDW.getDW_data_postgres(forums_list=forums_cve_mentions, start_date=start_date.strftime("%Y-%m-%d"),
    #                                        end_date=end_date.strftime("%Y-%m-%d"))
    # pickle.dump(df_posts, open('../../data/DW_data/dw_database_data_2016-17.pickle', 'wb'))
    KB_edgesDf = computeKBnetwork(start_date, end_date, df_posts)
    pickle.dump(KB_edgesDf, open('../../data/DW_data/KB_edges_df.pickle', 'wb'))

    # print(KB_edgesDf)

