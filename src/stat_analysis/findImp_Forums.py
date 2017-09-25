import sys
sys.path.insert(0, '../load_data/')

import load_data_api as ldap
import pandas as pd
import networkx as nx
import datetime as dt
import operator
import pickle
from sqlalchemy import create_engine


def dateToString(date):
    yearNum = str(date.year)
    monthNum = str(date.month)
    dayNum =str(date.day)
    if len(monthNum)<2:
        monthNum = "0"+monthNum
    if len(dayNum)<2:
        dayNum= "0"+dayNum
    return yearNum+"-"+monthNum+"-"+dayNum

def getPostsList(keyword_List, start_date, end_date):
    count_data = 0
    topicIds_list = []
    forum_topicIds = {}

    for kw in keyword_List:
        print("Keyword: ", kw)
        start = 0
        while True:
            print("Data count: ", count_data, " start: ", start)
            try:
                # results = ldap.getHackingPosts(start=start, limNum=5000, fId=f, fromDate=start_date, toDate=end_date)
                results = ldap.getHackingPosts_Content(searchContent=kw,
                                                       start=start, limNum=5000, fromDate=start_date, toDate=end_date)

            except:
                break

            if len(results) == 0:
                break

            # if count_data > 10000:
            #     break

            for r_idx in range(len(results)):
                item = results[r_idx]

                if item['postsId'] in postsId_seen:
                    continue
                postsId_seen.append(item['postsId'])
                if 'postedDate' not in item:
                    item['postedDate'] = ''
                if 'language' not in item:
                    item['language'] = ''
                item_dict = {'postsId': item['postsId'], "forumsId": item['forumsId'], 'postedDate': item['postedDate'],
                             'topicsId': item['topicId'], 'topicsName': item['topicsName'], 'language': item['language'],
                             'uid': item['uid']}

                item_df = pd.DataFrame(item_dict, index=[count])
                results_df = results_df.append(item_df)
                count += 1



            count_data += 5000
            start += 5000

    return forum_topicIds, topicIds_list


def getTopicIds(forumList, keyword_List, start_date, end_date):
    count_data = 0
    topicIds_list = []
    forum_topicIds = {}

    for kw in keyword_List:
    # for f in forumList:
        # print("Keyword: ", kw)
        start = 0
        while True:
            print("Data count: ", count_data, " start: ", start)
            try:
                results = ldap.getHackingPosts_Content_Forums(searchContent=kw, start=start, limNum=5000, fId=200, fromDate=start_date, toDate=end_date)
                # results = ldap.getHackingPosts_Content(searchContent=kw,
                #                                               start=start, limNum=5000, fromDate=start_date, toDate=end_date)

            except:
                break

            if len(results) == 0:
                break

            # if count_data > 10000:
            #     break

            for r_idx in range(len(results)):
                item = results[r_idx]

                if item['topicId'] not in topicIds_list:
                    topicIds_list.append(item['topicId'])

                f = item['forumsId']
                print(f, item['postedDate'])
                if f not in forum_topicIds:
                    forum_topicIds[f] = [item['topicId']]
                else:
                    forum_topicIds[f].append(item['topicId'])

            count_data += 5000
            start += 5000

    return forum_topicIds, topicIds_list


def topicCorrelation(topicsId_1, topicsId_2):
    totalCountForums = len(list(topicsId_1.keys()))
    print("Total global forums list: ", totalCountForums)
    count_forums_present = 0
    for f1 in topicsId_1:
        if f1 in topicsId_2:
            count_forums_present += 1

    print("% of forums relevant to Windows: ", count_forums_present/totalCountForums*100)


def getTopForumsByTopicsCount(forumTopicsList):
    forums_threadCount = {}
    for f in forumTopicsList:
        forums_threadCount[f] = len(forumTopicsList[f])
    sorted_forums = sorted(forums_threadCount.items(), key=operator.itemgetter(1), reverse=True)[:10]
    for k, v in sorted_forums:
        print(k, v)


def getCVEForums(kwList, cveData):
    forums = {}
    cves = {}
    for i, r in cveData.iterrows():
        fId = r['forumID']

        if fId not in cves:
            cves[fId] = []
        if fId not in forums:
            forums[fId] = 0
        if r['vulnId'] in cves[fId]:
            continue

        swTags = r['softwareTags']
        # if swTags == 'NA':
        #     continue
        datePost = dateToString(r['postedDate'])
        if datePost < '2015-01-01' or datePost >= '2016-04-01':
            continue

        cves[fId].append(r['vulnId'])
        forums[fId] += 1
        # flag = 0
        # for st in swTags:
        #     st = st.lower()
        #     for kw in kwList:
        #         if st in kw or kw in st:
        #             if fId not in forums:
        #                 forums[fId] = 0
        #             forums[fId] += 1
        #             if fId not in cves:
        #                 cves[fId] = []
        #             cves[fId].append(r['vulnId'])
        #
        #             flag = 1
        #             break
        #
        #     if flag == 1:
        #         break

    sortedForums = sorted(forums.items(), key=operator.itemgetter(1), reverse=True)
    return sortedForums


if __name__ == "__main__":
    # event_df = pd.read_csv('../../data/Armstrong_data/Windows_IE_DW_Jan15-Mar16.csv', encoding='ISO-8859-1')

    #0. Get the forums in which the CVEs are listed
    cveData = pickle.load(open("../../data/DW_data/08_29/Vulnerabilities-sample_v1+.pickle", 'rb'))
    cveData = cveData[cveData['indicator'] == 'Post']
    # print(cveData)
    # cveData = cveData[cveData['itemName'] == '']
    kw_List = ['windows', 'microsoft', 'vista', 'windows xp', 'win', 'windows 8', 'windows 8.1', 'windows 7',
               'operating system',  'linux', 'linux kernel', 'canonical', 'ubuntu', 'ubuntu os', 'apple', 'mac',
                 'mackintosh', 'mac_os', 'mac operating system', 'google', 'google chrome', 'chrome', 'chrome OS',
               'oracle', 'gnu', 'glibc', 'adobe', 'flash player', 'cisco', 'mcafee']

    sortedForums = getCVEForums(kw_List, cveData)
    for f, val in sortedForums:
        if val > 10:
            print(f, val)
    # 1. Find the forums and topics relevant to these keywords in the time frame
    # forumList =[]
    # kw_List = ['windows', 'microsoft', 'vista', 'windows xp', 'win', 'windows 8', 'windows 8.1', 'windows 7',
    #            'operating system',  'linux', 'linux kernel', 'canonical', 'ubuntu', 'ubuntu os', 'apple', 'mac',
    #              'mackintosh', 'mac_os', 'mac operating system', 'google', 'google chrome', 'chrome', 'chrome OS']
    # eventDesc = ['linux', 'linux kernel', 'canonical', 'ubuntu', 'ubuntu os']
    # start_date = dt.datetime.strptime('2016-01-01', '%Y-%m-%d')
    # end_date = dt.datetime.strptime('2016-03-30', '%Y-%m-%d')
    # forum_topics, topics_List = getTopicIds(forumList, kw_List, start_date, end_date)
    #
    # pickle.dump((forum_topics, topics_List), open('../../data/DW_data/windows_topicIDs_Jan-Mar2016.pickle', 'wb'))

    # 2. Find the forums relevant to one set of events out of all the events
    # forum_topics_1, topicsList_1 = pickle.load(open('../../data/DW_data/all_topicIDs_Jan-Mar2016.pickle', 'rb'))
    # forum_topics_2, topicsList_2 = pickle.load(open('../../data/DW_data/windows_topicIDs_Jan-Mar2016.pickle', 'rb'))
    #
    # topicCorrelation(forum_topics_1, forum_topics_2)

    # 3. Get the Top forums bu count of topic threads posted
    # getTopForumsByTopicsCount(forum_topics_1)

    # engine = create_engine('postgresql://postgres:Impossible2@10.218.109.4:5432/cve')
    # for f in forumsList:
    #     query = "select forumsid, topicid, posteddate::date, postsid, uid from dw_posts where posteddate::date > '" \
    #             + startDate + "' and posteddate::date < '" + endDate + "' and forumsid=" + str(f)
    #     print("ForumId: ", f)
    #     print(query)
    #
    #     df = pd.read_sql_query(query, con=engine)
    #     print(df)
        # results_df = results_df.append(df)

    # results_df.to_csv('../../data/DW_data/08_20/DW_data_selected_forums_Jul16.csv')