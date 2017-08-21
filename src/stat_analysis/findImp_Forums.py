import sys
sys.path.insert(0, '../load_data/')

import load_data_api as ldap
import pandas as pd
import networkx as nx
import datetime as dt
import operator
import pickle


def getTopicIds(keyword_List, start_date, end_date):
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

                if item['topicId'] not in topicIds_list:
                    topicIds_list.append(item['topicId'])

                f = item['forumsId']
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


if __name__ == "__main__":
    event_df = pd.read_csv('../../data/Armstrong_data/Windows_IE_DW_Jan15-Mar16.csv', encoding='ISO-8859-1')

    # 1. Find the forums and topics relevant to these keywords in the time frame
    kw_List = ['windows', 'microsoft', 'vista', 'windows xp', 'win', 'windows 8', 'windows 8.1', 'windows 7']
    # kw_List = ['windows', 'microsoft', 'vista', 'windows xp', 'win', 'windows 8', 'windows 8.1', 'windows 7',
    #            'operating system',  'linux', 'linux kernel', 'canonical', 'ubuntu', 'ubuntu os', 'apple', 'mac',
    #              'mackintosh', 'mac_os', 'mac operating system', 'google', 'google chrome', 'chrome', 'chrome OS']
    # eventDesc = ['linux', 'linux kernel', 'canonical', 'ubuntu', 'ubuntu os']
    start_date = dt.datetime.strptime('2016-01-01', '%Y-%m-%d')
    end_date = dt.datetime.strptime('2016-03-30', '%Y-%m-%d')
    forum_topics, topics_List = getTopicIds(kw_List, start_date, end_date)

    pickle.dump((forum_topics, topics_List), open('../../data/DW_data/windows_topicIDs_Jan-Mar2016.pickle', 'wb'))

    # 2. Find the forums relevant to one set of events out of all the events
    
