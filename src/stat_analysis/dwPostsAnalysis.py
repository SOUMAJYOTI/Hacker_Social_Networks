import pandas as pd
import networkx as nx
import datetime as dt
import operator
import matplotlib.pyplot as plt
import numpy as np
import pickle
import datetime


def topCPEWeekly(dfCPE):
    dfCPE = dfCPE.sort('postedDate', ascending=True)

    # Form the network for each week
    start_year = dfCPE['postedDate'].iloc[0].year
    start_month = dfCPE['postedDate'].iloc[0].month

    # print(start_year, start_month)

    start_day = 1
    currIndex = 0
    posts_WeeklyList = []
    daysMonths = [31, 30, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    numDaysCurrMonth = daysMonths[start_month-1]
    weeksList = []

    while True:
        if start_day < 10:
            start_dayStr = str('0') + str(start_day)
        else:
            start_dayStr = str(start_day)
        start_date = pd.to_datetime(
            str(start_year) + '-' + str(start_month) + '-' + start_dayStr, format='%Y-%m-%d')
        weeksList.append(str(start_year) + '-' + str(start_month) + '-' + start_dayStr)

        end_day = start_day + 7
        if end_day > numDaysCurrMonth:
            end_day = numDaysCurrMonth

        if end_day < 10:
            end_dayStr = str('0') + str(end_day)
        else:
            end_dayStr = str(end_day)
        end_date = pd.to_datetime(str(start_year) + '-' + str(start_month) + '-' + end_dayStr, format='%Y-%m-%d' )

        # print(start_date, end_date)
        posts_currWeek = dfCPE[dfCPE['postedDate'] >= start_date]
        posts_currWeek = posts_currWeek[posts_currWeek['postedDate'] < end_date]

        # print(posts_currWeek)
        posts_WeeklyList.append(posts_currWeek)
        currIndex += len(posts_currWeek)
        start_day = end_day
        print(start_day, end_day)
        if start_day >= 29:
            break

    return posts_WeeklyList


def dwPosts_analysis(df_data, topics):
    topic_forumsMap = {}
    for idx, row in topics.iterrows():
        topic = row['id']
        forum = row['forum_id']

        topic_forumsMap[topic]  = forum

    print('Assign forum....')
    forumList = []
    for idx, row in df_data.iterrows():
        topic = row['topic_id']

        if topic not in topic_forumsMap:
            forumList.append(' ')
        else:
            forumList.append(topic_forumsMap[topic])

    df_data['forum'] = forumList

    return df_data


def create_dwdatabase(df_data):
    forumsList = []
    topicsList = []
    postedDateList = []
    postedTimeList = []
    postidList = []
    uidList = []

    for idx, row in df_data.iterrows():
        forumsList.append(row['forum'])
        topicsList.append(row['topic_id'])
        postedDateList.append(row['date_posted'])
        postedTimeList.append('')
        postidList.append(row['id'])
        uidList.append(row['user_id'])

    df_database = pd.DataFrame()
    df_database['forumsid'] = forumsList
    df_database['topicid'] = topicsList
    df_database['posteddate'] = postedDateList
    df_database['postedtime'] = postedTimeList
    df_database['postsid'] = postidList
    df_database['uid'] = uidList

    return df_database


def groupForumThreads(df_data):
    # print(df_data[:10])
    forumsCount = (df_data.groupby(by='forum', ).size()).sort_values(ascending=False)

    forumsFinal = []
    for idx, row in forumsCount.iteritems():
        if row > 1000: # this is the threshold of forum posts count to consider
            forumsFinal.append(idx)

    print(forumsFinal)
    print(len(forumsFinal))


if __name__ == "__main__":
    dwData = pd.read_pickle('../../data/DW_data/new_DW/DW_postgres_data_new_2016-2017.pickle')
    topics = pd.read_pickle('../../data/DW_data/new_DW/topics_new.pickle')

    # df_data = dwPosts_analysis(dwData, topics)
    df_data = pickle.load(open('../../data/DW_data/new_DW/DW_postgres_data_forum_2016-2017.pickle', 'rb'))
    # pickle.dump(df_data, open('../../data/DW_data/DW_data_forum_2016-2017.pickle', 'wb'))

    # dwData = pd.read_pickle('../../data/DW_data/dw_database_data_2016-17.pickle')

    # pickle.dump(create_dwdatabase(df_data), open('../../data/DW_data/new_DW/dw_database_dataframe_2016-17_new.pickle', 'wb'))

    # df_data = pd.read_pickle('../../data/DW_data/new_DW/dw_database_dataframe_2016-17_new.pickle')
    #
    groupForumThreads(df_data)