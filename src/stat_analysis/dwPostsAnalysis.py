import pandas as pd
import networkx as nx
import datetime as dt
import operator
import matplotlib.pyplot as plt
import numpy as np
import pickle
import datetime


def plot_bar(data, xTicks, xLabels='', yLabels=''):
    hfont = {'fontname': 'Arial'}
    ind = np.arange(len(data))
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111)

    width=0.35
    rects1 = ax.bar(ind, data, width,
                    color='#0000ff')  # axes and labels
    ax.set_xlim(-width, len(ind) + width)
    # ax.set_ylim(87, 95)
    ax.set_ylabel(yLabels, size=30, **hfont)
    ax.set_xlabel(xLabels, size=30, **hfont)
    ax.set_xticks(ind)
    xtickNames = ax.set_xticklabels(xTicks, **hfont)
    plt.setp(xtickNames, rotation=45, fontsize=5)
    plt.grid(True)
    plt.xticks(size=20)
    plt.yticks(size=20)
    # plt.subplots_adjust(left=0.13, bottom=0.30, top=0.9)
    plt.subplots_adjust(left=0.13, bottom=0.25, top=0.9)
    ## add a legend
    # ax.legend( (rects1[0], ('Men', 'Women') )

    plt.show()
    plt.close()


def segmentPostsWeek(posts):
    posts['DateTime'] = posts['posteddate'].map(str) + ' ' + posts['postedtime'].map(str)
    posts['DateTime'] = posts['DateTime'].apply(lambda x:
                                                    datetime.datetime.strptime(x, '%Y-%m-%d %H:%M:%S'))
    posts = posts.sort('DateTime', ascending=True)

    # Form the network for each week
    start_year = posts['DateTime'].iloc[0].year
    start_month = posts['DateTime'].iloc[0].month

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
        start_date = datetime.datetime.strptime(str(start_year)+'-'+str(start_month)+'-'+start_dayStr+' 00:00:00', '%Y-%m-%d %H:%M:%S')
        weeksList.append(str(start_year)+'-'+str(start_month)+'-'+start_dayStr)

        end_day = start_day + 7
        if end_day > numDaysCurrMonth:
            end_day = numDaysCurrMonth

        if end_day < 10:
            end_dayStr = str('0') + str(end_day)
        else:
            end_dayStr = str(end_day)
        end_date = datetime.datetime.strptime(str(start_year) + '-' + str(start_month) + '-' + end_dayStr + ' 23:59:00',
                                                '%Y-%m-%d %H:%M:%S')

        posts_currWeek = posts[posts['DateTime'] >= start_date]
        posts_currWeek = posts_currWeek[posts_currWeek['DateTime'] < end_date]

        posts_WeeklyList.append(posts_currWeek)
        currIndex += len(posts_currWeek)
        start_day = end_day
        print(start_day, end_day)
        if start_day >= 29:
            break

    return posts_WeeklyList


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

if __name__ == "__main__":
    dwData = pd.read_pickle('../../data/DW_data/DW_data_new_2016-2017.pickle')
    topics = pd.read_pickle('../../data/DW_data/topics_new.pickle')

    # df_data = dwPosts_analysis(dwData, topics)
    # pickle.dump(df_data, open('../../data/DW_data/DW_data_forum_2016-2017.pickle', 'wb'))

    # dwData = pd.read_pickle('../../data/DW_data/dw_database_data_2016-17.pickle')

    # pickle.dump(create_dwdatabase(df_data), open('../../data/DW_data/dw_database_data_2016-17_new.pickle', 'wb'))

    df_data = pd.read_pickle('../../data/DW_data/dw_database_data_2016-17_new.pickle')
    print(df_data[:10])