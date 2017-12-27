import sys
sys.path.insert(0, '../load_data/')

import load_data_api as ldap
import pandas as pd
import networkx as nx
import datetime as dt
import operator
import matplotlib.pyplot as plt
import numpy as np
from sqlalchemy import create_engine
import pickle
import datetime
import load_dataDW as ldDW


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


if __name__ == "__main__":
    titles = pickle.load(open('../../data/DW_data/09_15/train/features/titles_weekly.pickle', 'rb'))
    forums = [88, 248, 133, 62, 161, 84, 60, 104, 173, 250, 105, 147, 40, 197]

    vulData = pickle.load(open('../../data/DW_data/09_15/Vulnerabilities-sample_v2+.pickle', 'rb'))
    vulData['postedDate'] = pd.to_datetime(vulData['postedDate'], format='%Y-%m-%d')
    vulDataFiltered = vulData[vulData['forumID'].isin(forums)]

    # df_cve_cpe = pd.read_csv('../../data/DW_data/09_15/CVE_CPE_groups.csv')

    start_date = "2016-04-01"
    end_date= "2016-05-01"
    time_gap = 1

    numThreads = []
    numPosts = []
    #0. Get the DW data
    for idx in range(6):
        # KB network formation
        start_month = int(start_date[5:7]) + idx
        end_month = start_month + time_gap
        if start_month < 10:
            start_monthStr = str('0') + str(start_month)
        else:
            start_monthStr = str(start_month)

        if end_month < 10:
            end_monthStr = str('0') + str(end_month)
        else:
            end_monthStr = str(end_month)

        start_dateCurr = start_date[:5] + start_monthStr + start_date[7:]
        end_dateCurr = start_date[:5] + end_monthStr + start_date[7:]
        print("KB info: ")
        print("Start date: ", start_dateCurr, " ,End date: ", end_dateCurr)
        df_KB = ldDW.getDW_data_postgres(forums, start_dateCurr, end_dateCurr)
        vulDataFiltered = vulData[vulData['postedDate'] >= start_dateCurr]
        vulDataFiltered = vulDataFiltered[vulDataFiltered['postedDate'] < end_dateCurr]
        # print(vulDataFiltered)
        postsWeekly = segmentPostsWeek(df_KB)
        # postsWeekly = topCPEWeekly(vulDataFiltered)  # exit()

        for w in range(len(postsWeekly)):
            # print(len(postsWeekly[w]))
            numPosts.append(len(postsWeekly[w]))
            numThreads.append(len(list(set(postsWeekly[w]['topicid']))))

    plot_bar(numThreads, titles, 'Time frame(start of week)', 'Number of threads')
    plot_bar(numPosts, titles, 'Time frame(start of week)', 'Number of posts')


  
