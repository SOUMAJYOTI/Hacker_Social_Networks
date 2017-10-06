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
    plt.subplots_adjust(left=0.13, bottom=0.50, top=0.9)
    ## add a legend
    # ax.legend( (rects1[0], ('Men', 'Women') )

    plt.show()
    plt.close()


# def getThreads(forumList, startDate, endDate):
#     engine = create_engine('postgresql://postgres:Impossible2@10.218.109.4:5432/cve')
#     results_df = pd.DataFrame()
#     for f in forumList:
#         query = "select forumsid, language,  postcontent, posteddate, topicid, topicsname, uid from " \
#                 " dw_posts where posteddate::date > '" \
#                     + startDate + "' and posteddate::date < '" + endDate + "' and forumsid=" + str(f)
#
#         df = pd.read_sql_query(query, con=engine)
#         results_df = results_df.append(df)
#
#     return results_df


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
    daysMonths = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
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
        if start_day >= 29:
            break

    return posts_WeeklyList


if __name__ == "__main__":
    titles = pickle.load(open('../../data/DW_data/09_15/train/features/titles_weekly.pickle', 'rb'))
    forums = [88, 248, 133, 62, 161, 84, 60, 104, 173, 250, 105, 147, 40, 197]

    start_date = "2016-01-01"
    end_date= "2016-04-01"
    KB_gap = 3

    numThreads = []
    numPosts = []
    #0. Get the DW data
    for idx in range(6):
        # KB network formation
        start_month = int(start_date[5:7]) + idx
        end_month = start_month + KB_gap
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

        postsWeekly = segmentPostsWeek(df_KB)

        for w in range(len(postsWeekly)):
            numPosts.append(len(postsWeekly[w]))
            numThreads.append(len(list(set(postsWeekly[w]['topicid']))))

    plot_bar(numThreads, titles, 'Time frame(start of week)', 'Number of threads')
    plot_bar(numPosts, titles, 'Time frame(start of week)', 'Number of posts')


    # topicsId_list = list(set(posts_df['topicid'].tolist()))
    # threadLength_list = threadsLenDist(posts_df, topicsId_list)

    # print(threadLength_list)
    # plot_hist(threadLength_list, 20)
    # data_to_plot = []
    # threadLength_list = sorted(threadLength_list.items(), key=operator.itemgetter(0))
    # for k, v in threadLength_list:
    #     data_to_plot.append(v)
    #     xLabels = ['< 10', '10 and 20', '20 and 50', '50 and 100', '100 and 1000', '> 1000']
    # plot_bar(data_to_plot, xLabels)


