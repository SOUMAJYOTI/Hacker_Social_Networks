import sys
sys.path.insert(0, '../network_analysis/')
sys.path.insert(0, '../load_data/')
sys.path.insert(0, '../stat_analysis/')

import load_data_api as ldap
import pandas as pd
import networkx as nx
import datetime as dt
import operator
import pickle
from sqlalchemy import create_engine
import userAnalysis as usAn
import numpy as np
import networkx.algorithms.cuts as nxCut
import datetime
import createConnections as ccon
import load_dataDW as ldDW
import matplotlib.pyplot as plt


def dateToString(date):
    yearNum = str(date.year)
    monthNum = str(date.month)
    dayNum =str(date.day)
    if len(monthNum)<2:
        monthNum = "0"+monthNum
    if len(dayNum)<2:
        dayNum= "0"+dayNum
    return yearNum+"-"+monthNum+"-"+dayNum


def countConversations(start_date, end_date, forums):
    start_year = 2016
    start_month = 4

    daysMonths = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    df_postsTS = pd.DataFrame()

    datesList = []
    numPostsList = []
    uidsList = []
    uidsCount = []

    while start_month <= 12:
        print("Start Date:", start_date )
        postsDf = ldDW.getDW_data_postgres(forums, start_date, end_date)
        postsDf['DateTime'] = postsDf['posteddate'].map(str) + ' ' + postsDf['postedtime'].map(str)
        postsDf['DateTime'] = postsDf['DateTime'].apply(lambda x:
                                                        datetime.datetime.strptime(x, '%Y-%m-%d %H:%M:%S'))
        postsDf = postsDf.sort('DateTime', ascending=True)

        start_day = 1
        while True:
            usersTempList = []
            if start_day < 10:
                start_dayStr = str('0') + str(start_day)
            else:
                start_dayStr = str(start_day)

            if start_month < 10:
                start_monthStr = str('0') + str(start_month)
            else:
                start_monthStr = str(start_month)

            start_date = datetime.datetime.strptime(
                str(start_year) + '-' + start_monthStr + '-' + start_dayStr + ' 00:00:00', '%Y-%m-%d %H:%M:%S')

            end_date = datetime.datetime.strptime(
                str(start_year) + '-' + start_monthStr + '-' + start_dayStr + ' 23:59:00',
                '%Y-%m-%d %H:%M:%S')

            posts_currDay = postsDf[postsDf['DateTime'] >= start_date]
            posts_currDay = posts_currDay[posts_currDay['DateTime'] < end_date]

            datesList.append(start_date)
            numPostsList.append(len(posts_currDay))

            for idx, row in posts_currDay.iterrows():
                usersTempList.append(row['uid'])

            uidsList.append(usersTempList)
            uidsCount.append(len(list(set(usersTempList))))
            start_day += 1

            # Break condition
            if start_day > daysMonths[start_month - 1]:
                break

        start_month += 1
        if start_month < 10:
            start_monthStr = str('0') + str(start_month)
        else:
            start_monthStr = str(start_month)
        start_date = str(start_year) + '-' + start_monthStr + '-01'
        end_date = str(start_year) + '-' + start_monthStr + '-' + str(
                                                daysMonths[start_month - 1])

    df_postsTS['date'] = datesList
    df_postsTS['number_posts'] = numPostsList
    df_postsTS['users'] = uidsList
    df_postsTS['number_users'] = uidsCount

    return df_postsTS

if __name__ == "__main__":
    titles = pickle.load(open('../../data/DW_data/09_15/train/features/titles_weekly.pickle', 'rb'))
    forums_cve_mentions = [88, 248, 133, 49, 62, 161, 84, 60, 104, 173, 250, 105, 147, 40, 197]
    # engine = create_engine('postgresql://postgres:Impossible2@10.218.109.4:5432/cyber_events_pred')
    # query = "select vendor, product, cluster_tag, cve from  cve_cpegroups"
    # posts_df = pickle.load(open('../../data/DW_data/09_15/train/data/DW_data_selected_forums_Oct15-Mar16.pickle', 'rb'))
    vulData = pickle.load(open('../../data/DW_data/08_29/Vulnerabilities-sample_v2+.pickle', 'rb'))
    vulDataFiltered = vulData[vulData['forumID'].isin(forums_cve_mentions)]

    read_path = '../../data/Armstrong_data/eventsDF_v1.0-demo.csv'
    amEvents = pd.read_csv(read_path)


    start_date = '2016-04-01'
    end_date = '2016-5-01'
    df_postsTS = countConversations(start_date, end_date, forums_cve_mentions)
    print(df_postsTS)
    pickle.dump(df_postsTS, open('../../data/DW_data/posts_daysV1.0.pickle', 'wb'))

    df_postsTS.plot(x='date', y='number_posts')
    plt.grid()
    plt.xticks(size=20)
    plt.yticks(size=20)
    plt.xlabel('Date Time frame', size=20)
    plt.ylabel('Number of posts', size=20)
    plt.subplots_adjust(left=0.13, bottom=0.15, top=0.9)

    plt.show()
    plt.close()

    df_postsTS.plot(x='date', y='number_users')
    plt.grid()
    plt.xticks(size=20)
    plt.yticks(size=20)
    plt.xlabel('Date Time frame', size=20)
    plt.ylabel('Number of Users', size=20)
    plt.subplots_adjust(left=0.13, bottom=0.15, top=0.9)

    plt.show()

    # df_cve_cpe = pd.read_csv('../../data/DW_data/09_15/CVE_CPE_groups.csv')
    #
    # users_CVE_map, CVE_users_map = user_CVE_groups(df_cve_cpe, vulData)
    # feat_experts, feat_topUsers, titlesList = \
    #     monthlyFeatureCompute(forums_cve_mentions, start_date, users_CVE_map, CVE_users_map, vulDataFiltered,
    #                           df_cve_cpe, amEvents, titles)
