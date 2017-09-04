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

def threadsTimeDist(data_df):
    threadids = list(set(data_df['topicid']))

    # for tid in threadids:
    #     dataThread = data_df[data_df['topicid'] == tid]



def threadsLenDist(data_df, topicsList):
    threadLength_list = []
    threadLenDist = {}
    threadLenDist[10] = 0
    threadLenDist[20] = 0
    threadLenDist[50] = 0
    threadLenDist[100] = 0
    threadLenDist[1000] = 0
    threadLenDist[2000] = 0

    for tl in topicsList:
        threadsList = data_df[data_df['topicid'] == tl]
        if len(threadsList) < 10:
            threadLenDist[10] += 1
        elif len(threadsList) >=10 and len(threadsList) < 20:
            threadLenDist[20] += 1
        elif len(threadsList) >=20 and len(threadsList) < 50:
            threadLenDist[50] += 1
        elif len(threadsList) >=50 and len(threadsList) < 100:
            threadLenDist[100] += 1
        elif len(threadsList) >=100 and len(threadsList) < 1000:
            threadLenDist[1000] += 1
        elif len(threadsList) >=1000:
            threadLenDist[2000] += 1

    print(threadLenDist)
    return threadLenDist


def plot_hist(data, numBins, xLabel='', yLabel='', titleName=''):
    plt.figure()
    n, bins, patches = plt.hist(data, bins=numBins, facecolor='g')
    plt.xlabel(xLabel, size=25)
    plt.ylabel(yLabel, size=25)
    plt.title(titleName, size=25)
    plt.grid(True)
    plt.tick_params('x', labelsize=20)
    plt.tick_params('y', labelsize=20)
    plt.show()


def plot_histLine(data, xLabel='', yLabel='', titleName=''):
    in_values = sorted(set(data))
    list_degree_values = list(data)
    in_hist = [list_degree_values.count(x) for x in in_values]

    plt.figure()
    plt.loglog(in_values, in_hist, basex=2, basey=2)
    # plt.xlim([1, 2 ** 14])
    plt.xlabel(xLabel, size=25)
    plt.ylabel(yLabel, size=25)
    plt.title(titleName, size=25)
    plt.grid(True)
    plt.tick_params('x', labelsize=20)
    plt.tick_params('y', labelsize=20)
    plt.show()


def plot_bar(data, xLabels):
    hfont = {'fontname': 'Arial'}
    ind = np.arange(len(data))
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111)

    width=0.35
    rects1 = ax.bar(ind, data, width,
                    color='#0000ff')  # axes and labels
    ax.set_xlim(-width, len(ind) + width)
    # ax.set_ylim(87, 95)
    ax.set_ylabel('Number of threads', size=40, **hfont)
    ax.set_xlabel('Length of threads', size=40, **hfont)
    ax.set_xticks(ind)
    xtickNames = ax.set_xticklabels(xLabels, **hfont)
    plt.setp(xtickNames, rotation=45, fontsize=5)
    plt.grid(True)
    plt.xticks(size=25)
    plt.yticks(size=25)
    plt.subplots_adjust(left=0.13, bottom=0.30, top=0.9)
    ## add a legend
    # ax.legend( (rects1[0], ('Men', 'Women') )

    plt.show()
    plt.close()


def getthreads(forumList, startDate, endDate):
    engine = create_engine('postgresql://postgres:Impossible2@10.218.109.4:5432/cve')
    results_df = pd.DataFrame()
    for f in forumList:
        print(f)
        query = "select forumsid, language,  postcontent, posteddate, topicid, topicsname, uid from " \
                " dw_posts where posteddate::date > '" \
                    + startDate + "' and posteddate::date < '" + endDate + "' and forumsid=" + str(f)

        df = pd.read_sql_query(query, con=engine)
        results_df = results_df.append(df)

    return results_df

if __name__ == "__main__":
    forums = [88, 248, 133, 62, 161, 84, 60, 104, 173, 250, 105, 147, 40, 197]

    startDate = "2016-01-01"
    endDate = "2016-04-01"

    #0. Get the DW data
    dwData = getthreads(forumList=forums, startDate=startDate, endDate=endDate)
    pickle.dump(dwData, open('../../data/DW_data/08_29/DW_data_selected_forums_Jan-Mar16.pickle', 'wb'))

    threadsTimeDist(dwData)

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



