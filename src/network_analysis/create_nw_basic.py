import pandas as pd
import pickle
import operator
import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
from sqlalchemy import create_engine


def binarySearch(alist, item):
    first = 0
    last = len(alist)-1
    found = False

    while first<=last and not found:
        midpoint = (first + last)//2
        if alist[midpoint] == item:
            found = True
        else:
            if item < alist[midpoint]:
                last = midpoint-1
            else:
                first = midpoint+1

    return found


def store_neighbors(network_df):
    network_nbrList = {}
    for idx, row in network_df.iterrows():
        src = row['source']
        tgt = row['target']
        rtTime = pd.to_datetime(row['date'])

        if src not in network_nbrList:
            network_nbrList[src] = []

        if tgt not in network_nbrList[src]:
            network_nbrList[src].append(tgt)

    return network_nbrList


def store_edges(network_df):
    network_edgeList = []
    for idx, row in network_df.iterrows():
        src = row['source']
        tgt = row['target']

        if (src, tgt) not in network_edgeList:
            network_edgeList.append((src, tgt))

    return network_edgeList


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


def computeCentrality(network, arg):
    if arg == "InDegree":
        cent = nx.in_degree_centrality(network)

    if arg == "OutDegree":
        cent = nx.out_degree_centrality(network)

    if arg == "Pagerank":
        cent = nx.pagerank(network)

    # if arg == "OutDegree":
    #     cent = nx.out_degree_centrality(network)

    return cent


if __name__ == "__main__":
    startDate = "2010-07-01"
    endDate = "2016-08-01"

    results_df = pd.DataFrame()
    # forumsList = [259]
    # forumsList = [113]

    # engine = create_engine('postgresql://postgres:Impossible2@10.218.109.4:5432/cve')
    # for f in forumsList:
    #     query = "select forumsid, topicid, posteddate::date, postsid, uid from dw_posts where posteddate::date > '" \
    #             + startDate + "' and posteddate::date < '" + endDate + "' and forumsid=" + str(f)
    #     print("ForumId: ", f)
    #     print(query)
    #
    #     df = pd.read_sql_query(query, con=engine)
    #     print(df)
    #     results_df = results_df.append(df)

    # results_df.to_csv('../../data/DW_data/08_20/DW_data_selected_forums_Jul16.csv')

    dw_user_edges = pickle.load(open('../../data/Mohammed/DW_user_edges_DataFrame_June15-June16.pickle', 'rb'))
    posts_df = pd.read_csv('../../data/DW_data/08_20/DW_data_selected_forums_Jan-Mar16.csv')

    topicsId_list = list(set(posts_df['topicid'].tolist()))
    dw_user_edges = dw_user_edges[dw_user_edges['topicid'].isin(topicsId_list)]


