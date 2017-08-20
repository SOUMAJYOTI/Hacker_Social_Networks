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


if __name__ == "__main__":
    startDate = "2015-01-01"
    endDate = "2016-04-01"

    results_df = pd.DataFrame()
    forumsList = [250, 220, 219, 179, 265, 98, 150, 121, 35, 214, 266, 89, 71, 197,
                  146, 147, 107, 64, 218, 135, 257, 84, 213, 243, 211, 161, 236, 38,
                  159, 176, 88, 229, 259]
    engine = create_engine('postgresql://postgres:Impossible2@10.218.109.4:5432/cve')
    for f in forumsList:
        query = "select forumsid, topicid, posteddate::date, postsid, uid from dw_posts where posteddate::date > '" \
                + startDate + "' and posteddate::date < '" + endDate + "' and forumsid= " + str(f)
        print("ForumId: ", f)

        df = pd.read_sql_query(query, con=engine)
        print(df)
        results_df = results_df.append(df)

    print(results_df)