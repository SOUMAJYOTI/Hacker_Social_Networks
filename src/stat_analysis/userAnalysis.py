import pandas as pd
import pickle
import operator
import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
from sqlalchemy import create_engine
import datetime

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


def edgeCountPairs(network_df):
    edgeCount = {}
    for idx, row in network_df.iterrows():
        src = row['source']
        tgt = row['target']

        pairUsers = str(src) + '_' + str(tgt)
        if pairUsers not in edgeCount:
            edgeCount[pairUsers] = 0

        edgeCount[pairUsers] += 1

    return edgeCount


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


def relevantInfUsers(centDict, newUsers):
    countRel = 0
    for uid in newUsers:
        if uid in centDict.keys():
            countRel += 1

    return countRel


def topKUsers(centDict, K):
    sortedDict = sorted(centDict.items(), key=operator.itemgetter(1), reverse=True)[:K]
    newCentDict = {}
    for key, v in sortedDict:
        newCentDict[key] = v

    return newCentDict


def computeCentrality(network, arg):
    if arg == "InDegree":
        cent = nx.in_degree_centrality(network)

    if arg == "OutDegree":
        cent = nx.out_degree_centrality(network)

    if arg == "Pagerank":
        cent = nx.pagerank(network)

    if arg == "core":
        cent = nx.core_number(network)

    if arg == 'neighbors':
        cent = nx.degree(network)

    return cent


def plot_DegDist(data, title=''):
    sorted_X = sorted(set(data.values()))
    Y = list(data.values())
    distributionX = [Y.count(x) for x in sorted_X]
    plt.figure()
    plt.loglog(sorted_X, distributionX, 'ro', basex=2, basey=2)
    # plt.xlim([])
    plt.xlabel('Out-Degree (Neighbors)', size=35)
    plt.ylabel('Number of nodes', size=30)
    plt.xticks(size=30)
    plt.yticks(size=30)
    plt.title(title, size=25)
    plt.grid(True)
    plt.show()


def userDegBoxPlot(degList, xlabels):
    fig = plt.figure(1, figsize=(10, 8))

    # Create an axes instance
    ax = fig.add_subplot(111)

    # Create the boxplot
    bp = ax.boxplot(degList, patch_artist=True)

    for box in bp['boxes']:
        # change outline color
        box.set(color='#000000', linewidth=2)
        # change fill color
        box.set(facecolor='#FFFFFF')

        ## change color and linewidth of the whiskers
        # for whisker in bp['whiskers']:
        #     whisker.set(color='#7570b3', linewidth=2)

        ## change color and linewidth of the caps
        # for cap in bp['caps']:
        #     cap.set(color='#7570b3', linewidth=2)

        ## change color and linewidth of the medians
    for median in bp['medians']:
        median.set(color='#000000', linewidth=4)

        ## change the style of fliers and their fill
    for flier in bp['fliers']:
        flier.set(marker='o', color='#e7298a', alpha=0.5)

    # ax.set_title('Motif transition:' + str(m4) + '-->' + str(m5))
    # ax.set_ylabel('Shortest Path', size=)
    ax.set_ylim([0, 30])
    ax.set_xticklabels(xlabels, rotation=60, ha='right')


    plt.tick_params('y', labelsize=20)
    plt.tick_params('x', labelsize=20)
    plt.xlabel('Date(Start of each week)', fontsize=25)
    plt.ylabel('Degrees', fontsize=25)
    plt.title('Users : In-Degree Distribution', fontsize=25)
    plt.subplots_adjust(left=0.12, bottom=0.25, top=0.95)

    plt.grid(True)
    plt.show()
    # plt.savefig(file_save)
    plt.close()


if __name__ == "__main__":
    startDate = "2010-07-01"
    endDate = "2016-08-01"

    # Load the data
    startyear = 2016
    startmonth = 9
    endyear = 2017
    endmonth = 8
    start_dates = [datetime.date(m // 12, m % 12 + 1, 1) for m in range(startyear * 12 + startmonth - 1, endyear * 12 + endmonth)]
    degList = pickle.load(open('../../data/DW_data/features/usersDegDistribution.pickle', 'rb'))
    for degl in degList:
        count = 0
        for d in degl:
            if d > 10:
                count += 1

        print(count, len(degl), count/len(degl)*100)
    # userDegBoxPlot(degList, start_dates)