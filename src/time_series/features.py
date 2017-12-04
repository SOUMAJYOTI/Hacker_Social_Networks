import pandas as pd
import networkx as nx
import operator
import pickle
# import userAnalysis as usAn
import numpy as np
import networkx.algorithms.cuts as nxCut
import datetime
import src.network_analysis.createConnections as ccon
import matplotlib.pyplot as plt
from dateutil.relativedelta import relativedelta
import multiprocessing
import gc
import re
import community


def threadCommon(df_posts, experts):
    topics = list(set(df_posts['topicid']))
    count_topicsComm = 0
    for topicid in topics:
        threads = df_posts[df_posts['topicid'] == topicid]
        threads.is_copy = False
        threads['DateTime'] = threads['posteddate'].map(str) + ' ' + threads['postedtime'].map(str)
        threads['DateTime'] = threads['DateTime'].apply(lambda x:
                                                        datetime.datetime.strptime(x, '%Y-%m-%d %H:%M:%S'))
        threads = threads.sort('DateTime', ascending=True)
        threadUsers = threads['uid']

        for e in experts:
            if e in threadUsers:
                count_topicsComm += 1
                break

    return count_topicsComm

# def avgTime_user_interact() ## DO NOT HAVE THE TIME OF POSTING CORRECTLY

def community_detect(nw, experts, users):
    partition = community.best_partition(nw)

    comm_experts = [] # communities experts belong to
    for e in experts:
        comm_experts.append(partition[e])

    comm_experts = list(set(comm_experts))

    ''' Check how many users in the current day share communities with the experts '''
    user_count = 0
    for u in users:
        if partition[u] in comm_experts:
            


if __name__ == "__main__":
    G = nx.erdos_renyi_graph(10, 0.01)
    community_detect(G, [], [])







