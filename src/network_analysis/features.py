import pandas as pd
import operator
import pickle
import numpy as np
import networkx.algorithms.cuts as nxCut
import datetime
import community
import networkx as nx
# import networkx.laplacian


def convert_graph_single_source(G, experts):
    '''
    The idea is to consider all the experts as one single source in the graph
    :param G:
    :param experts:
    :return:
    '''
    nbrs_dict = {} # compute the neighbors of the experts
    edge_list_new = [] # new edge list
    for e in G.edges():
        src, tgt = e

        # Ignore edges between the experts
        if src in e and tgt in e:
            continue

        if src in experts:
            edge_list_new.append(('super_source', tgt))

        if tgt in experts:
            edge_list_new.append((src, 'super_source'))

    G_new = nx.DiGraph()
    G_new.add_edges_from(edge_list_new)

    return G_new


def threadCommon(df_posts, experts):
    # print(len(experts))
    topics = list(set(df_posts['topicid']))
    count_topicsComm = 0
    for topicid in topics:
        threads = df_posts[df_posts['topicid'] == topicid]
        threads.is_copy = False
        threads['DateTime'] = threads['posteddate'].map(str) + ' ' + threads['postedtime'].map(str)
        threads['DateTime'] = threads['DateTime'].apply(lambda x:
                                                        datetime.datetime.strptime(x, '%Y-%m-%d %H:%M:%S'))
        threads = threads.sort('DateTime', ascending=True)
        threadUsers = list(threads['uid'].astype(str))

        # print(type(threadUsers[0]), type(experts[0]), threadUsers[0], experts[0])
        for e in experts:
            e = str(e)
            if e in threadUsers:
                count_topicsComm += 1
                break

    return count_topicsComm

# def avgTime_user_interact() ## DO NOT HAVE THE TIME OF POSTING CORRECTLY


def community_detect(nw, experts, users):
    G_new = convert_graph_single_source(nw, experts)
    partition = community.best_partition(G_new.to_undirected())

    # comm_experts = [] # communities experts belong to
    # for e in experts:
    #     comm_experts.append(partition[e])

    # comm_experts = list(set(comm_experts))
    # print(comm_experts, list(set(list(partition.values()))))
    ''' Check how many users in the current day share communities with the experts '''
    user_count = 0
    for u in users:
        if u in experts:
            user_count += 1
        else:
            print(partition[u])
            if partition[u] == partition['super_source']:# in comm_experts:
                user_count += 1

    print(user_count , len(users))
    return user_count / len(users) # normalized feature values


def shortestPaths_singleSource(network, experts, users):
    '''
    Compute the shortest paths between each user and the pool of experts
    :param network:
    :param experts:
    :param users:
    :return:
    '''

    G_new = convert_graph_single_source(network, experts)
    sum_path_length = 0
    count_user_paths = 0
    for u in users:
        u = str(u)
        if u in experts:
            continue
        try:
            p = nx.shortest_path(G_new, source=u, target='super_source')
            sum_path_length += len(p)
            count_user_paths += 1
        except:
            continue
    if count_user_paths == 0:
        return -1
    else:
        return sum_path_length / count_user_paths


def shortestPaths(network, experts, users):
    '''
    Compute the shortest paths between each user and the pool of experts
    :param network:
    :param experts:
    :param users:
    :return:
    '''

    avg_path_length = 0
    count_user_paths = 0
    for u in users:
        u = str(u)
        sum_paths = 0
        count_paths = 0
        for e in experts:
            e = str(e)
            try:
                if u == e:
                    continue
                p = nx.shortest_path(network, source=u, target=e)
                sum_paths += len(p)
                count_paths += 1
            except:
                continue

        if count_paths > 0:
            avg_path_length += (sum_paths/count_paths)
            count_user_paths += 1
    if count_user_paths == 0:
        return -1
    else:
        return avg_path_length / count_user_paths


''' Maximum flow as a measure of trust propagation between experts and users '''
# def maximum_flow(network, experts, users):


def computeDegreeMatrix(network):
    num_nodes = len(list(network.nodes()))

    volume = 0
    for n in network.nodes():
        volume += (network.out_degree(n) + network.in_degree(n))

    return volume


# Compute the commute-time distance between the users and the experts
def commuteTime(G, pseudo_lapl_G, experts, users):
    # Form the Laplacian of the graph
    # print('Computing laplacian...')

    volume = computeDegreeMatrix(G)
    avg_dist = 0.
    count_user_paths = 0
    for u in users:
        u = str(u)
        sum_dist = 0
        count_paths = 0
        for e in experts:
            e = str(e)
            # try:
            if u == e:
                continue
            l_ii = pseudo_lapl_G[u, u]
            l_jj = pseudo_lapl_G[e, e]
            l_ij = pseudo_lapl_G[u, e]
            sum_dist += (volume * (l_ii + l_jj - 2*l_ij))
            print(sum_dist)
            count_paths += 1
            # except:
            #     continue

        if count_paths > 0:
            avg_dist += (sum_dist / count_paths)
            count_user_paths += 1

    # print(avg_dist/ count_user_paths)
    if count_user_paths == 0:
        return -1
    else:
        return avg_dist/ count_user_paths


def commuteTime_singleSource(G, experts, users):
    # Form the Laplacian of the graph
    # print('Computing laplacian...')
    lapl_mat = nx.laplacian_matrix(G)
    pseudo_lapl_mat = np.linalg.pinv(lapl_mat)  # Compute the pseudo-inverse of the graph laplacian

    volume = computeDegreeMatrix(G)
    avg_dist = 0.
    count_user_paths = 0
    for u in users:
        u = str(u)
        sum_dist = 0
        count_paths = 0
        for e in experts:
            e = str(e)
            # try:
            if u == e:
                continue
            l_ii = pseudo_lapl_mat[u, u]
            l_jj = pseudo_lapl_mat[e, e]
            l_ij = pseudo_lapl_mat[u, e]
            sum_dist += (volume * (l_ii + l_jj - 2 * l_ij))
            count_paths += 1
            # except:
            #     continue

        if count_paths > 0:
            avg_dist += (sum_dist / count_paths)
            count_user_paths += 1

    return avg_dist / count_user_paths


#
#
# if __name__ == "__main__":
#     G = nx.erdos_renyi_graph(10, 0.01)
#     community_detect(G, [], [])






