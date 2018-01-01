import operator
import numpy as np
import networkx.algorithms.cuts as nxCut
import datetime
import community
import networkx as nx


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
        # if src in experts and tgt in experts:
        #     continue

        if src in experts:
            edge_list_new.append(('super_source', tgt))
        elif tgt in experts:
            edge_list_new.append((src, 'super_source'))
        elif src not in experts and tgt not in experts:
            edge_list_new.append((src, tgt))
        else:
            continue
    G_new = nx.DiGraph()
    G_new.add_edges_from(edge_list_new)

    return G_new


def threadCommon(df_posts, experts):
    topics = list(set(df_posts['topicid']))
    count_topicsComm = 0
    for topicid in topics:
        threads = df_posts[df_posts['topicid'] == topicid]
        threads.is_copy = False
        if threads.iloc[0]['postedtime'] == '':
            threads['DateTime'] = threads['posteddate'].map(str)
            threads['DateTime'] = threads['DateTime'].apply(lambda x:
                                                            datetime.datetime.strptime(x, '%Y-%m-%d'))
        else:
            threads['DateTime'] = threads['posteddate'].map(str) + ' ' + threads['postedtime'].map(str)
            threads['DateTime'] = threads['DateTime'].apply(lambda x:
                                                            datetime.datetime.strptime(x, '%Y-%m-%d %H:%M:%S'))
        threads = threads.sort_values(['DateTime'], ascending=True)
        threadUsers = list(threads['uid'])

        # print(type(threadUsers[0]), type(experts[0]), threadUsers[0], experts[0])
        for e in experts:
            if e in threadUsers:
                count_topicsComm += 1
                break

    return count_topicsComm


def community_experts(nw, experts):
    '''
    This function returns the communities of the experts in the KB graph
    '''
    partition = community.best_partition(nw.to_undirected())

    ''' List of Communities experts belong to '''
    comm_experts = []
    for e in experts:
        comm_experts.append(partition[e])

    comm_experts = list(set(comm_experts))

    return comm_experts, partition


def approximate_community_detect(G_merge, comm_partition, comm_experts, KB_users, experts, users):
    '''
    This function returns the number of users daily who share communities with the experts
    :param nw:
    :param users:
    :param comm_partition: communities of all nodes in KB graph
    :param comm_experts: communities of experts
    :return:
    '''
    ''' Check how many users in the current day share communities with the experts '''
    user_count = 0
    for u in users:
        if u in experts:
            user_count += 1
        else:
            '''
            Condition 1: If user is present in KB graph, check its community and return appropriate result
            Condition 2: If user is not in KB graph, i.e new user
                        Sub 1: If user is a direct neighbor of expert, return same community and hence true result
                                                        OR
                        Sub 2: If user has a neighbor who shares a community with expert, return true result (2-hop)
            '''
            if u in KB_users:
                if comm_partition[u] in comm_experts:
                    user_count += 1
            else:
                nbrs_user = list(G_merge.neighbors(u))
                common_exp = list(set(nbrs_user).intersection(set(experts)))
                if len(common_exp) >= 1:
                    user_count += 1
                    break

                # Reaching this loop means Condition 2: Sub 1 is not satisifed
                for nb in nbrs_user:
                    try:
                        if comm_partition[nb] in comm_experts:
                            user_count += 1
                            break
                    except:
                        continue

    # print(user_count, len(users))
    return user_count # TODO: normalized feature values


def shortestPaths(network, experts, users, weighted=False):
    '''
    Compute the shortest paths between FROM experts TO users
    :param experts:
    :param users:
    :return:
    '''

    sum_path_length = 0
    count_user_paths = 0
    for u in users:
        min_sp = 100000 # stores the minimum among all experts
        for e in experts:
            try:
                if u == e:
                    continue
                if weighted == False:
                    p = nx.shortest_path_length(network, source=e, target=u) # SOUREC  = expert
                else:
                    p = nx.shortest_path_length(network, source=e, target=u, weight='weight')
                if p < min_sp:
                    min_sp = p
            except nx.NetworkXNoPath:
                continue

        if min_sp < 100000:
            sum_path_length += min_sp
            count_user_paths += 1
        else:
            continue

    if count_user_paths == 0:
        return -1.
    else:
        return sum_path_length / count_user_paths


''' Maximum flow as a measure of trust propagation between experts and users '''
def maximum_flow(network, experts, users):
    ''

def computeDegreeMatrix(network):
    num_nodes = len(list(network.nodes()))

    volume = 0
    for n in network.nodes():
        volume += (network.out_degree(n) + network.in_degree(n))

    return volume


# Compute the commute-time distance between the users and the experts
def commuteTime(pseudo_lapl_G, nodeIndexMap, experts, users):
    nodeList = list(nodeIndexMap.keys())
    avg_mooreInv = np.mean(pseudo_lapl_G.diagonal())

    # volume = computeDegreeMatrix(G)
    avg_dist = 0.
    count_user_paths = 0
    for u in users:
        sum_dist = 0
        count_exp_paths = 0
        for e in experts:
            if u == e:
                continue
            if u not in nodeList:
                l_ii = avg_mooreInv
                l_ij = np.mean(pseudo_lapl_G[:, nodeIndexMap[e]])
            else:
                l_ii = pseudo_lapl_G[nodeIndexMap[u], nodeIndexMap[u]]
                l_ij = pseudo_lapl_G[nodeIndexMap[u], nodeIndexMap[e]]

            l_jj = pseudo_lapl_G[nodeIndexMap[e], nodeIndexMap[e]]
            sum_dist += ((l_ii + l_jj - 2*l_ij))
            # print(sum_dist)
            count_exp_paths += 1

        if count_exp_paths > 0:
            avg_dist += (sum_dist / count_exp_paths)
            count_user_paths += 1

    # print(avg_dist/ count_user_paths)
    if count_user_paths == 0:
        return 0.0
    else:
        return np.log(avg_dist)


def Conductance(network, userG1, userG2):
    try:
        conductanceVal = nxCut.conductance(network, userG1, userG2)
    except nx.NetworkXError:
        conductanceVal = 0.

    return conductanceVal


def centralities(network, arg, users):
    cent = {}
    if arg == "InDegree":
        cent = nx.in_degree_centrality(network)

    if arg == "OutDegree":
        cent = nx.out_degree_centrality(network)

    if arg == "PageRank":
        cent = nx.pagerank(network)

    if arg == "Core":
        cent = nx.core_number(network)

    cent_sum = 0.
    for u in users:
        cent_sum += cent[u]

    return cent_sum / len(users)


def getDegDist(G, users):
    degList = [] # Stores the in-degree list ---> how many people replied to this guy

    for u in users:
        in_deg = G.in_degree(u)
        degList.append(in_deg)

    return degList









