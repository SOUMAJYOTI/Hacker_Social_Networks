from scipy.stats import powerlaw

import numpy as np
import networkx as nx
import random


class BaseGraph:

    def __init__(self):
        self.structure = nx.Graph()

    def nodes(self, node_data=False):
        return self.structure.nodes(data=node_data)

    def edges(self, edge_data=False):
        return self.structure.edges(data=edge_data)

    def neighbors(self, node):
        return self.structure.neighbors(node)

    def order(self):
        return self.structure.order()

    def size(self, weight=None):
        return self.structure.size(weight=weight)

    def edge_weight(self, a, b):
        return self.structure[a][b]['weight']

    def __createGraph(self, nodes, edges):
        self.structure.add_nodes_from(nodes)
        self.structure.add_weighted_edges_from(edges)

    def __computeCentralities(self, arg):
        cent = 0.
        if arg == "InDegree":
            cent = nx.in_degree_centrality(self.structure)

        if arg == "OutDegree":
            cent = nx.out_degree_centrality(self.structure)

        if arg == "Pagerank":
            cent = nx.pagerank(self.structure)

        if arg == "core":
            cent = nx.core_number(self.structure)

        return cent


class CommunityClusters(BaseGraph):
    def __init__(self, lambdaVal=10):
        self.lambdaVal = lambdaVal
        self.clusters = {}
        self.clCount = 0
        self.clusterNodeDict = {}
        BaseGraph.__init__(self)

    def __constructCommunities(self):
        for n in self.nodes():
            self.cluster = n

        # Randomly select the seed nodes
        numNodes = len(self.structure)
        nodes =


        # Starting from the maximum value of $\lambda$, incrementally connect the users
        # Connect the users if $w_ij$ $\geq$ $\lambda$
        while True:
            for n in self.nodes():
                if n not in self.clusterNodeDict:
                    self.clusterNodeDict[n] = self.clCount

                for nbr in self.neighbors(n):
                    if self.edge_weight(n, nbr) >= self.lambdaVal:
                        if nbr not in self.clusterNodeDict:
                            self.clusterNodeDict[nbr] = self.clusterNodeDict[n]

                        self.clusters[self.clCount]







