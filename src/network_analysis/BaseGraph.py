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

    def order(self):
        return self.structure.order()

    def size(self, weight=None):
        return self.structure.size(weight=weight)

    def createGraph(self, nodes, edges):
        self.structure.add_nodes_from(nodes)
        self.structure.add_weighted_edges_from(edges)

    def computeCentralities(self, arg):
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


class CommunityDendograms(BaseGraph):
    def __init__(self, lambdaVal=10):
        self.lambdaVal = lambdaVal
        BaseGraph.__init__(self)

    def _constructCommunities(self):
        for n in self.nodes():
            self.cluster = n

        # Starting from the maximum value of $\lambda$, incrementally connect the users
        # Connect the users if $w_ij$ $\geq$ $\lambda$
        while True:

            





