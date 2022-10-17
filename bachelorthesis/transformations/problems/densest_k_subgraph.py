"""QUBO Transformation of the Densest k-Subgraph problem"""
import itertools
from typing import Union, List

import qubovert as qv
from networkx import Graph, DiGraph, non_edges
from numpy import zeros
from numpy.typing import NDArray

from bachelorthesis.transformations.problem import Problem


class DensestKSubgraph(Problem):
    def __init__(self, graph: Union[Graph, DiGraph], k: int):
        """The Densest k-Subgraph Problem

        Parameters
        ----------
        graph
            A networkx graph of the problem instance
        """
        self.graph: Union[Graph, DiGraph] = graph
        self.n: int = self.graph.order()
        self.k: int = k

    def gen_qubo(self) -> NDArray:
        """Code as example taken directly from Calude 2019 Paper -> doesn't match
        at all, but results can likely be modelled using the existing function."""

        qubo = zeros((self.n, self.n))

        for idx in range(self.n):
            for idx2 in range(self.n):
                qubo[idx][idx2] = self.n

        for idx in range(self.n):
            qubo[idx][idx] -= 2 * self.n * self.k

        for (idx, idx2) in self.graph.edges():
            qubo[idx][idx2] -= 1

        for idx in range(self.n):
            for idx2 in range(self.n):
                if idx > idx2:
                    qubo[idx2][idx] += qubo[idx][idx2]
                    qubo[idx][idx2] = 0

        return qubo
