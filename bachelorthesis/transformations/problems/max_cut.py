"""QUBO Transformation of the Max Cut problem"""
import itertools
from typing import Union, List

import qubovert as qv
from networkx import Graph, DiGraph, non_edges
from numpy import zeros
from numpy.typing import NDArray

from bachelorthesis.transformations.problem import Problem


class MaxCut(Problem):
    def __init__(self, graph: Union[Graph, DiGraph]):
        """The Max Cut Problem

        Parameters
        ----------
        graph
            A networkx graph of the problem instance
        """
        self.graph: Union[Graph, DiGraph] = graph
        self.n: int = self.graph.order()

    def gen_qubo(self) -> NDArray:
        # create list of variable names in order to ensure correct mapping
        var_names: List[str] = [f"x({node})" for node in range(self.n)]

        # create list of QUBO variables for use in the hamiltonian
        x: List[List[qv.QUBO]] = [
            qv.QUBO.create_var(var_names[node]) for node in range(self.n)
        ]

        a = 1

        hamiltonian = qv.QUBO()

        hamiltonian += -a * sum(  # nodes are in different sets
            x[edge[0]] + x[edge[1]] - 2 * x[edge[0]] * x[edge[1]]
            for edge in self.graph.edges
        )

        hamiltonian.set_mapping({var_names[i]: i for i in range(len(var_names))})

        if not hamiltonian.Q:
            return zeros((self.n * self.n, self.n * self.n))

        qubo = qv.utils.qubo_to_matrix(Q=hamiltonian.to_qubo(), array=True)

        return qubo
