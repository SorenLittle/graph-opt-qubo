"""QUBO Transformation of Traveling Salesperson problem"""
import itertools
from typing import Union, List

import qubovert as qv
from networkx import Graph, DiGraph, non_edges
from numpy import zeros
from numpy.typing import NDArray

from bachelorthesis.transformations.problem import Problem


class HamiltonianCycle(Problem):
    def __init__(self, graph: Union[Graph, DiGraph]):
        """The Traveling Salesperson Problem

        Parameters
        ----------
        graph
            A networkx graph of the problem instance
        """
        self.graph: Union[Graph, DiGraph] = graph
        self.n: int = self.graph.order()

    def gen_qubo(self) -> NDArray:
        if not self.graph.is_directed():
            graph: DiGraph = self.graph.to_directed()
        else:
            graph: DiGraph = self.graph

        # create list of variable names in order to ensure correct mapping
        var_names: List[str] = [
            f"x({node})({position})"
            for node, position in itertools.product(range(self.n), range(self.n))
        ]

        # create list of QUBO variables for use in the hamiltonian
        x: List[List[qv.QUBO]] = [
            [
                qv.QUBO.create_var(var_names[self.n * node + position])
                for position in range(self.n)
            ]
            for node in range(self.n)
        ]

        a = 1

        hamiltonian = qv.QUBO()

        hamiltonian += (
            a
            * sum(  # one node per position
                (1 - sum(x[v][j] for j in range(self.n))) ** 2 for v in range(self.n)
            )
            + a
            * sum(  # one position per node
                (1 - sum(x[v][j] for v in range(self.n))) ** 2 for j in range(self.n)
            )
            + a
            * sum(  # no invalid traversals
                sum(x[u][j] * x[v][(j + 1) % self.n] for j in range(self.n))
                for u, v in non_edges(graph)
            )
        )

        hamiltonian.set_mapping({var_names[i]: i for i in range(len(var_names))})
        hamiltonian.pop(())  # no idea why hamiltonian is adding a (): 16 k, v pair...

        if not hamiltonian.Q:
            return zeros((self.n * self.n, self.n * self.n))

        qubo = qv.utils.qubo_to_matrix(Q=hamiltonian.to_qubo(), array=True)

        return qubo
