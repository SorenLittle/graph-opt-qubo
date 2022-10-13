"""QUBO Transformation of the Clique Cover problem"""
import itertools
from typing import Union, List

import qubovert as qv
from networkx import Graph, DiGraph, non_edges
from numpy import zeros
from numpy.typing import NDArray

from bachelorthesis.transformations.problem import Problem


class CliqueCover(Problem):
    def __init__(self, colors: int, graph: Union[Graph, DiGraph]):
        """The Clique Cover Problem

        Parameters
        ----------
        colors
            The number of colors with which the graph should be colored

        graph
            A networkx graph of the problem instance
        """
        self.graph: Union[Graph, DiGraph] = graph
        self.k: int = self.graph.order()
        self.n: int = colors

    def gen_qubo(self) -> NDArray:
        # create list of variable names in order to ensure correct mapping
        var_names: List[str] = [
            f"x({node})({color})"
            for node, color in itertools.product(range(self.n), range(self.k))
        ]

        # create list of QUBO variables for use in the hamiltonian
        x: List[List[qv.QUBO]] = [
            [
                qv.QUBO.create_var(var_names[self.n * node + position])
                for position in range(self.n)
            ]
            for node in range(self.k)
        ]

        a = 1
        b = 2 * 1

        hamiltonian = qv.QUBO()

        print(3)
        hamiltonian += a * sum(
            (1 - sum(x[v][i] for i in range(self.n))) ** 2 for v in range(self.k)
        ) + b * sum(
            (
                0.5
                * (-1 + sum(x[v][i] for v in range(self.k)))
                * sum(x[v][i] for v in range(self.k))
                - sum(x[u][i] * x[v][i] for u, v in self.graph.edges())
            )
            for i in range(self.n)
        )

        print(4)
        hamiltonian.set_mapping({var_names[i]: i for i in range(len(var_names))})
        hamiltonian.pop(())  # no idea why hamiltonian is adding a (): 16 k, v pair...

        print(5)
        if not hamiltonian.Q:
            return zeros((self.n * self.n, self.n * self.n))

        qubo = qv.utils.qubo_to_matrix(Q=hamiltonian.to_qubo(), array=True)

        return qubo
