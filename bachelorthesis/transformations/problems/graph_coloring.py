"""QUBO Formulation of the Graph Coloring Problem"""
import itertools
from typing import List

import qubovert as qv
from networkx import Graph
from numpy import zeros
from numpy.typing import NDArray

from bachelorthesis.transformations.problem import Problem


class GraphColoring(Problem):
    def __init__(self, colors: int, graph: Graph):
        """The Graph Coloring problem

        Parameters
        ----------
        colors
            the number of number of colors with which the graph should be colored

        graph
            A networkx graph of the problem instance
        """

        self.graph: Graph = graph
        self.n: int = self.graph.order()
        self.k: int = colors

    def gen_qubo(self) -> NDArray:
        # create list of variable names in order to ensure correct mapping
        var_names: List[str] = [
            f"x({node})({color})"
            for node, color in itertools.product(range(self.n), range(self.k))
        ]

        # create list of QUBO variables for use in the hamiltonian
        x: List[List[qv.QUBO]] = [
            [
                qv.QUBO.create_var(var_names[self.k * node + position])
                for position in range(self.k)
            ]
            for node in range(self.n)
        ]

        # scaling constants
        a = 4
        c = 2

        # create hamiltonian of the problem
        hamiltonian = qv.QUBO()
        hamiltonian += a * sum(
            (1 - sum(x[v][i] for i in range(self.k))) ** 2 for v in range(self.n)
        ) + c * sum(
            sum(x[v][j] * x[w][j] for j in range(self.k))
            for v in range(self.n)
            for w in self.graph.neighbors(v)
        )

        hamiltonian.set_mapping({var_names[i]: i for i in range(len(var_names))})
        hamiltonian.pop(())  # no idea why hamiltonian is adding a (): 16 k, v pair...

        # return a correctly sized matrix if the problem is empty
        if not hamiltonian.Q:
            return zeros((self.n * self.k, self.n * self.k))

        qubo = qv.utils.qubo_to_matrix(Q=hamiltonian.to_qubo(), array=True)

        return qubo
