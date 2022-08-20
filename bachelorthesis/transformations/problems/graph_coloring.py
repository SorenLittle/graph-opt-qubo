"""QUBO Formulation of the Graph Coloring Problem"""
import itertools
from typing import List

from networkx import Graph
from numpy import zeros
from numpy.typing import NDArray
import qubovert as qv

from bachelorthesis.transformations.problem import Problem


class GraphColoring(Problem):
    def __init__(self, colors: int, graph: Graph):
        """The Graph Coloring problem

        Parameters
        ----------
        k_colors
            the number of number of colors with which the induced subgraph
            should be k-colorable

        graph
            A networkx graph of the problem instance
        """

        self.colors = colors
        self.graph: Graph = graph

    def gen_qubo(self) -> NDArray:
        n = self.graph.order()
        k = self.colors

        # create list of variable names in order to ensure correct mapping
        var_names: List[str] = [
            f"x({node})({color})"
            for node, color in itertools.product(range(n), range(k))
        ]

        # create list of QUBO variables for use in the hamiltonian
        x_: List[qv.QUBO] = [qv.QUBO.create_var(var_names[i]) for i in range(n)]
        x = [x_[n: n + n] for n in range(0, len(x_), n)]

        # scaling constants
        a = 4
        c = 2

        # create hamiltonian of the problem
        hamiltonian = qv.QUBO()
        hamiltonian += a * sum(
            (1 - sum(x[v][i] for i in range(k))) ** 2 for v in range(n)
        ) + c * sum(
            sum(x[v][j] * x[w][j] for j in range(n))
            for v in range(n)
            for w in self.graph.neighbors(v)
        )

        hamiltonian.set_mapping({var_names[i]: i for i in range(len(var_names))})

        # return a correctly sized matrix if the problem is empty
        if not hamiltonian.Q:
            return zeros((n * k, n * k))

        qubo = qv.utils.qubo_to_matrix(Q=hamiltonian.to_qubo(), array=True)

        return qubo
