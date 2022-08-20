"""QUBO Formulation of the Longest Path Problem"""
import itertools
from typing import List

from networkx import Graph
from numpy import zeros
from numpy.typing import NDArray
import qubovert as qv

from bachelorthesis.transformations.problem import Problem


class LongestPath(Problem):
    def __init__(self, graph: Graph, start_node: int, terminal_node: int, steps: int):
        """The Longest Path Problem

        Parameters
        ----------
        graph
            A networkx graph of the problem instance
        """
        self.graph: Graph = graph
        self.start_node: int = start_node
        self.terminal_node: int = terminal_node
        self.steps: int = steps
        self.n: int = self.graph.order()
        self.k: int = self.steps + 1

    def gen_qubo(self) -> NDArray:
        # create list of variable names in order to ensure correct mapping
        var_names: List[str] = [
            f"x({node})({position})"
            for node, position in itertools.product(range(self.k), range(self.n))
        ]

        # create list of QUBO variables for use in the hamiltonian
        x: List[List[qv.QUBO]] = [
            [
                qv.QUBO.create_var(var_names[self.k * node + position])
                for position in range(self.k)
            ]
            for node in range(self.n)
        ]

        # set scaling constant based on heuristic in paper
        edge_weights = [
            self.graph.get_edge_data(edge[0], edge[1]).get("weight")
            for edge in self.graph.edges
        ]
        a: int = self.steps * max(edge_weights)

        h_p = sum(
            -sum(x[i][p] ** 2 for i in range(self.n))
            + 2
            * sum(
                x[i][p] * x[j][p] for i in range(self.n) for j in range(i + 1, self.n)
            )
            for p in range(self.k)
        )

        h_v = sum(
            sum(x[i][p] * x[i][q] for p in range(self.k) for q in range(p + 1, self.k))
            for i in range(self.n)
        )

        h_w = sum(
            -self.graph.get_edge_data(i, j).get("weight") * x[i][p] * x[j][p + 1]
            for i in range(self.n)
            for j in range(i, self.n)
            # NOTE: To match the papers example this range is (i, n) instead of (n). The
            # actual formula in the paper is likely incorrect as it would not produce
            # the same qubo matrix they give as an example
            if (i, j) in self.graph.edges
            for p in range(self.k - 1)
        )

        h_w_invalid = sum(
            x[i][p] * x[j][p + 1]
            for i in range(self.n)
            for j in range(i, self.n)  # NOTE: same as in h_w
            if (i, j) not in self.graph.edges
            for p in range(self.k - 1)
        )

        hamiltonian = qv.QUBO()
        hamiltonian += (
            a * -(x[self.start_node][0] ** 2)
            + a * -(x[self.terminal_node][self.steps] ** 2)
            + a * h_p
            + a * h_v
            + h_w
            + a * h_w_invalid
        )

        hamiltonian.set_mapping({var_names[i]: i for i in range(len(var_names))})

        if not hamiltonian.Q:
            return zeros((self.n * self.k, self.n * self.k))

        qubo = qv.utils.qubo_to_matrix(Q=hamiltonian.to_qubo(), array=True)

        return qubo
