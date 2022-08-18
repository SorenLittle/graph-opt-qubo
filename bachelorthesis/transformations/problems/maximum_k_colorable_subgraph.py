import itertools
from typing import List

from networkx import Graph
from numpy import zeros
from numpy.typing import NDArray
import qubovert as qv

from transformations.problem import Problem


class MaximumKColorableSubgraph(Problem):
    def __init__(self, k_colors: int, graph: Graph):
        """The Maximal k-colorable Induced Subgraph Problem

        Parameters
        ----------
        k_colors
            the number of number of colors with which the induced subgraph
            should be k-colorable

        graph
            A networkx graph of the problem instance
        """

        self.k_colors = k_colors
        self.graph: Graph = graph

    def gen_qubo(self) -> NDArray:
        n = self.graph.order()
        k = self.k_colors

        # create list of variable names in order to ensure correct mapping
        var_names: List[str] = [
            f"x({node})({color})"
            for node, color in itertools.product(range(n), range(k))
        ]

        # create list of QUBO variables for use in the hamiltonian
        x: List[List[qv.QUBO]] = [
            [qv.QUBO.create_var(var_names[k * node + color]) for color in range(k)]
            for node in range(n)
        ]

        # create hamiltonian of the problem
        hamiltonian = qv.QUBO()
        hamiltonian += -sum(x[i][r] for i in range(n) for r in range(k))

        # TODO: figure out how to use damn slack variables for inequalities
        #       glover paper has a basic explanation

        # hamiltonian += sum(
        #
        # )

        hamiltonian.set_mapping({var_names[i]: i for i in range(len(var_names))})

        # return a correctly sized matrix if the problem is empty
        if not hamiltonian.Q:
            return zeros((n * k, n * k))

        qubo = qv.utils.qubo_to_matrix(Q=hamiltonian.to_qubo(), array=True)

        return qubo
