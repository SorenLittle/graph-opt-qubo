from typing import List

from networkx import Graph
from numpy import zeros
from numpy.typing import NDArray
import qubovert as qv

from transformations.problem import Problem


class MaximalIndependentSet(Problem):
    def __init__(
            self,
            graph: Graph
    ):
        """The Maximal Independent Set Problem

        Parameters
        ----------
        graph
            A networkx graph of the problem instance
        """

        self.graph: Graph = graph

    def gen_qubo(self) -> NDArray:
        n = self.graph.order()

        # create list of of variable names in order to ensure correct mapping
        var_names: List[str] = [
            f'x({node})'
            for node in range(n)
        ]

        # create list of QUBO variables for use in the hamiltonian
        x: List[qv.QUBO] = [
            qv.QUBO.create_var(
                var_names[node]
            )
            for node in range(n)
        ]

        # create hamiltonian of the problem
        hamiltonian = qv.QUBO()
        hamiltonian += - sum(
            x[i]
            for i in self.graph.nodes
        ) + sum(
            x[i] * x[j]
            for i, j in self.graph.edges
        )

        hamiltonian.set_mapping(
            {var_names[i]: i for i in range(len(var_names))}
        )

        # return a correctly sized matrix if the problem is empty
        if not hamiltonian.Q:
            return zeros((n, n))

        qubo = qv.utils.qubo_to_matrix(
            Q=hamiltonian.to_qubo(), array=True
        )

        return qubo
