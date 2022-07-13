from typing import List

from networkx import Graph
from numpy import zeros
from numpy.typing import NDArray
import qubovert as qv

from transformations.problem import Problem


class MinimalSpanningTreeDegreeConstraint(Problem):
    def __init__(
            self,
            graph: Graph,
            delta: int
    ):
        """The Minimal Spanning Tree Problem

        Parameters
        ----------
        graph
            A networkx graph of the problem instance

        delta
            The maximal degree constraint
        """

        self.graph: Graph = graph
        self.delta: int = delta

    def gen_qubo(self) -> NDArray:

        n = self.graph.order()

        # create list of variable names in order to ensure correct mapping
        var_names: List[str] = [
            f'x({node})'
            for node in range(n)
        ]  # TODO: this needs Ergänzung

        # create list of QUBO variables for use in the hamiltonian
        x: List[qv.QUBO] = [
            qv.QUBO.create_var(
                var_names[node]
            )
            for node in range(n)
        ]

        # set scaling constants according to heuristic in paper
        a: float = 1  # TODO
        b: float = 1  # TODO

        # TODO: nothing new past here!!

        # create hamiltonian of the problem
        hamiltonian = qv.QUBO()
        hamiltonian_a = (
            a * (1 - sum(...)) ** 2 +
            a * sum(...) ** 2 +
            a * sum(...) ** 2 +
            a * sum(...) ** 2 +
            a * sum(...) ** 2 +
            a * sum(...)
        )

        hamiltonian_b = b * sum(...)

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
