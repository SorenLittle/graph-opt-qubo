from typing import List

import qubovert as qv
from networkx import DiGraph
from numpy import zeros
from numpy.typing import NDArray

from transformations.problem import Problem


class MaximumFlow(Problem):
    def __init__(
            self,
            graph: DiGraph
    ):
        """The Maximum Flow Problem

        Parameters
        ----------
        graph
            A networkx graph of the problem instance
        """

        self.graph: DiGraph = graph

        # In order to make our output domain binary, we need to create a new
        # graph using our input graph. Every edge in the input graph has a
        # weight corresponding to its maximal possible flow

    def gen_qubo(self) -> NDArray:
        edges = self.graph.edges()

        # create list of variable names in order to ensure correct mapping
        var_names: List[str] = [
            f'x({u, v})({flow})'
            for u, v in edges
            for flow in range(self.graph.get_edge_data(u, v).get('weight'))
        ]

        # create list of QUBO variables for use in the hamiltonian
        x: List[qv.QUBO] = [
            qv.QUBO.create_var(
                var_names[var]
            )
            for var in range(len(var_names))
        ]

        a: float = .5  # TODO: define this

        hamiltonian = qv.QUBO()
        hamiltonian += sum(
            (
                sum(
                    x[edge][flow]
                    for edge in edges if edge[0] == i
                    for flow in range(
                        self.graph.get_edge_data(edge[0], edge[1]).get('weight')
                    )
                ) - sum(
                    x[edge][flow]
                    for edge in edges if edge[1] == i
                    for flow in range(
                        self.graph.get_edge_data(edge[0], edge[1]).get('weight')
                    )
                ) ** 2
                for i in range(self.graph.order())
            )
        ) - a * sum(
            x[edge][flow] ** 2
            for edge in edges if edge[1] == i
            for flow in range(
                self.graph.get_edge_data(edge[0], edge[1]).get('weight')
            )
        )

        hamiltonian.set_mapping(
            {var_names[i]: i for i in range(len(var_names))}
        )

        # return a correctly sized matrix if the problem is empty
        if not hamiltonian.Q:
            return zeros((len(var_names), len(var_names)))

        qubo = qv.utils.qubo_to_matrix(
            Q=hamiltonian.to_qubo(), array=True
        )

        return qubo


graph = DiGraph()
graph.add_weighted_edges_from(
    [(0, 1, 1), (0, 2, 2), (1, 3, 2), (2, 3, 1)]
)

test = MaximumFlow(
    graph=graph
)

test.gen_qubo()
