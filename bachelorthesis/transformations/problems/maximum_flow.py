from typing import List

import qubovert as qv
from networkx import DiGraph
from numpy import zeros
from numpy.typing import NDArray

from bachelorthesis.transformations.problem import Problem


class MaximumFlow(Problem):
    def __init__(self, graph: DiGraph):
        """The Maximum Flow Problem

        Parameters
        ----------
        graph
            A networkx graph of the problem instance

        Notes
        -----
        The start and terminal nodes in the network must be nodes 0 and n-1
        respectively such that n is the number of nodes in graph
        """

        self.graph: DiGraph = graph

    def gen_qubo(self) -> NDArray:
        n = self.graph.order()  # number of nodes, see |V|
        e_prime = 0  # number of edges in transformed graph (binary), see e`
        edges = []  # list of actual edges in original graph (u, v)

        # initialize edges and e_prime so that the domain is binary
        for i, edge in enumerate(self.graph.edges()):
            weight = self.graph.get_edge_data(edge[0], edge[1]).get("weight")
            for flow in range(weight):
                e_prime += 1
            edges.append(edge)

        # create list of variable names in order to ensure correct mapping
        var_names: List[str] = [
            f"x({i})({flow})"
            for i, edge in enumerate(edges)
            for flow in range(self.graph.get_edge_data(edge[0], edge[1]).get("weight"))
        ]

        # create list of QUBO variables for use in the hamiltonian
        x: List[List[qv.QUBO]] = [
            [
                qv.QUBO.create_var(var_names[var_names.index(edge_flow)])
                for edge_flow in [e for e in var_names if e[:4] == f"x({i})"]
            ]
            for i, edge in enumerate(edges)
        ]

        # set scaling alpha based on heuristic in paper
        c = 0
        for edge in edges:
            if edge[0] == 0:
                for flow in range(
                    self.graph.get_edge_data(edge[0], edge[1]).get("weight")
                ):
                    c += 1

        a: float = 1 / (c - 0.5)

        # define hamiltonian
        hamiltonian = qv.QUBO()
        hamiltonian += sum(
            (
                sum(
                    x[i][flow]
                    for i, edge in enumerate(edges)
                    if edge[1] == node
                    for flow in range(
                        self.graph.get_edge_data(edge[0], edge[1]).get("weight")
                    )
                )
                - sum(
                    x[i][flow]
                    for i, edge in enumerate(edges)
                    if edge[0] == node
                    for flow in range(
                        self.graph.get_edge_data(edge[0], edge[1]).get("weight")
                    )
                )
            )
            ** 2
            for node in range(1, n - 1)
        ) - a * sum(
            x[i][flow] ** 2
            for i, edge in enumerate(edges)
            if edge[0] == 0
            for flow in range(self.graph.get_edge_data(edge[0], edge[1]).get("weight"))
        )

        hamiltonian.set_mapping({var_names[i]: i for i in range(len(var_names))})

        # return a correctly sized matrix if the problem is empty
        if not hamiltonian.Q:
            return zeros((len(var_names), len(var_names)))

        qubo = qv.utils.qubo_to_matrix(Q=hamiltonian.to_qubo(), array=True)

        return qubo
