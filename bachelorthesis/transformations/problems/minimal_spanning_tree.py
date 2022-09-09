from typing import List, Union

from networkx import Graph, DiGraph
import numpy as np
from numpy import zeros
from numpy.typing import NDArray
import qubovert as qv

from bachelorthesis.transformations.problem import Problem


class MinimalSpanningTreeDegreeConstraint(Problem):
    def __init__(self, graph: Union[Graph, DiGraph], delta: int):
        """The Minimal Spanning Tree Problem

        Parameters
        ----------
        graph
            A networkx graph of the problem instance

        delta
            The maximal degree constraint
        """

        self.graph: Union[Graph, DiGraph] = graph
        self.delta: int = delta

    def gen_qubo(self) -> NDArray:
        if not self.graph.is_directed():
            graph: DiGraph = self.graph.to_directed()
        else:
            graph: DiGraph = self.graph

        n_vertex: int = graph.number_of_nodes()
        n_half: int = int(np.ceil(n_vertex / 2))

        M = int(np.floor(np.lograph(max_degree) + 1))

        weights = list(nx.get_edge_attributes(graph, "weight").values())
        if A is None:
            diff = max(weights) - min(weights)
            A = diff * (B / ba_ratio)

        # Initialize variable names
        y_name = {e: f"y({e})" for e in graph.edges}
        x_name = {v: [f"x({v})({i})" for i in range(n_half + 1)] for v in graph.nodes}
        x_name2 = {e: [f"x2({e})({i})" for i in range(n_half)] for e in graph.edges}
        z_name = {v: [f"z({v})({i})" for i in range(M)] for v in graph.nodes}

        # Initialize binary variables
        y_vars = {e: qv.QUBO.create_var(y_name[e]) for e in graph.edges}
        x_vars = {
            v: [qv.QUBO.create_var(x_name[v][i]) for i in range(n_half + 1)]
            for v in graph.nodes
        }
        x_vars2 = {
            e: [qv.QUBO.create_var(x_name2[e][i]) for i in range(n_half)]
            for e in graph.edges
        }
        z_vars = {
            v: [qv.QUBO.create_var(z_name[v][i]) for i in range(M)] for v in graph.nodes
        }

        hamiltonian = qv.QUBO()

        # First Term
        hamiltonian += (1 - sum(x_vars[v][0] for v in graph.nodes)) ** 2

        # Second Term
        hamiltonian += sum(
            (1 - sum(x_vars[v][i] for i in range(n_half + 1))) ** 2 for v in graph.nodes
        )

        # Third Term
        hamiltonian += sum(
            (
                y_vars[u, v]
                - sum(x_vars2[u, v][i] + x_vars2[v, u][i] for i in range(n_half))
            )
            ** 2
            for u, v in graph.edges
        )

        # Fourth Term
        hamiltonian += sum(
            (x_vars[v][i + 1] - sum(x_vars2[e][i] for e in graph.in_edges(v))) ** 2
            for v in graph.nodes
            for i in range(n_half)
        )

        # Fith Term
        count_val = [2 ** i for i in range(M - 1)] + [max_degree + 1 - 2 ** (M - 1)]
        hamiltonian += sum(
            (
                sum(z_vars[v][i] * count_val[i] for i in range(M))
                - sum(
                    x_vars2[u, w][i] + x_vars2[w, u][i]
                    for i in range(n_half)
                    for u, w in graph.edges(v)
                )
            )
            ** 2
            for v in graph.nodes
        )

        # Sixth Term
        hamiltonian += sum(
            x_vars2[u, v][i] * (2 - x_vars[u][i] - x_vars[v][i + 1])
            for i in range(n_half)
            for u, v in graph.edges
        )

        hamiltonian *= A

        # Optimization Objective
        hamiltonian += B * sum(
            x_vars2[u, v][i] * (data["weight"] - min(weights))
            for u, v, data in graph.edges.data()
            for i in range(n_half)
        )

        # Names for mapping
        all_names = [y_name[e] for e in graph.edges]
        all_names += [x_name[v][i] for i in range(n_half + 1) for v in graph.nodes]
        all_names += [x_name2[e][i] for i in range(n_half) for e in graph.edges]
        all_names += [z_name[v][i] for i in range(M) for v in graph.nodes]

        # Mapping to QUBO
        mapping = {v: i for i, v in enumerate(all_names)}
        hamiltonian.set_mapping(mapping)

        # # Make sure qv uses all qubits
        hamiltonian += sum(y_vars[e] for e in graph.edges)
        hamiltonian += sum(x_vars[v][i] for i in range(n_half + 1) for v in graph.nodes)
        hamiltonian += sum(x_vars2[e][i] for i in range(n_half) for e in graph.edges)
        hamiltonian += sum(z_vars[v][i] for i in range(M) for v in graph.nodes)

        # Generate QUBO
        qubo = qv.utils.qubo_to_matrix(hamiltonian.to_qubo().Q, array=True)

        return qubo
