from pprint import pprint
from typing import List

import qubovert as qv
from networkx import DiGraph
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

        Notes
        -----
        The start and terminal nodes in the network must be nodes 0 and n-1
        respectively such that n is the number of nodes in graph
        """

        self.graph: DiGraph = graph

        # In order to make our output domain binary, we need to create a new
        # graph using our input graph. Every edge in the input graph has a
        # weight corresponding to its maximal possible flow

    def gen_qubo(self) -> NDArray:
        n = self.graph.order()  # number of nodes, see |V|
        e = 0  # number of edges in original graph
        e_prime = 0  # number of edges in transformed graph (binary), see e`
        e_prime_mapping = {}
        edges = []  # list of actual edges in original graph (u, v)

        for i, edge in enumerate(self.graph.edges()):

            weight = self.graph.get_edge_data(edge[0], edge[1]).get('weight')
            for flow in range(weight):
                # e_prime_mapping[i] =
                e_prime += 1

            edges.append(edge)

        print(e_prime, e_prime_mapping)

        # create list of variable names in order to ensure correct mapping
        var_names: List[str] = [
            f'x({i})({flow})'
            for i, edge in enumerate(edges)
            for flow in range(
                self.graph.get_edge_data(
                    edge[0],
                    edge[1]
                ).get('weight')
            )
        ]
        print(var_names)

        # indexes = []
        # for i, edge in enumerate(edges):
        #     for flow in range(
        #             self.graph.get_edge_data(edge[0], edge[1]).get('weight')
        #     ):
        #         indexes.append(i + flow)
        #         if flow >= 1:
        #             i += 1
        #
        # print(indexes)

        # create list of QUBO variables for use in the hamiltonian
        x: List[List[qv.QUBO]] = [
            [
                qv.QUBO.create_var(var_names[var_names.index(edge_flow)])
                # for flow in range(
                # self.graph.get_edge_data(
                #     edge[0],
                #     edge[1]
                # ).get('weight')
                for edge_flow in [e for e in var_names if e[:4] == f'x({i})']
                # for flow in edges if flow[:4] == f'x({i})'
            ]
            for i, edge in enumerate(edges)
        ]
        pprint(x)

        a: float = .5  # TODO: define this

        # hamiltonian = qv.QUBO()
        # hamiltonian += sum(
        #     (
        #             sum(
        #                 x[e_prime_mapping[i]][flow]
        #                 for i, edge in enumerate(edges)
        #                 if edge[1] == node
        #                 for flow in range(
        #                     self.graph.get_edge_data(
        #                         edge[0],
        #                         edge[1]
        #                     ).get('weight')
        #                 )
        #             ) -
        #             sum(
        #                 x[e_prime_mapping[i]][flow]
        #                 for i, edge in enumerate(edges)
        #                 if edge[0] == node
        #                 for flow in range(
        #                     self.graph.get_edge_data(
        #                         edge[0],
        #                         edge[1]
        #                     ).get('weight')
        #                 )
        #             )
        #     ) ** 2
        #     for node in range(1, n - 1)
        # ) - a * sum(
        #     x[i][flow] ** 2
        #     for i, edge in enumerate(edges)
        #     if edge[0] == 0
        #     for flow in range(
        #         self.graph.get_edge_data(
        #             edge[0],
        #             edge[1]
        #         ).get('weight')
        #     )
        # )

        # for node in range(1, n):
        #     for
        # i, edge in enumerate(edges):
        # if edge[1] == node:
        #     for
        # flow in range(
        #     self.graph.get_edge_data(
        #         edge[0],
        #         edge[1]
        #     ).get('weight')):
        # print('penalty', '\nnode', node, '\ni', i, '\nedge', edge,
        #       '\nflow', flow, '\n')
        # if edge[0] == node:
        #     for
        # flow in range(
        #     self.graph.get_edge_data(
        #         edge[0],
        #         edge[1]
        #     ).get('weight')):
        # print('reward', '\nnode', node, '\ni', i, '\nedge', edge,
        #       '\nflow', flow, '\n')

        # hamiltonian += sum(
        #     (
        #             sum(
        #                 x[edge_idx][flow]
        #                 for edge_idx in range(e)
        #                 if edges[edge_idx][1] == i
        #                 for flow in range(
        #                     self.graph.get_edge_data(
        #                         edges[edge_idx][0],
        #                         edges[edge_idx][1]).get(
        #                         'weight')
        #                 )
        #             ) -
        #             sum(
        #                 x[edge_idx][flow]
        #                 for edge_idx in range(e)
        #                 if edges[edge_idx][0] == i
        #                 for flow in range(
        #                     self.graph.get_edge_data(
        #                         edges[edge_idx][0],
        #                         edges[edge_idx][1]).get(
        #                         'weight')
        #                 )
        #             )
        #     ) ** 2
        #     for i in range(1, n - 1)
        # ) - a * sum(
        #     x[edge_idx][flow]
        #     for edge_idx in range(e)
        #     if edges[edge_idx][0] == 0
        #     for flow in range(
        #         self.graph.get_edge_data(
        #             edges[edge_idx][0],
        #             edges[edge_idx][1]).get(
        #             'weight')
        #     )
        # )
        #
        # hamiltonian += sum(
        #     (
        #             sum(
        #                 x[edge_prime][flow]
        #                 for edge_prime in range(e_prime)
        #                 if edges[e_prime_mapping.get(edge_prime)][1] == i
        #                 for flow in range(
        #                     self.graph.get_edge_data(
        #                         edges[e_prime_mapping.get(edge_prime)][0],
        #                         edges[e_prime_mapping.get(edge_prime)][1]).get(
        #                         'weight')
        #                 )
        #             ) -
        #             sum(
        #                 x[edge_prime][flow]
        #                 for edge_prime in range(e_prime)
        #                 if edges[e_prime_mapping.get(edge_prime)][0] == i
        #                 for flow in range(
        #                     self.graph.get_edge_data(
        #                         edges[e_prime_mapping.get(edge_prime)][0],
        #                         edges[e_prime_mapping.get(edge_prime)][1]).get(
        #                         'weight')
        #                 )
        #             )
        #     ) ** 2
        #     for i in range(1, n - 1)
        # ) - a * sum(
        #     x[edge_prime][flow]
        #     for edge_prime in range(e_prime)
        #     if edges[e_prime_mapping.get(edge_prime)][0] == 0
        #     for flow in range(
        #         self.graph.get_edge_data(
        #             edges[e_prime_mapping.get(edge_prime)][0],
        #             edges[e_prime_mapping.get(edge_prime)][1]).get(
        #             'weight')
        #     )
        # )

        # hamiltonian.set_mapping(
        #     {var_names[i]: i for i in range(len(var_names))}
        # )
        #
        # pprint(hamiltonian.Q)
        #
        # # return a correctly sized matrix if the problem is empty
        # if not hamiltonian.Q:
        #     return zeros((len(var_names), len(var_names)))
        #
        # qubo = qv.utils.qubo_to_matrix(
        #     Q=hamiltonian.to_qubo(), array=True
        # )
        #
        # print(qubo)
        #
        # return qubo
