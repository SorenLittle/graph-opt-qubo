"""Class for creating graph based optimizations"""

from typing import Union, Dict, List

from networkx import Graph, DiGraph, is_directed
from numpy import zeros
from numpy.typing import NDArray


class GraphOptimization:
    """Generalized Graph Based Optimizations"""

    def __init__(self, graph: Union[Graph, DiGraph]):
        self.graph: Union[Graph, DiGraph] = graph
        self.is_directed: bool = is_directed(graph)
        self.n: int = graph.order()
        self.nodes: List[int] = list(self.graph.nodes)

    def generate_qubo(
        self,
        positions: int = 1,
        start_node: int = None,
        terminal_node: int = None,
        **kwargs
    ) -> NDArray:
        """Generates a QUBO for the Graph Optimization with the specified constraints

        Parameters
        ----------
        positions
        start_node
        terminal_node
        kwargs

        Returns
        -------

        """

        # TODO: raise Error if start/terminal nodes not in the graph

        qubo: NDArray = zeros((self.n * positions, self.n * positions))

        print(kwargs)

        # populate the qubo with constraints
        self._add_structure_constraints(
            qubo=qubo,
            positions=positions,
            start_node=start_node,
            terminal_node=terminal_node,
            **kwargs
        )

        return qubo

    def _add_structure_constraints(
        self,
        qubo: NDArray,
        positions: int,
        start_node: int,
        terminal_node: int,
        start_node_score: float = None,
        terminal_node_score: float = None,
        diagonal: float = None,
        nodes_with_edges: float = None,
        one_node_many_positions: float = None,
        one_position_many_nodes: float = None,
        **kwargs
    ):
        """Handles constraints related to graph structure (not edges)"""

        # START NODE
        if start_node and start_node_score:
            start_idx = start_node * positions + 0
            qubo[start_idx][start_idx] += start_node_score

        # TERMINAL NODE
        if terminal_node and terminal_node_score:
            terminal_idx = terminal_node * positions + positions - 1
            qubo[terminal_idx][terminal_idx] += terminal_node_score

        # ITERATIVE CONSTRAINTS
        for node in self.nodes:
            for position in range(positions):
                # CALCULATE INDEX
                # simple    : With positions = 1, we iterate over the range loop once
                #             and position takes the value 0 -> correct indexing
                # positional: The below formula calculates the correct index of a node
                #             in the context of a positional optimization

                idx: int = node * positions + position

                # DIAGONAL
                if diagonal:
                    qubo[idx][idx] += diagonal

                # NODES WITH EDGES
                if nodes_with_edges:
                    qubo[idx][idx] += nodes_with_edges

                # ONE NODE MANY POSITIONS
                if one_node_many_positions:
                    remaining_positions = range(position + 1, positions)
                    for position2 in remaining_positions:
                        idx2: int = node * positions + position2
                        qubo[idx][idx2] += one_node_many_positions

                # ONE POSITION MANY NODES
                if one_position_many_nodes:
                    remaining_nodes = range(
                        node + 1, len(self.nodes)
                    )  # TODO: why don't we look at nodes before this one?
                    for node2 in remaining_nodes:
                        idx2: int = node2 * positions + position
                        qubo[idx][idx2] += one_position_many_nodes
