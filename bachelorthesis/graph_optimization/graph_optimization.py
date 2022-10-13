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
        # self.nodes: List[int] = list(self.graph.nodes)

    def generate_qubo(
        self,
        positions: int = 1,
        start_node: int = None,
        terminal_node: int = None,
        **kwargs,
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

        # populate the qubo with constraints
        self._add_structure_constraints(
            qubo=qubo,
            positions=positions,
            start_node=start_node,
            terminal_node=terminal_node,
            **kwargs,
        )

        self._add_edge_constraints(
            qubo=qubo,
            positions=positions,
            start_node=start_node,
            terminal_node=terminal_node,
            **kwargs,
        )

        return qubo

    def _add_structure_constraints(
        self,
        qubo: NDArray,
        positions: int = 1,
        start_node: int = None,
        terminal_node: int = None,
        start_node_score: float = None,
        terminal_node_score: float = None,
        diagonal: float = None,
        nodes_with_edges: float = None,
        one_node_many_positions: float = None,
        one_position_many_nodes: float = None,
        **kwargs,
    ):
        """Handles constraints related to graph structure (not edges)"""

        # START NODE
        if (start_node is not None) and start_node_score:
            start_idx = start_node * positions + 0
            qubo[start_idx][start_idx] += start_node_score

        # TERMINAL NODE
        if (terminal_node is not None) and terminal_node_score:
            terminal_idx = terminal_node * positions + positions - 1
            qubo[terminal_idx][terminal_idx] += terminal_node_score

        # ITERATIVE CONSTRAINTS
        nodes = list(self.graph.nodes)
        for node_idx in range(len(nodes)):

            connected_nodes_lt: List = [
                nodes.index(node)
                for origin, node in self.graph.edges()
                if nodes.index(origin) == node_idx  # TODO: this would be the place for the boolean flag
            ]
            connected_nodes_gt: List = [
                nodes.index(node)
                for node, origin in self.graph.edges()
                if nodes.index(origin) == node_idx
            ]
            connected_nodes = connected_nodes_lt + connected_nodes_gt

            for position in range(positions):
                # CALCULATE INDEX
                # simple    : With positions = 1, we iterate over the range loop once
                #             and position takes the value 0 -> correct indexing
                # positional: The below formula calculates the correct index of a node
                #             in the context of a positional optimization

                idx: int = node_idx * positions + position

                # DIAGONAL
                if diagonal:
                    qubo[idx][idx] += diagonal

                # NODES WITH EDGES
                if nodes_with_edges:
                    for node2_idx in connected_nodes:
                        qubo[idx][idx] += nodes_with_edges

                # ONE NODE MANY POSITIONS
                if one_node_many_positions:
                    remaining_positions = range(position + 1, positions)
                    for position2 in remaining_positions:
                        idx2: int = node_idx * positions + position2
                        qubo[idx][idx2] += one_node_many_positions

                # ONE POSITION MANY NODES
                if one_position_many_nodes:
                    remaining_nodes = range(
                        node_idx + 1, len(nodes)
                    )  # TODO: why don't we look at nodes before this one?
                    for node2_idx in remaining_nodes:
                        idx2: int = node2_idx * positions + position
                        qubo[idx][idx2] += one_position_many_nodes

    def _add_edge_constraints(
        self,
        qubo: NDArray,
        positions: int = 1,
        double_count_edges: bool = False,
        double_count_edges_cycles: bool = False,
        edges: float = None,  # TODO: edge_cycles, edge_self, edge_w_self_factor?
        edge_weights_factor: float = None,
        edge_weights_cycles_factor: float = None,
        non_edges: float = None,
        non_edges_self: float = None,
        non_edges_cycles: float = None,
        **kwargs,
    ):
        """Handles constraints related to the edges in the graph"""

        nodes = list(self.graph.nodes)
        for node_idx in range(len(nodes)):

            connected_nodes_lt: List[int] = [
                nodes.index(node)
                for origin, node in self.graph.edges()
                if nodes.index(origin) == node_idx
            ]
            connected_nodes_gt: List[int] = [
                nodes.index(node)
                for node, origin in self.graph.edges()
                if nodes.index(origin) == node_idx
            ]
            connected_nodes = connected_nodes_lt + connected_nodes_gt

            for position in range(positions):
                # CALCULATE INDEX
                # simple    : With positions = 1, we iterate over the range loop once
                #             and position takes the value 0 -> correct indexing
                # positional: The below formula calculates the correct index of a node
                #             in the context of a positional optimization

                idx: int = node_idx * positions + position

                # EDGES
                if edges:
                    for node2_idx in connected_nodes:
                        idx2: int = node2_idx * positions + position
                        if node_idx <= node2_idx:
                            qubo[idx][idx2] += edges
                        elif node2_idx < node_idx and double_count_edges:
                            qubo[idx2][idx] += edges

                # EDGE WEIGHTS + CYCLES
                if edge_weights_factor:
                    for node2 in connected_nodes:
                        # TODO: node2 >= test is "only" for undirected -> do directed graphs break
                        if position < positions - 1:
                            idx2: int = node2 * positions + position + 1

                            edge_data = self.graph.get_edge_data(node, node2)
                            if weight := edge_data.get("weight"):
                                if node < node2:
                                    qubo[idx][idx2] += weight * edge_weights_factor
                                if node2 <= node and double_count_edges:
                                    qubo[idx2][idx] += weight * edge_weights_factor

                        # EDGE WEIGHTS CYCLES
                        elif (
                            (not (node2 <= node) or double_count_edges)  # DOUBLE COUNT
                            and position == positions - 1
                            and edge_weights_cycles_factor
                        ):
                            idx2: int = node2 * positions

                            edge_data = self.graph.get_edge_data(node, node2)
                            if weight := edge_data.get("weight"):
                                if node < node2:
                                    qubo[idx][idx2] += (
                                        weight * edge_weights_cycles_factor
                                    )
                                if double_count_edges and node2 <= node:
                                    qubo[idx2][idx] += (
                                        weight * edge_weights_cycles_factor
                                    )

                # NON EDGES + CYCLES
                if non_edges:
                    unconnected_nodes = [
                        n for n in self.nodes if n not in connected_nodes and n != node
                    ]

                    for node2 in unconnected_nodes:
                        if positions > 1 and position < positions - 1:
                            idx2: int = node2 * positions + position + 1
                            if node < node2:
                                qubo[idx][idx2] += non_edges
                            elif node2 <= node and double_count_edges:
                                qubo[idx2][idx] += non_edges

                        # TODO: necessary?
                        elif positions == 1:
                            if node < node2:
                                idx2: int = node2
                                qubo[idx][idx2] += non_edges

                        # NON EDGES CYCLE

                        elif (
                            (not (node2 <= node) or double_count_edges_cycles)
                            and positions > 1
                            and position == positions - 1
                            and non_edges_cycles
                        ):
                            idx2: int = node2 * positions
                            if node < node2:
                                qubo[idx][idx2] += non_edges_cycles
                            if node2 <= node and double_count_edges_cycles:
                                qubo[idx2][idx] += non_edges_cycles

                # NON EDGES SELF
                if non_edges_self:
                    if position < positions - 1:
                        idx2: int = idx + 1
                        qubo[idx][idx2] += non_edges_self
