"""Tests that GraphOptimization generates correct qubos"""
from hypothesis import example, given, settings, note
from networkx import Graph
from numpy import allclose, set_printoptions

from bachelorthesis.graph_optimization import GraphOptimization
from bachelorthesis.transformations import LongestPath
from bachelorthesis.tests.problem_parameters import (
    longest_path_params,
    graph_coloring_params,
)
from bachelorthesis.transformations.problems.graph_coloring import GraphColoring

example_graph = Graph(
    [
        (0, 1, {"weight": 8}),
        (0, 2, {"weight": 7}),
        (1, 3, {"weight": 2}),
        (2, 3, {"weight": 8}),
    ]
)


class TestGraphOptimization:
    @example({"graph": example_graph, "steps": 3, "start_node": 0, "terminal_node": 1})
    @given(longest_path_params())
    @settings(deadline=1000)
    def test_longest_path(self, params):
        """Test if GraphOptimization encodes the Longest Path problem properly"""
        set_printoptions(linewidth=1000)

        graph: Graph = params["graph"]

        note(f"graph: ({{{graph.nodes}}}, {{{{{graph.edges(data=True)}}})")

        real = LongestPath(**params).gen_qubo()

        g_opt = GraphOptimization(graph=graph)

        edge_weights = [
            graph.get_edge_data(edge[0], edge[1]).get("weight") for edge in graph.edges
        ]
        scaling_constant: int = params["steps"] * max(edge_weights, default=0)

        longest_path_constraints = {
            "diagonal": -scaling_constant,
            "start_node_score": -scaling_constant,
            "terminal_node_score": -scaling_constant,
            "one_position_many_nodes": 2 * scaling_constant,
            "one_node_many_positions": scaling_constant,
            "edge_weights_factor": -1,
            "non_edges": scaling_constant,
            "non_edges_self": scaling_constant,
        }

        ours = g_opt.generate_qubo(
            positions=params["steps"] + 1,
            start_node=params["start_node"],
            terminal_node=params["terminal_node"],
            **longest_path_constraints,
        )

        note("real:")
        note(real)  # noqa
        note("ours:")
        note(ours)  # noqa
        note("difference:")
        note(real - ours)

        assert allclose(real, ours) == True

    @example({"graph": example_graph, "colors": 2})
    @given(graph_coloring_params())
    @settings(deadline=1000)
    def test_graph_coloring(self, params):
        """Test GraphOptimization for Graph Coloring"""
        set_printoptions(linewidth=1000)

        graph: Graph = params["graph"]
        note(f"graph: ({{{graph.nodes}}}, {{{{{graph.edges(data=True)}}})")

        real = GraphColoring(**params).gen_qubo()

        g_opt = GraphOptimization(graph=graph)
        a = 4

        graph_coloring_constraints = {
            "diagonal": -a,
            "one_node_many_positions": 2 * a,
            "edges": a,
        }
        ours = g_opt.generate_qubo(
            positions=params["colors"], **graph_coloring_constraints
        )

        note("real:")
        note(real)  # noqa
        note("ours:")
        note(ours)  # noqa
        note("difference:")
        note(real - ours)

        assert allclose(real, ours) == True
