"""Tests that GraphOptimization generates correct qubos"""
from typing import Union

from hypothesis import example, given, settings, note
from hypothesis_networkx import graph_builder
from networkx import Graph
from numpy import allclose, set_printoptions

from bachelorthesis.graph_optimization import GraphOptimization
from bachelorthesis.transformations import (
    CliqueCover,
    LongestPath,
    GraphColoring,
    TravelingSalesperson,
    HamiltonianCycle,
    MaxCut,
    MaxClique,
    MaximumFlow,
    MinimumVertexCover,
    MaximumIndependentSet,
)
from bachelorthesis.tests.problem_parameters import (
    longest_path_params,
    graph_coloring_params,
    traveling_salesperson_params,
    maximum_flow_params,
)

example_graph = Graph(
    [
        (0, 1, {"weight": 3}),
        (0, 2, {"weight": 3}),
        (1, 3, {"weight": 3}),
        (2, 3, {"weight": 3}),
    ]
)


class TestGraphOptimization:
    @example({"graph": example_graph, "colors": 2})
    @given(graph_coloring_params())
    @settings(deadline=None)
    def test_clique_cover(self, params):
        """Test GraphOptimization for Clique Cover"""
        set_printoptions(linewidth=1000)
        graph: Graph = params["graph"]
        note(f"graph: ({{{graph.nodes}}}, {{{{{graph.edges(data=True)}}})")

        real = CliqueCover(**params).gen_qubo()

        g_opt = GraphOptimization(graph=graph)

        clique_cover_constraints = {
            "diagonal": -1,
            "one_node_many_positions": 2,
            "non_edges": 2,
        }

        ours = g_opt.generate_qubo(
            positions=params["colors"], **clique_cover_constraints
        )

        note("real:")
        note(real)  # noqa
        note("ours:")
        note(ours)  # noqa
        note("difference:")
        note(real - ours)

        assert allclose(real, ours) == True

    @example({"graph": example_graph, "steps": 3, "start_node": 0, "terminal_node": 1})
    @given(longest_path_params())
    @settings(deadline=None)
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
            "invalid_traversal": scaling_constant,
            "invalid_traversal_self": scaling_constant,
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
    @settings(deadline=None)
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

    @example({"graph": example_graph})
    @given(traveling_salesperson_params())
    @settings(deadline=None)
    def test_traveling_salesperson(self, params):
        """Test GraphOptimization for Traveling Salesperson"""
        set_printoptions(linewidth=1000)

        graph: Graph = params["graph"]
        note(f"graph: ({{{graph.nodes}}}, {{{{{graph.edges(data=True)}}})")

        real = TravelingSalesperson(**params).gen_qubo()

        g_opt = GraphOptimization(graph=graph)

        b: int = 1
        edge_weights = [
            graph.get_edge_data(edge[0], edge[1]).get("weight") for edge in graph.edges
        ]
        a: int = b * max(edge_weights, default=0) + 1

        traveling_salesperson_constraints = {
            "double_count_edges": True,
            "double_count_edges_cycles": True,
            "diagonal": -2 * a,
            "one_node_many_positions": 2 * a,
            "one_position_many_nodes": 2 * a,
            "invalid_traversal": a,
            "invalid_traversal_cycles": a,
            "edge_weights_factor": b,
            "edge_weights_cycles_factor": b,
        }

        ours = g_opt.generate_qubo(
            positions=graph.order(), **traveling_salesperson_constraints
        )

        note("real:")
        note(real)  # noqa
        note("ours:")
        note(ours)  # noqa
        note("difference:")
        note(real - ours)

        assert allclose(real, ours) == True

    @example({"graph": example_graph})
    @given(traveling_salesperson_params())
    @settings(deadline=None)
    def test_hamiltonian_cycle(self, params):
        """Test GraphOptimization for Hamiltonian Cycle"""
        set_printoptions(linewidth=1000)

        graph: Graph = params["graph"]
        note(f"graph: ({{{graph.nodes}}}, {{{{{graph.edges(data=True)}}})")

        real = HamiltonianCycle(**params).gen_qubo()

        g_opt = GraphOptimization(graph=graph)

        a: int = 1

        hamiltonian_cycle_constraints = {
            "double_count_edges": True,
            "double_count_edges_cycles": True,
            "diagonal": -2 * a,
            "one_node_many_positions": 2 * a,
            "one_position_many_nodes": 2 * a,
            "invalid_traversal": a,
            "invalid_traversal_cycles": a,
        }

        ours = g_opt.generate_qubo(
            positions=graph.order(), **hamiltonian_cycle_constraints
        )

        note("real:")
        note(real)  # noqa
        note("ours:")
        note(ours)  # noqa
        note("difference:")
        note(real - ours)

        assert allclose(real, ours) == True

    @example({"graph": example_graph})
    @given(
        graph=graph_builder(graph_type=Graph, min_nodes=4, max_nodes=40, min_edges=2)
    )
    @settings(deadline=None)
    def test_max_clique(self, graph: Union[dict, Graph]):  # give returns dict :shrug:
        """Test GraphOptimization for Max Clique"""
        set_printoptions(linewidth=1000)

        try:
            graph: Graph = Graph(graph.get("graph"))
        except AttributeError:
            graph: Graph = Graph(graph)
        note(f"graph: ({{{graph.nodes}}}, {{{{{graph.edges(data=True)}}})")

        real = MaxClique(graph=graph).gen_qubo()

        g_opt = GraphOptimization(graph=graph)

        a: int = 1
        b: int = 2

        max_clique_constraints = {
            "diagonal": -a,
            "non_edges": b,
        }

        ours = g_opt.generate_qubo(**max_clique_constraints)

        note("real:")
        note(real)  # noqa
        note("ours:")
        note(ours)  # noqa
        note("difference:")
        note(real - ours)

        assert allclose(real, ours) == True

    @example({"graph": example_graph})
    @given(
        graph=graph_builder(graph_type=Graph, min_nodes=4, max_nodes=40, min_edges=2)
    )
    @settings(deadline=None)
    def test_maximum_independent_set(
        self, graph: Union[dict, Graph]
    ):  # give returns dict :shrug:
        """Test GraphOptimization for Maximum Independent Set"""
        set_printoptions(linewidth=1000)

        try:
            graph: Graph = Graph(graph.get("graph"))
        except AttributeError:
            graph: Graph = Graph(graph)
        note(f"graph: ({{{graph.nodes}}}, {{{{{graph.edges(data=True)}}})")

        real = MaximumIndependentSet(graph=graph).gen_qubo()

        g_opt = GraphOptimization(graph=graph)

        a: int = 1
        b: int = 2

        maximum_independent_set_constraints = {
            "diagonal": -a,
            "edges": b,
        }

        ours = g_opt.generate_qubo(**maximum_independent_set_constraints)

        note("real:")
        note(real)  # noqa
        note("ours:")
        note(ours)  # noqa
        note("difference:")
        note(real - ours)

        assert allclose(real, ours) == True

    @example({"graph": example_graph})
    @given(
        graph=graph_builder(graph_type=Graph, min_nodes=4, max_nodes=40, min_edges=2)
    )
    @settings(deadline=None)
    def test_max_cut(self, graph: Union[dict, Graph]):  # give returns dict :shrug:
        """Test GraphOptimization for Max Cut"""
        set_printoptions(linewidth=1000)

        try:
            graph: Graph = Graph(graph.get("graph"))
        except AttributeError:
            graph: Graph = Graph(graph)
        note(f"graph: ({{{graph.nodes}}}, {{{{{graph.edges(data=True)}}})")

        real = MaxCut(graph=graph).gen_qubo()

        g_opt = GraphOptimization(graph=graph)

        a: int = 1

        max_cut_constraints = {"nodes_with_edges": -a, "edges": 2 * a}

        ours = g_opt.generate_qubo(**max_cut_constraints)

        note("real:")
        note(real)  # noqa
        note("ours:")
        note(ours)  # noqa
        note("difference:")
        note(real - ours)

        assert allclose(real, ours) == True

    @example({"graph": example_graph})
    @given(
        graph=graph_builder(graph_type=Graph, min_nodes=4, max_nodes=40, min_edges=2)
    )
    @settings(deadline=None)
    def test_minimum_vertex_cover(self, graph: Union[dict, Graph]):
        """Test GraphOptimization for Minimum Vertex Cover"""
        set_printoptions(linewidth=1000)

        try:
            graph: Graph = Graph(graph.get("graph"))
        except AttributeError:
            graph: Graph = Graph(graph)
        note(f"graph: ({{{graph.nodes}}}, {{{{{graph.edges(data=True)}}})")

        real = MinimumVertexCover(graph=graph).gen_qubo()

        g_opt = GraphOptimization(graph=graph)

        a: int = 1

        min_vertex_cover_constraints = {
            "diagonal": a,
            "nodes_with_edges": -a,
            "edges": a,
        }

        ours = g_opt.generate_qubo(**min_vertex_cover_constraints)

        note("real:")
        note(real)  # noqa
        note("ours:")
        note(ours)  # noqa
        note("difference:")
        note(real - ours)

        assert allclose(real, ours) == True

    # @example({"graph": example_graph})
    # @given(maximum_flow_params())
    # @settings(deadline=None)
    # def test_maximum_flow(self, params):
    #     """Test GraphOptimization for Maximum Flow"""
    #     set_printoptions(linewidth=1000)
    #
    #     graph: Graph = params["graph"]
    #     note(f"graph: ({{{graph.nodes}}}, {{{{{graph.edges(data=True)}}})")
    #
    #     real = MaximumFlow(**params).gen_qubo()
    #
    #     g_opt = GraphOptimization(graph=graph)
    #
    #     a: int = 1
    #
    #     hamiltonian_cycle_constraints = {
    #
    #     }
    #
    #     ours = g_opt.generate_qubo(
    #         positions=graph.order(), **hamiltonian_cycle_constraints
    #     )
    #
    #     note("real:")
    #     note(real)  # noqa
    #     note("ours:")
    #     note(ours)  # noqa
    #     note("difference:")
    #     note(real - ours)
    #
    #     assert allclose(real, ours) == True
