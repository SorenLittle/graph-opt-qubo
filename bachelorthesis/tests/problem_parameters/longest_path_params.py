"""Valid params for Longest Path problem"""
from hypothesis import assume
from hypothesis.strategies import fixed_dictionaries, floats, integers, composite
from hypothesis_networkx import graph_builder
from networkx import Graph


@composite
def longest_path_params(draw):
    """Generates valid params"""

    edge_data = fixed_dictionaries(
        {
            "weight": floats(
                allow_nan=False, allow_infinity=False, max_value=1000.0, min_value=0.0
            )
        }
    )

    graph: Graph = draw(
        graph_builder(graph_type=Graph, edge_data=edge_data, min_nodes=2, max_nodes=8)
    )
    edge_weights = [graph.get_edge_data(u, v).get("weight") for (u, v) in graph.edges]
    assume(0.0 not in edge_weights)

    steps: int = draw(integers(min_value=1, max_value=graph.order() - 1))
    start_node: int = draw(integers(min_value=0, max_value=graph.order() - 1))
    terminal_node: int = draw(integers(min_value=0, max_value=graph.order() - 1))
    assume(start_node != terminal_node)

    return {
        "steps": steps,
        "start_node": start_node,
        "terminal_node": terminal_node,
        "graph": graph,
    }
