"""Valid params for Graph Coloring problem"""
from hypothesis.strategies import composite, fixed_dictionaries, floats
from hypothesis_networkx import graph_builder
from networkx import Graph


@composite
def traveling_salesperson_params(draw):
    """Generates valid params"""

    edge_data = fixed_dictionaries(
        {
            "weight": floats(
                allow_nan=False,
                allow_infinity=False,
                max_value=1000.0,
                min_value=1.0,
            )
        }
    )

    graph: Graph = draw(
        graph_builder(graph_type=Graph, edge_data=edge_data, min_nodes=2, max_nodes=8)
    )

    return {
        "graph": graph,
    }
