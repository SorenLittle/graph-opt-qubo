"""Valid params for Graph Coloring problem"""
from hypothesis.strategies import integers, composite
from hypothesis_networkx import graph_builder
from networkx import Graph


@composite
def graph_coloring_params(draw):
    """Generates valid params"""

    graph: Graph = draw(
        graph_builder(graph_type=Graph, min_nodes=2, max_nodes=8, min_edges=2)
    )

    colors: int = draw(integers(min_value=2, max_value=16))

    return {
        "colors": colors,
        "graph": graph,
    }
