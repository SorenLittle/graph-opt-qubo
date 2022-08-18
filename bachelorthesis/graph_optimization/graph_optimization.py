"""Class for creating graph based optimizations"""

from typing import Union

from networkx import Graph, DiGraph, is_directed
from numpy import zeros
from numpy.typing import NDArray


class GraphOptimization:
    """Generalized Graph Based Optimizations"""

    def __init__(self, graph: Union[Graph, DiGraph], positions: int = 1):
        self.graph: Union[Graph, DiGraph] = graph
        self.is_directed: bool = is_directed(graph)
        self.positions: int = positions
        self.n: int = graph.order()

        self.qubo: NDArray = zeros((self.n * self.positions, self.n * self.positions))

    def
