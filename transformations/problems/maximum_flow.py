from networkx import Graph
from numpy.typing import NDArray

from transformations.problem import Problem


class MaximumFlow(Problem):
    def __init__(
            self,
            graph: Graph
    ):
        """The Maximum Flow Problem

        Parameters
        ----------
        graph
            A networkx graph of the problem instance
        """

        self.graph: Graph = graph

        # In order to make our output domain binary, we need to create a new
        # graph using our input graph. Every edge in the input graph has a
        # weight corresponding to its maximal possible flow

    def gen_qubo(self) -> NDArray:
        # TODO: try to implement without needing to create new graph

        ...
