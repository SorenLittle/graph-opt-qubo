from abc import abstractmethod

from numpy.typing import NDArray


class Problem:
    @abstractmethod
    def gen_qubo(self) -> NDArray:
        """Generates a qubo representation of the given Problem"""

        raise NotImplementedError
