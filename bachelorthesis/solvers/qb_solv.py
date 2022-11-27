"""QBSolv solver from Dwave"""

from dwave_qbsolv import QBSolv


def solve(qubo, **kwargs):
    """Solve a qubo with QBSolv"""

    sampler = QBSolv()
    result = sampler.sample_qubo(qubo, **kwargs).record
    return result["sample"], result["energy"]
