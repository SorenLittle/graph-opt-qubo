"""Tabu search solver from Dwave"""

import tabu


def solve(qubo, **kwargs):
    """Solve a qubo with tabu search"""

    sampler = tabu.TabuSampler()
    result = sampler.sample_qubo(qubo, **kwargs).record
    return result["sample"], result["energy"]
