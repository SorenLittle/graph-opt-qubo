"""Simulated annealing solver from Dwave"""

import neal


def solve(qubo, **kwargs):
    """Solve a qubo with simulated annealing"""

    sampler = neal.SimulatedAnnealingSampler()
    result = sampler.sample_qubo(qubo, **kwargs).record
    return result["sample"], result["energy"]
