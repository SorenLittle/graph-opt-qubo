"""Repeated simulated annealing solver from Dwave"""

import neal


def solve(qubo, **kwargs):
    """Solve a qubo with repeated simulated annealing"""

    sampler = neal.SimulatedAnnealingSampler()
    state = None

    for _ in range(kwargs["num_repeats"]):
        if not state:
            result = sampler.sample_qubo(qubo, **kwargs).record
        else:
            result = sampler.sample_qubo(
                qubo, initial_states=state["sample"], **kwargs
            ).record
        state = result

    return state["sample"], state["energy"]
