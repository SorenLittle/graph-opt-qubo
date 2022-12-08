"""Quantum Annealer from D-Wave"""
import json
from pathlib import Path

from uqo.Problem import Qubo
from uqo.client.config import Config
from uqo.examples.examples import ping as ping_


def get_uqo_config():
    base_path = Path(__file__).parent.resolve()
    private_key = base_path / "credentials/private_keys/client.key_secret"

    # generate private key if it doesn't exist already
    if not private_key.exists():
        from uqo.generate_certificates import generate_certificates

        generate_certificates(base_path / Path("credentials"))

    # get secret token from credentials
    with open(base_path / Path("credentials/uqo_token.json"), "r") as token_file:
        data = json.load(token_file)
        token = data.get("token")

    # initialize config data
    config_data = {
        "method": "token",
        "endpoint": "uq.mobile.ifi.lmu.de:30000",
        "credentials": token,
        "private_key_file": private_key,
    }

    return Config(**config_data)


def solve(qubo, **kwargs):
    """Solve a qubo with quantum annealing"""

    problem = (
        Qubo(get_uqo_config(), qubo).with_platform("dwave").with_solver("Advantage_system4.1")
    )

    result = problem.solve(times=kwargs.get("repeats", 1)).sampleset.first
    sample = [int(binary) for _, binary in result[0].items()]
    energy = result[1]
    return sample, energy


def find_embedding(qubo):
    """Find qubo embedding for quantum annealer"""

    problem = (
        Qubo(get_uqo_config(), qubo).with_platform("dwave").with_solver("Advantage_system4.1")
    )
    return problem.find_pegasus_embedding()

    # TODO: THIS IS THE LINE FOR DRAWING
    # problem.draw_pegasus_embedding("embedding_pegasus.pdf")
