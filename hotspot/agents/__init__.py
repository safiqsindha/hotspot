"""Built-in agents. Each implements ``act(observation) -> action``."""

from hotspot.agents.oracle_agent import OracleAgent
from hotspot.agents.random_agent import RandomAgent

__all__ = ["OracleAgent", "RandomAgent"]
