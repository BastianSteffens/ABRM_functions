"""
The :code:`pyswarms.backend.multi_swarms` module implements
swarm for multiple-objective PSO based on Pareto Front
"""

from .multi_swarms import *
from .pareto_front import ParetoFront

__all__ = ["multi_swarms", "pareto_front"]
