# -*- coding: utf-8 -*-

"""
Particle Swarm Optimization (PSO) toolkit
=========================================
PySwarms is a particle swarm optimization (PSO) toolkit that enables
researchers to test variants of the PSO technique in different contexts.
Users can define their own function, or use one of the benchmark functions
in the library. It is built on top of :code:`numpy` and :code:`scipy`, and
is very extensible to accommodate other PSO variations.
"""

__author__ = """Lester James V. Miranda"""
__email__ = "ljvmiranda@gmail.com"
__version__ = "1.1.0"

from .single import global_best, local_best, general_optimizer,local_best_BS

from .multi import multiple_objective,multiple_objective_BS

from .particles import particle, multi_particle
from .discrete import binary
from .utils.decorators import cost

__all__ = ["global_best", "local_best", "general_optimizer", "multiple_objective", "binary", "cost", "particle", "multi_particle"]
