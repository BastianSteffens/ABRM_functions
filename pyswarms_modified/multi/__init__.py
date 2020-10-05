"""
The :mod:`pyswarms.multi` module a
continuous multiple-objective optimization. These require multiple
objective functions that can be optimized in a continuous space.
.. note::
    PSO algorithms scale with the search space. This means that, by
    using larger boundaries, the final results are getting larger
    as well.
.. note::
    Please keep in mind that Python has a biggest float number.
    So using large boundaries in combination with exponentiation or
    multiplication can lead to an :code:`OverflowError`.
"""

from .multiple_objective import MOPSO
from .multiple_objective_BS import MOPSO


__all__ = ["MOPSO"]