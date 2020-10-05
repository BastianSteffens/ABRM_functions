from .pareto_front import ParetoFront, is_dominant

"""
MultiSwarm Class Backend

This module implements a MultiSwarm class that holds various attributes in
the swarm such as position, velocity, options, etc. You can use this
as input to MOPSO. Utilises Pareto front.
"""

# Import modules
import random
import numpy as np
from attr import attrib, attrs
from attr.validators import instance_of
from ..swarms import Swarm

@attrs
class MultiSwarm(Swarm):
    """A MultiSwarm Class

    This class offers a swarm that can be used in Multiobjective PSO. 
    It contains various attributes that are commonly-used in most swarm implementations,
    including a Pareto front that is required for MOPSO

    To initialize this class, **simply supply values for the position and
    velocity matrix**. The other attributes are automatically filled. If you want to
    initialize random values, take a look at:

    * :func:`pyswarms.backend.generators.generate_swarm`: for generating positions randomly.
    * :func:`pyswarms.backend.generators.generate_velocity`: for generating velocities randomly.

    If your swarm requires additional parameters (say c1, c2, and w in gbest
    PSO), simply pass them to the :code:`options` dictionary. Some options are compalsory, see below.

    As an example, say we want to create a swarm by generating particles
    randomly. We can use the helper methods above to do our job:

    .. code-block:: python

        import pyswarms.backend as P
        from pyswarms.backend.multi_swarms.multi_swarms import MultiSwarm

        # Let's generate a 10-particle swarm with 10 dimensions
        init_positions = P.generate_swarm(n_particles=10, dimensions=10)
        init_velocities = P.generate_velocity(n_particles=10, dimensions=10)
        # Say, particle behavior is governed by parameters `foo` and `bar`
        my_options = {'foo': 0.4, 'bar': 0.6}
        # Initialize the swarm
        my_swarm = MultiSwarm(position=init_positions, velocity=init_velocities, options=my_options)

    From there, you can now use all the methods in :mod:`pyswarms.backend`.
    Of course, the process above has been abstracted by the method
    :func:`pyswarms.backend.generators.create_multi_swarm` so you don't have to
    write the whole thing down.

    Note that, since the MultiSwarm uses Pareto dominance, best_cost and best_pos attributes
    are surrogated and do not represent the true best cost and position.

    Compalsory options
    ------------------
    obj_dimensions : int
        number of objective functions to be evaluated

    grid_size : int
        number of cells to be used in a grid per dimension

    obj_bounds : list(list)
        bounds of objective functions

    grid_weight_coef : float
        coefficient to be used for weitghted random picking of solutions
        must be >1

    Attributes
    ----------
    position : numpy.ndarray
        position-matrix at a given timestep of shape :code:`(n_particles, dimensions)`
    velocity : numpy.ndarray
        velocity-matrix at a given timestep of shape :code:`(n_particles, dimensions)`
    n_particles : int (default is :code:`position.shape[0]`)
        number of particles in a swarm.
    dimensions : int (default is :code:`position.shape[1]`)
        number of dimensions in a swarm.
    options : dict (default is empty dictionary)
        various options that govern a swarm's behavior.
    pbest_pos : numpy.ndarray (default is :code:`None`)
        personal best positions of each particle of shape :code:`(n_particles, dimensions)`
    best_pos : numpy.ndarray (default is empty array)
        best position found by the swarm of shape :code:`(dimensions, )` for the
        :code:`Star`topology and :code:`(dimensions, particles)` for the other
        topologies
    pbest_cost : numpy.ndarray (default is empty array)
        personal best costs of each particle of shape :code:`(n_particles, )`
    best_cost : float (default is :code:`np.inf`)
        best cost found by the swarm
    current_cost : numpy.ndarray (default is empty array)
        the current cost found by the swarm of shape :code:`(n_particles, dimensions)`
    """
    archive = attrib(type=ParetoFront, default=None)
    
    # Surrogate for utility functions
    best_cost = attrib(
        type=np.ndarray,
        default=np.array([]),
        validator=instance_of(np.ndarray),
    )
    
    def update_archive(self):
        self.archive.insert_all(zip(self.position, self.current_cost))
    
    def update_personal_best(self):
        for i in range(self.n_particles):
            if(is_dominant(self.pbest_cost[i],self.current_cost[i])):
                self.pbest_cost[i] = self.current_cost[i]
                self.pbest_pos[i] = self.position[i]
                
    def generate_global_best(self):
        return self.archive.get_random_item()
