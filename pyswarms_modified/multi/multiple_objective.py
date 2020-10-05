# -*- coding: utf-8 -*-

r"""
A Multiobjective Particle Swarm Optimisation algorithm.
It takes a set of candidate solutions, and tries to find 
the Pareto front of solution using a position-velocity update method. 
The position update can be defined as:
.. math::
   x_{i}(t+1) = x_{i}(t) + v_{i}(t+1)
Where the position at the current timestep :math:`t` is updated using
the computed velocity at :math:`t+1`. Furthermore, the velocity update
is defined as:
.. math::
   v_{ij}(t + 1) = m * v_{ij}(t) + c_{1}r_{1j}(t)[y_{ij}(t) − x_{ij}(t)]
                   + c_{2}r_{2j}(t)[\hat{y}_{j}(t) − x_{ij}(t)]
Here, :math:`c1` and :math:`c2` are the cognitive and social parameters
respectively. They control the particle's behavior given two choices: (1) to
follow its *personal best* or (2) follow the swarm's *global best* position.
Global best for this algorithm is surrogated, by randomly picking a solution
from the Pareto front according to their fitness.
Overall, this dictates if the swarm is explorative or exploitative in nature.
In addition, a parameter :math:`w` controls the inertia of the swarm's
movement.
An example usage is as follows:
.. code-block:: python
    import pyswarms as ps
    from pyswarms.utils.functions import single_obj as fx
    # Set-up hyperparameters
    options = {'c1': 0.5, 'c2': 0.3, 'w':0.9}
    # Call instance of GlobalBestPSO
    optimizer = ps.single.GlobalBestPSO(n_particles=10, dimensions=2,
                                        options=options)
    # Perform optimization
    stats = optimizer.optimize(fx.sphere, iters=100)
This algorithm was adapted from the work of C.A.C. Coello et al. in 
Multiobjective Particle Swarm Optimization [ITEC2004]_.
.. [ITEC2004] C. A. C. Coello, G. T. Pulido and M. S. Lechuga, 
    "Handling multiple objectives with particle swarm optimization,"
    IEEE Transactions on Evolutionary Computation, vol. 8, no. 3, 
    pp. 256-279, June 2004.
"""

import logging
import numpy as np
from time import sleep

from ..base import SwarmOptimizer
from ..utils import Reporter
from ..backend.handlers import BoundaryHandler, VelocityHandler
from ..backend.generators import create_multi_swarm
from ..backend.operators import compute_velocity, compute_position

class MOPSO(SwarmOptimizer):
    def __init__(
        self,
        n_particles,
        dimensions,
        options,
        bounds=None,
        bh_strategy="periodic",
        velocity_clamp=None,
        vh_strategy="unmodified",
        center=1.00,
        ftol=-np.inf,
        init_pos=None,
    ):
        """Initialize the swarm
        Attributes
        ----------
        n_particles : int
            number of particles in the swarm.
        dimensions : int
            number of dimensions in the space.
        options : dict with keys :code:`{'c1', 'c2', 'w'}`
            a dictionary containing the parameters for the specific
            optimization technique.
                * c1 : float
                    cognitive parameter
                * c2 : float
                    social parameter
                * w : float
                    inertia parameter
                * obj_bounds : list(float) of shape (2, obj_dimension)
                    bounds of objective functions
                * obj_dimensions : int
                    number of objective functions to be used
                * grid_size : int
                    number of cells to be used in a grid per dimension
                * grid_weight_coef : float
                    coefficient to be used for weitghted random picking of solutions,
                    must be >1
        bounds : tuple of :code:`np.ndarray` (default is :code:`None`)
            a tuple of size 2 where the first entry is the minimum bound
            while the second entry is the maximum bound. Each array must
            be of shape :code:`(dimensions,)`.
        bh_strategy : String
            a strategy for the handling of out-of-bounds particles.
        velocity_clamp : tuple (default is :code:`None`)
            a tuple of size 2 where the first entry is the minimum velocity
            and the second entry is the maximum velocity. It
            sets the limits for velocity clamping.
        vh_strategy : String
            a strategy for the handling of the velocity of out-of-bounds particles.
        center : list (default is :code:`None`)
            an array of size :code:`dimensions`
        ftol : float
            relative error in objective_func(best_pos) acceptable for
            convergence
        init_pos : :code:`numpy.ndarray` (default is :code:`None`)
            option to explicitly set the particles' initial positions. Set to
            :code:`None` if you wish to generate the particles randomly.
        """
        super(MOPSO, self).__init__(
            n_particles=n_particles,
            dimensions=dimensions,
            options=options,
            bounds=bounds,
            velocity_clamp=velocity_clamp,
            center=center,
            ftol=ftol,
            init_pos=init_pos,
        )

        # Initialize logger
        self.rep = Reporter(logger=logging.getLogger(__name__))
        # Initialize the resettable attributes
        self.reset()
        self.bh = BoundaryHandler(strategy=bh_strategy)
        self.vh = VelocityHandler(strategy=vh_strategy)
        self.name = __name__
        
    def optimize(self, objective_func, iters, fast=False, **kwargs):
        """Optimize the swarm for a number of iterations
        Performs the optimization to evaluate the objective
        function :code:`f` for a number of iterations :code:`iter.`
        Parameters
        ----------
        objective_func : function 
            objective functions to be evaluated, must return a tuple of costs
        iters : int
            number of iterations
        fast : bool (default is False)
            if True, time.sleep is not executed
        kwargs : dict
            arguments for the objective function
        Returns
        -------
        list(tuple)
            the final pareto front, in a form of a list of tuples in a form of 
            (position, costs)
        """
        self.rep.log("Obj. func. args: {}".format(kwargs), lvl=logging.DEBUG)
        self.rep.log(
            "Optimize for {} iters with {}".format(iters, self.options),
            lvl=logging.INFO,
        )
        
        self.swarm.current_cost = objective_func(self.swarm.position, **kwargs)
        self.swarm.update_archive()
        self.swarm.pbest_pos = self.swarm.position
        self.swarm.pbest_cost = self.swarm.current_cost
        for _ in self.rep.pbar(iters, self.name):
            if not fast:
                sleep(0.01)
            # fmt: off
            self.swarm.current_cost = objective_func(self.swarm.position, **kwargs)
            self.swarm.update_personal_best()
            self.swarm.update_archive()
            self.swarm.best_pos, self.swarm.best_cost = self.swarm.generate_global_best()
            # fmt: on
            self.rep.hook(best_cost=self.swarm.archive.get_best_cost(), mean_cost=self.swarm.archive.aggregate(np.mean))
            # Save to history 
            hist = self.ToHistory(
                # TODO: best of the swarm
                best_cost=self.swarm.best_cost,
                mean_pbest_cost=np.mean(self.swarm.pbest_cost),
                mean_neighbor_cost=self.swarm.best_cost,
                position=self.swarm.position,
                velocity=self.swarm.velocity,
            )
            self._populate_history(hist)
            # Perform velocity and position updates
            self.swarm.velocity = compute_velocity(
                self.swarm, self.velocity_clamp, self.vh, self.bounds
            )
            self.swarm.position = compute_position(
                self.swarm, self.bounds, self.bh
            )
        front = self.swarm.archive.get_front()
        # Write report in log and return final cost and position
        self.rep.log(
            "Optimization finished | front size: {}".format(
                len(front)
            ),
            lvl=logging.INFO,
        )
        return front
    
    def reset(self):
        """Reset the attributes of the optimizer
        All variables/atributes that will be re-initialized when this
        method is defined here. Note that this method
        can be called twice: (1) during initialization, and (2) when
        this is called from an instance.
        It is good practice to keep the number of resettable
        attributes at a minimum. This is to prevent spamming the same
        object instance with various swarm definitions.
        Normally, swarm definitions are as atomic as possible, where
        each type of swarm is contained in its own instance. Thus, the
        following attributes are the only ones recommended to be
        resettable:
        * Swarm position matrix (self.pos)
        * Velocity matrix (self.pos)
        * Best scores and positions (gbest_cost, gbest_pos, etc.)
        Otherwise, consider using positional arguments.
        """
        # Initialize history lists
        self.cost_history = []
        self.mean_pbest_history = []
        self.mean_neighbor_history = []
        self.pos_history = []
        self.velocity_history = []
    
        # Initialize the swarm
        self.swarm = create_multi_swarm( # To be changed 
            n_particles=self.n_particles,
            dimensions=self.dimensions,
            bounds=self.bounds,
            center=self.center,
            init_pos=self.init_pos,
            clamp=self.velocity_clamp,
            options=self.options,
        )