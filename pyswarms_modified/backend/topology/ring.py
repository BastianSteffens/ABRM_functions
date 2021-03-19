# -*- coding: utf-8 -*-

"""
A Ring Network Topology

This class implements a ring topology. In this topology,
the particles are connected with their k nearest neighbors.
This social behavior is often found in LocalBest PSO
optimizers.
"""

# Import standard library
import logging

# Import modules
import numpy as np
from scipy.spatial import cKDTree
import random
from scipy.sparse.csgraph import connected_components, dijkstra

from .. import operators as ops
from ..handlers import BoundaryHandler, VelocityHandler
from ...utils.reporter import Reporter
from .base import Topology


class Ring(Topology):
    def __init__(self, static=False):
        """Initializes the class

        Parameters
        ----------
        static : bool (Default is :code:`False`)
            a boolean that decides whether the topology
            is static or dynamic
        """
        super(Ring, self).__init__(static)
        self.rep = Reporter(logger=logging.getLogger(__name__))

    def compute_gbest(self, swarm, p, k,neighbours_random = False, **kwargs):
        """Update the global best using a ring-like neighborhood approach

        This uses the cKDTree method from :code:`scipy` to obtain the nearest
        neighbors.

        Parameters
        ----------
        swarm : pyswarms.backend.swarms.Swarm
            a Swarm instance
        p: int {1,2}
            the Minkowski p-norm to use. 1 is the
            sum-of-absolute values (or L1 distance) while 2 is
            the Euclidean (or L2) distance.
        k : int
            number of neighbors to be considered. Must be a
            positive integer less than :code:`n_particles`

        Returns
        -------
        numpy.ndarray
            Best position of shape :code:`(n_dimensions, )`
        float
            Best cost
        """
        try:
            # Check if the topology is static or not and assign neighbors
            if (self.static and self.neighbor_idx is None) or not self.static:
                # Obtain the nearest-neighbors for each particle
                tree = cKDTree(swarm.position)
                _, self.neighbor_idx = tree.query(swarm.position, p=p, k=k)

            # Map the computed costs to the neighbour indices and take the
            # argmin. If k-neighbors is equal to 1, then the swarm acts
            # independently of each other.
            if k == 1:
                # The minimum index is itself, no mapping needed.
                self.neighbor_idx = self.neighbor_idx[:, np.newaxis]
                best_neighbor = np.arange(swarm.n_particles)
            elif neighbours_random == True:
                if (self.static and self.neighbor_idx is None) or not self.static:
                    adj_matrix = self.__compute_neighbors(swarm, k)
                    self.neighbor_idx = np.array([adj_matrix[i].nonzero()[0] for i in range(swarm.n_particles)])
                idx_min = np.array([swarm.pbest_cost[self.neighbor_idx[i]].argmin() for i in range(len(self.neighbor_idx))])
                best_neighbor = np.array([self.neighbor_idx[i][idx_min[i]] for i in range(len(self.neighbor_idx))]).astype(int)

                # # Obtain best cost and position
                # best_cost = np.min(swarm.pbest_cost[best_neighbor])
                # best_pos = swarm.pbest_pos[best_neighbor]

                # except AttributeError:
                #     self.rep.logger.exception(
                #         "Please pass a Swarm class. You passed {}".format(type(swarm))
                #     )
                #     raise
                # else:
                #     return (best_pos, best_cost)

            else:
                idx_min = swarm.pbest_cost[self.neighbor_idx].argmin(axis=1)
                best_neighbor = self.neighbor_idx[
                    np.arange(len(self.neighbor_idx)), idx_min
                ]
            # Obtain best cost and position
            best_cost = np.min(swarm.pbest_cost[best_neighbor])
            best_pos = swarm.pbest_pos[best_neighbor]
        except AttributeError:
            self.rep.logger.exception(
                "Please pass a Swarm class. You passed {}".format(type(swarm))
            )
            raise
        else:
            return (best_pos, best_cost)

    def compute_velocity(
        self,
        swarm,
        clamp=None,
        vh=VelocityHandler(strategy="unmodified"),
        bounds=None,
    ):
        """Compute the velocity matrix

        This method updates the velocity matrix using the best and current
        positions of the swarm. The velocity matrix is computed using the
        cognitive and social terms of the swarm.

        A sample usage can be seen with the following:

        .. code-block :: python

            import pyswarms.backend as P
            from pyswarms.backend.swarm import Swarm
            from pyswarms.backend.handlers import VelocityHandler
            from pyswarms.backend.topology import Ring

            my_swarm = P.create_swarm(n_particles, dimensions)
            my_topology = Ring(static=False)
            my_vh = VelocityHandler(strategy="invert")

            for i in range(iters):
                # Inside the for-loop
                my_swarm.velocity = my_topology.update_velocity(my_swarm, clamp, my_vh,
                bounds)

        Parameters
        ----------
        swarm : pyswarms.backend.swarms.Swarm
            a Swarm instance
        clamp : tuple of floats (default is :code:`None`)
            a tuple of size 2 where the first entry is the minimum velocity
            and the second entry is the maximum velocity. It
            sets the limits for velocity clamping.
        vh : pyswarms.backend.handlers.VelocityHandler
            a VelocityHandler instance
        bounds : tuple of :code:`np.ndarray` or list (default is :code:`None`)
            a tuple of size 2 where the first entry is the minimum bound while
            the second entry is the maximum bound. Each array must be of shape
            :code:`(dimensions,)`.

        Returns
        -------
        numpy.ndarray
            Updated velocity matrix
        """
        return ops.compute_velocity(swarm, clamp, vh, bounds)

    def compute_velocity_entropy(
        self,
        swarm,
        clamp=None,
        vh=VelocityHandler(strategy="unmodified"),
        bounds=None,
    ):
        """Compute the velocity matrix

        This method updates the velocity matrix using the best and current
        positions of the swarm. The velocity matrix is computed using the
        cognitive and social terms of the swarm.

        A sample usage can be seen with the following:

        .. code-block :: python

            import pyswarms.backend as P
            from pyswarms.backend.swarm import Swarm
            from pyswarms.backend.handlers import VelocityHandler
            from pyswarms.backend.topology import Ring

            my_swarm = P.create_swarm(n_particles, dimensions)
            my_topology = Ring(static=False)
            my_vh = VelocityHandler(strategy="invert")

            for i in range(iters):
                # Inside the for-loop
                my_swarm.velocity = my_topology.update_velocity(my_swarm, clamp, my_vh,
                bounds)

        Parameters
        ----------
        swarm : pyswarms.backend.swarms.Swarm
            a Swarm instance
        clamp : tuple of floats (default is :code:`None`)
            a tuple of size 2 where the first entry is the minimum velocity
            and the second entry is the maximum velocity. It
            sets the limits for velocity clamping.
        vh : pyswarms.backend.handlers.VelocityHandler
            a VelocityHandler instance
        bounds : tuple of :code:`np.ndarray` or list (default is :code:`None`)
            a tuple of size 2 where the first entry is the minimum bound while
            the second entry is the maximum bound. Each array must be of shape
            :code:`(dimensions,)`.

        Returns
        -------
        numpy.ndarray
            Updated velocity matrix
        """
        return ops.compute_velocity_entropy(swarm, clamp, vh, bounds)

    def compute_position(
        self, swarm, bounds=None, bh=BoundaryHandler(strategy="periodic")
    ):
        """Update the position matrix

        This method updates the position matrix given the current position and
        the velocity. If bounded, it waives updating the position.

        Parameters
        ----------
        swarm : pyswarms.backend.swarms.Swarm
            a Swarm instance
        bounds : tuple of :code:`np.ndarray` or list (default is :code:`None`)
            a tuple of size 2 where the first entry is the minimum bound while
            the second entry is the maximum bound. Each array must be of shape
            :code:`(dimensions,)`.
        bh : pyswarms.backend.handlers.BoundaryHandler
            a BoundaryHandler instance

        Returns
        -------
        numpy.ndarray
            New position-matrix
        """
        return ops.compute_position(swarm, bounds, bh)

    
    def __compute_neighbors(self, swarm, k):
        """Helper method to compute the adjacency matrix of the topology

        This method computes the adjacency matrix of the topology using
        the randomized algorithm proposed in [TSWJ2013]. The resulting
        topology is a connected graph. This is achieved by creating three
        matrices:

            * adj_matrix :  The adjacency matrix of the generated graph.
                            It's initialized as an identity matrix to
                            make sure that every particle has itself as
                            a neighbour. This matrix is the return
                            value of the method.
            * neighbor_matrix : The matrix of randomly generated neighbors.
                                This matrix is a matrix of shape
                                :code:`(swarm.n_particles, k)`:
                                with randomly generated elements. It is used
                                to create connections in the :code:`adj_matrix`.
            * dist_matrix : The distance matrix computed with Dijkstra's
                            algorithm. It is used to determine where the
                            graph needs edges to change it to a connected
                            graph.

        Parameters
        ----------
        swarm : pyswarms.backend.swarms.Swarm
            a Swarm instance
        k : int
            number of neighbors to be considered. Must be a
            positive integer less than :code:`n_particles-1`

        Returns
        -------
        numpy.ndarray
            Adjacency matrix of the topology
        """

        adj_matrix = np.identity(swarm.n_particles, dtype=int)

        neighbor_matrix = np.array(
            [
                np.random.choice(
                    # Exclude i from the array
                    np.setdiff1d(np.arange(swarm.n_particles), np.array([i])),
                    k,
                    replace=False,
                )
                for i in range(swarm.n_particles)
            ]
        )

        # Set random elements to one using the neighbor matrix
        adj_matrix[
            np.arange(swarm.n_particles).reshape(swarm.n_particles, 1),
            neighbor_matrix,
        ] = 1
        adj_matrix[
            neighbor_matrix,
            np.arange(swarm.n_particles).reshape(swarm.n_particles, 1),
        ] = 1

        dist_matrix = dijkstra(
            adj_matrix,
            directed=False,
            return_predecessors=False,
            unweighted=True,
        )

        # Generate connected graph.
        while (
            connected_components(
                adj_matrix, directed=False, return_labels=False
            )
            != 1
        ):
            for i, j in itertools.product(range(swarm.n_particles), repeat=2):
                if dist_matrix[i][j] == np.inf:
                    adj_matrix[i][j] = 1

        return adj_matrix
