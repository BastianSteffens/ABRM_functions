# -*- coding: utf-8 -*-

"""
Swarm Operation Backend

This module abstracts various operations in the swarm such as updating
the personal best, finding neighbors, etc. You can use these methods
to specify how the swarm will behave.
"""

# Import standard library
import logging

# Import modules
import numpy as np

from ..utils.reporter import Reporter
from .handlers import BoundaryHandler, VelocityHandler
from functools import partial

### BS ###
import multiprocessing as mp

import numpy as np
import pandas as pd
import subprocess
import time
import os
from os import path
import datetime
from datetime import date
import pickle
import bz2
import _pickle as cPickle
from scipy import interpolate
import shutil
import re
from pathlib import Path
import glob
from pyentrp import entropy as ent
from skimage.util.shape import view_as_windows


rep = Reporter(logger=logging.getLogger(__name__))


def compute_pbest(swarm):
    """Update the personal best score of a swarm instance

    You can use this method to update your personal best positions.

    .. code-block:: python

        import pyswarms.backend as P
        from pyswarms.backend.swarms import Swarm

        my_swarm = P.create_swarm(n_particles, dimensions)

        # Inside the for-loop...
        for i in range(iters):
            # It updates the swarm internally
            my_swarm.pbest_pos, my_swarm.pbest_cost = P.update_pbest(my_swarm)

    It updates your :code:`current_pbest` with the personal bests acquired by
    comparing the (1) cost of the current positions and the (2) personal
    bests your swarm has attained.

    If the cost of the current position is less than the cost of the personal
    best, then the current position replaces the previous personal best
    position.

    Parameters
    ----------
    swarm : pyswarms.backend.swarm.Swarm
        a Swarm instance

    Returns
    -------
    numpy.ndarray
        New personal best positions of shape :code:`(n_particles, n_dimensions)`
    numpy.ndarray
        New personal best costs of shape :code:`(n_particles,)`
    """
    try:
        # Infer dimensions from positions
        dimensions = swarm.dimensions
        # Create a 1-D and 2-D mask based from comparisons
        mask_cost = swarm.current_cost < swarm.pbest_cost
        mask_pos = np.repeat(mask_cost[:, np.newaxis], dimensions, axis=1)
        # Apply masks
        new_pbest_pos = np.where(~mask_pos, swarm.pbest_pos, swarm.position)
        new_pbest_cost = np.where(
            ~mask_cost, swarm.pbest_cost, swarm.current_cost
        )
    except AttributeError:
        rep.logger.exception(
            "Please pass a Swarm class. You passed {}".format(type(swarm))
        )
        raise
    else:
        return (new_pbest_pos, new_pbest_cost)


def compute_velocity(swarm, clamp, vh, bounds=None):
    """Update the velocity matrix

    This method updates the velocity matrix using the best and current
    positions of the swarm. The velocity matrix is computed using the
    cognitive and social terms of the swarm. The velocity is handled
    by a :code:`VelocityHandler`.

    A sample usage can be seen with the following:

    .. code-block :: python

        import pyswarms.backend as P
        from pyswarms.swarms.backend import Swarm, VelocityHandler

        my_swarm = P.create_swarm(n_particles, dimensions)
        my_vh = VelocityHandler(strategy="invert")

        for i in range(iters):
            # Inside the for-loop
            my_swarm.velocity = compute_velocity(my_swarm, clamp, my_vh, bounds)

    Parameters
    ----------
    swarm : pyswarms.backend.swarms.Swarm
        a Swarm instance
    clamp : tuple of floats, optional
        a tuple of size 2 where the first entry is the minimum velocity
        and the second entry is the maximum velocity. It
        sets the limits for velocity clamping.
    vh : pyswarms.backend.handlers.VelocityHandler
        a VelocityHandler object with a specified handling strategy.
        For further information see :mod:`pyswarms.backend.handlers`.
    bounds : tuple of numpy.ndarray or list, optional
        a tuple of size 2 where the first entry is the minimum bound while
        the second entry is the maximum bound. Each array must be of shape
        :code:`(dimensions,)`.

    Returns
    -------
    numpy.ndarray
        Updated velocity matrix
    """
    try:
        # Prepare parameters
        swarm_size = swarm.position.shape
        c1 = swarm.options["c1"]
        c2 = swarm.options["c2"]
        w = swarm.options["w"]
        direction = swarm.options["direction"]

        # Compute for cognitive and social terms
        cognitive = (
            c1
            * np.random.uniform(0, 1, swarm_size)
            * (swarm.pbest_pos - swarm.position)
        )
        social = (
            c2
            * np.random.uniform(0, 1, swarm_size)
            * (swarm.best_pos - swarm.position)
        )

        ### BS ###
        # implement ARPSO
    # Function setDirection
    # if (dir > 0 && diversity < dLow) dir = -1;
    # if (dir < 0 && diversity > dHigh) dir = 1;
    # setDirection(); // new!
    # updateVelocity();
    # newPosition();
    # assignFitness();
    # calculateDiversity(); // new!
    # what infulueces entropy: n_paramters,n_particles, rough
    # try and calculate the max_entropy possible for given pso config
    # print(ent.shannon_entropy(np.arange(10000)))
    #
        # entropy = load_iteration_entropy_tof()

        # if entropy >= 1300 and entropy <= 2500:
        #     temp_velocity = (w * swarm.velocity) + cognitive + social
        #     print("entropy within bounds:{}".format(entropy))
        #     print(temp_velocity)
        # else:
        #     temp_velocity = (w * swarm.velocity) + (-cognitive + -social)
        #     print("entropy out of bounds:{}".format(entropy))
        #     print(temp_velocity)
    #  if (dir > 0 && diversity < dLow) dir = -1;
    #  if (dir < 0 && diversity > dHigh) dir = 1;

        # Compute temp velocity (subject to clamping if possible)
        # temp_velocity = compute_ARPSO_velocity(w,swarm,cognitive,social,direction)
        temp_velocity = (w * swarm.velocity) + cognitive + social
        updated_velocity = vh(
            temp_velocity, clamp, position=swarm.position, bounds=bounds
        )

        ###BS###
        #adjust w with damping factor put this after calcluations are done.
        w_damp = swarm.options["d"]
        w = w*w_damp
        #update w in options dict. this should slow down veloctiy over time
        swarm.options["w"] = w

    except AttributeError:
        rep.logger.exception(
            "Please pass a Swarm class. You passed {}".format(type(swarm))
        )
        raise
    except KeyError:
        rep.logger.exception("Missing keyword in swarm.options")
        raise
    else:
        return updated_velocity


def compute_position(swarm, bounds, bh):
    """Update the position matrix

    This method updates the position matrix given the current position and the
    velocity. If bounded, the positions are handled by a
    :code:`BoundaryHandler` instance

    .. code-block :: python

        import pyswarms.backend as P
        from pyswarms.swarms.backend import Swarm, VelocityHandler

        my_swarm = P.create_swarm(n_particles, dimensions)
        my_bh = BoundaryHandler(strategy="intermediate")

        for i in range(iters):
            # Inside the for-loop
            my_swarm.position = compute_position(my_swarm, bounds, my_bh)

    Parameters
    ----------
    swarm : pyswarms.backend.swarms.Swarm
        a Swarm instance
    bounds : tuple of numpy.ndarray or list, optional
        a tuple of size 2 where the first entry is the minimum bound while
        the second entry is the maximum bound. Each array must be of shape
        :code:`(dimensions,)`.
    bh : pyswarms.backend.handlers.BoundaryHandler
        a BoundaryHandler object with a specified handling strategy
        For further information see :mod:`pyswarms.backend.handlers`.

    Returns
    -------
    numpy.ndarray
        New position-matrix
    """
    try:
        temp_position = swarm.position.copy()
        temp_position += swarm.velocity

        if bounds is not None:
            temp_position = bh(temp_position, bounds)

        position = temp_position
    except AttributeError:
        rep.logger.exception(
            "Please pass a Swarm class. You passed {}".format(type(swarm))
        )
        raise
    else:
        return position


def compute_objective_function(swarm, objective_func,setup,iteration, pool=None, **kwargs):

    """Evaluate particles using the objective function

    This method evaluates each particle in the swarm according to the objective
    function passed.

    If a pool is passed, then the evaluation of the particles is done in
    parallel using multiple processes.

    Parameters
    ----------
    swarm : pyswarms.backend.swarms.Swarm
        a Swarm instance
    objective_func : function
        objective function to be evaluated
    setup: dict
        dictionary that contains all important information for modelling settings ### BS ###
    pool: multiprocessing.Pool
        multiprocessing.Pool to be used for parallel particle evaluation
    kwargs : dict
        arguments for the objective function

    Returns
    -------
    numpy.ndarray
        Cost-matrix for the given swarm

    data_to_save :
        data that need to be saved
    """
    if pool is None:
        all_particles = objective_func(swarm,setup,iteration)
        # swarm_misfit,swarm_performance = all_particles.particle_iterator()
        all_particles.particle_iterator()
        print("misfit:{}".format(all_particles.misfit_swarm))

        return all_particles.misfit_swarm,all_particles.swarm_performance
    # thoights on this: when doing single processing then I should call particle.particle_iterator that returns same stuff as the swarm iterator
    # if I do it in multiplrocessing then I should do call particle.calcluate_particle(i) and not feed the objective function into the pool thing, but that method of that class.

    else:
        # p = mp.pool(pooler)
        particle_array = np.arange(0,swarm.n_particles)
        all_particles = objective_func(swarm,setup,iteration)
        particle_performance = pool.map(all_particles.calculate_particle_parallel,[particle_no for particle_no in particle_array])
        pool.close()
        swarm_performance = np.concatenate(particle_performance)
        print(swarm_performance)
        swarm_misfit = 1
        # # print(shape(particle_misfit))
        # results = pool.map(
        #     partial(objective_func, **kwargs),
        #     np.array_split(swarm.position, pool._processes),
        # )
        return swarm_misfit, swarm_performance

def compute_ARPSO_velocity(w,swarm,cognitive,social,direction):
    # calculate + entropy of larger blocks (of mean or sum or median) (upscaled) 
    # inbetween iterations and then of each model take the similar block, and
    # calculate their entropy. use that as a penalty function. want that to be high.
    # Function setDirection
    # if (dir > 0 && diversity < dLow) dir = -1;
    # if (dir < 0 && diversity > dHigh) dir = 1;
    # setDirection(); // new!
    # updateVelocity();
    # newPosition();
    # assignFitness();
    # calculateDiversity(); // new!
    # what infulueces entropy: n_paramters,n_particles, rough
    # try and calculate the max_entropy possible for given pso config
    # print(ent.shannon_entropy(np.arange(10000)))
    #

    # loading in settings that I set up on init_ABRM.py for this run
    pickle_file = "C:/AgentBased_RM/Output/variable_settings.pickle"
    with open(pickle_file, "rb") as f:
        setup = pickle.load(f)

    folder_path = setup["folder_path"]
    n_particles = setup["n_particles"]
    n_grid_cells = 200

    # filepath
    output_file_partilce_values_converted = "/swarm_particle_values_converted_all_iter.csv"
    output_file_partilce_values_converted_file_path = folder_path + output_file_partilce_values_converted
 
    # check if file exists
    if os.path.exists(output_file_partilce_values_converted_file_path):

        #load file
        swarm_particle_values_converted_all_iter = pd.read_csv(output_file_partilce_values_converted_file_path)

        # check latest value for tof_upscaled_entropy_swarm
        swarm_upscaled_entropy = swarm_particle_values_converted_all_iter["tof_upscaled_entropy_swarm"].iloc[-1]
        
        # check iteration the pso is at
        n_iterations = swarm_particle_values_converted_all_iter["iteration"].iloc[-1] + 1

        # maximum possible entropy
        max_entropy = n_grid_cells * ent.shannon_entropy(np.arange(n_particles * n_iterations))
        print("maxentropy:")
        print(max_entropy)
        print("swarm_entropy")
        print(swarm_upscaled_entropy)
        # acceptable entropy
        max_entropy = max_entropy * 0.85
        min_entropy = max_entropy * 0.75

        if direction > 0 and swarm_upscaled_entropy < min_entropy:
            direction = -1
            print("entropy out of bounds:{}".format(swarm_upscaled_entropy))

        if direction < 0 and swarm_upscaled_entropy > max_entropy:
            direction = 1
            print("entropy within bounds:{}".format(swarm_upscaled_entropy))

        temp_velocity = (w * swarm.velocity) + direction * (cognitive + social)

        swarm.options["direction"] = direction   


    return temp_velocity