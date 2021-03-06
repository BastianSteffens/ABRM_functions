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
# import multiprocessing as mp

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

def compute_pbest_entropy(swarm):
    """Update the personal best score of a swarm instance

        In this setting there are no historic data--> new pbest is current run
    """
    try:
        new_pbest_pos = swarm.position
        new_pbest_cost = swarm.particle_quality
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

def compute_velocity_entropy(swarm, clamp, vh, bounds=None):
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
            * (swarm.best_pos - swarm.position)
        )
        social = (
            c2
            * np.random.uniform(0, 1, swarm_size)
            * (swarm.global_best_pos - swarm.position)
        )

        ### BS ###
        # implement ARPSO
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

def compute_objective_function_entropy(swarm, objective_func,setup,iteration, pool=None, **kwargs):

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

    ### BS ###
    if pool is None:
        all_particles = objective_func(swarm,setup,iteration)
        all_particles.particle_iterator()
        print("misfit:{}".format(all_particles.misfit_swarm))

        return all_particles.misfit_swarm,all_particles.entropy_contribution_swarm, all_particles.LC_swarm, all_particles.swarm_performance, setup

    else:
        particle_array = np.arange(0,swarm.n_particles)
        all_particles = objective_func(swarm,setup,iteration)
        particle_list = pool.map(all_particles.calculate_particle_parallel,[particle_no for particle_no in particle_array])
        n_particles = setup["n_particles"]
        misfit_swarm = []
        LC_swarm = []
        entropy_contribution_swarm = []
   
        # swarm_performance = pd.DataFrame(columns = ["EV","tD","F","Phi","LC","tof","iteration","particle_no","misfit","entropy_contribution"])
        swarm_performance = pd.DataFrame(columns = ["EV","tD","F","Phi","LC","tof_for","tof_back","tof_combi","prod_part","prod_inj","iteration","particle_no","misfit","entropy_contribution"])

        n_particles = setup["n_particles"]
        for i in range(n_particles):
            if setup["n_voronoi"] > 0:
                setup["assign_voronoi_zone_" +str(i)] = particle_list[i]["assign_voronoi_zone_" +str(i)]
            particle_performance = particle_list[i]["particle_performance"]
            swarm_performance = swarm_performance.append(particle_performance,ignore_index = True)
            misfit_swarm.append(particle_performance[particle_performance.particle_no == i].misfit.unique())
            LC_swarm.append(particle_performance[particle_performance.particle_no == i].LC.unique())
            entropy_contribution_swarm.append(particle_performance[particle_performance.particle_no == i].entropy_contribution.unique())

        misfit_swarm = np.array(misfit_swarm).flatten()
        LC_swarm = np.array(LC_swarm).flatten()
        entropy_contribution_swarm = np.array(entropy_contribution_swarm).flatten()
        print("misfit:{}".format(misfit_swarm))
        
        return misfit_swarm, entropy_contribution_swarm, LC_swarm, swarm_performance, setup

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

    ### BS ###
    if pool is None:
        all_particles = objective_func(swarm,setup,iteration)
        all_particles.particle_iterator()
        print("misfit:{}".format(all_particles.misfit_swarm))

        return all_particles.misfit_swarm, all_particles.LC_swarm, all_particles.swarm_performance, setup

    else:
        particle_array = np.arange(0,swarm.n_particles)
        all_particles = objective_func(swarm,setup,iteration)
        particle_list = pool.map(all_particles.calculate_particle_parallel,[particle_no for particle_no in particle_array])
        n_particles = setup["n_particles"]
        misfit_swarm = []
        LC_swarm = []
   
        # swarm_performance = pd.DataFrame(columns = ["EV","tD","F","Phi","LC","tof","iteration","particle_no","misfit"])
        swarm_performance = pd.DataFrame(columns = ["EV","tD","F","Phi","LC","tof_for","tof_back","tof_combi","prod_part","prod_inj","iteration","particle_no","misfit"])
        particle_performance["tof_for"] = FD_data[5]
        particle_performance["tof_back"] = FD_data[6]
        particle_performance["tof_combi"] = FD_data[7]
        particle_performance["prod_part"] = FD_data[8]
        particle_performance["inj_part"] = FD_data[9]
        n_particles = setup["n_particles"]
        for i in range(n_particles):
            if setup["n_voronoi"] > 0:
                setup["assign_voronoi_zone_" +str(i)] = particle_list[i]["assign_voronoi_zone_" +str(i)]
            particle_performance = particle_list[i]["particle_performance"]
            swarm_performance = swarm_performance.append(particle_performance,ignore_index = True)
            misfit_swarm.append(particle_performance[particle_performance.particle_no == i].misfit.unique())
            LC_swarm.append(particle_performance[particle_performance.particle_no == i].LC.unique())

        misfit_swarm = np.array(misfit_swarm).flatten()
        LC_swarm = np.array(LC_swarm).flatten()
        print("misfit:{}".format(misfit_swarm))
        
        return misfit_swarm, LC_swarm, swarm_performance, setup

def compute_objective_function_multi(swarm, objective_func,setup,iteration, pool=None, **kwargs):

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

    ### BS ###
    if pool is None:
        all_particles_multi = objective_func(swarm,setup,iteration)
        all_particles_multi.particle_iterator_multi()
        print("misfit:{}".format(all_particles_multi.misfit_swarm))

        return all_particles_multi.misfit_swarm, all_particles_multi.LC_swarm, all_particles_multi.swarm_performance, setup

    else:
        particle_array = np.arange(0,swarm.n_particles)
        all_particles_multi = objective_func(swarm,setup,iteration)
        # particle_no = particle_no.item()
        n_particles = setup["n_particles"]
        n_shedules = setup["n_shedules"]
        particle_list = pool.map(all_particles_multi.calculate_particle_parallel_multi  ,[particle_no for particle_no in particle_array])

        misfit_swarm = np.zeros((n_particles,n_shedules))
        LC_swarm = np.zeros((n_particles,n_shedules))
        swarm_performance = pd.DataFrame()#columns = ["EV","tD","F","Phi","LC","tof","iteration","particle_no","misfit"])
        n_particles = setup["n_particles"]
        for i in range(n_particles):
            if setup["n_voronoi"] > 0:
                setup["assign_voronoi_zone_" +str(i)] = particle_list[i]["assign_voronoi_zone_" +str(i)]
            particle_performance = particle_list[i]["particle_performance"]
            swarm_performance = swarm_performance.append(particle_performance,ignore_index = True)
            for shedule_no in range(n_shedules):
                misfit = "misfit_" + str(shedule_no)
                LC = "LC_" + str(shedule_no)
                misfit_swarm[i,shedule_no] = particle_performance[particle_performance.particle_no == i][misfit].unique()
                LC_swarm[i,shedule_no] = particle_performance[particle_performance.particle_no == i][LC].unique()

        print("misfit:{}".format(misfit_swarm))
        
        return misfit_swarm, LC_swarm, swarm_performance, setup

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

def mutate_particle(pos, bounds, iteration, n_iterations, mutationrate):
    """ to prevent mature convergence to a false pareto front, a mutation factor is included that tries to explore 
        with all the particles at the beginning of the search. Then, we decrease rapidly (with respect to the number 
        of iterations) the number of particles that are affected by the mutation operator. Note that our mutation operator
        is applied not only to the particles of the swarm, but also to the range of each design variable of the problem 
        to be solved (using the same variation function). See Coello Coello et al. Handling Multiple Objectives with PSO (2004)."""

    if (np.random.random() < ((1.0 - iteration / n_iterations) ** (5.0 /mutationrate))): # Flip
        whichdim = np.random.randint(0, len(bounds))
        bound = bounds[whichdim]
        mutrange = (bound[1] - bound[0])*((1.0 - iteration / n_iterations)** (5.0 / mutationrate))
        ub = pos[whichdim] + mutrange
        lb = pos[whichdim] - mutrange

        # Fix if out of bounds
        if(lb < bound[0]):
            lb = bound[0]
        if(ub < bound[1]):
            ub = bound[1]

        pos[whichdim] = np.random.random() * (ub - lb) + lb

    return pos


def mutate(swarm, bounds, iteration, n_iterations, mutationrate):
    """ to prevent mature convergence to a false pareto front, a mutation factor is included that tries to explore 
        with all the particles at the beginning of the search. Then, we decrease rapidly (with respect to the number 
        of iterations) the number of particles that are affected by the mutation operator. Note that our mutation operator
        is applied not only to the particles of the swarm, but also to the range of each design variable of the problem 
        to be solved (using the same variation function). See Coello Coello et al. Handling Multiple Objectives with PSO (2004)."""
    try:
        temp_position = swarm.position.copy()

        position = np.array([mutate_particle(pos, bounds, iteration, n_iterations, mutationrate) for pos in temp_position]) # Iterate over particles

    except AttributeError:
        rep.logger.exception(
            "Please pass a Swarm class. You passed {}".format(type(swarm))
        )
        raise
    else:
        return position