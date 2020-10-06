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
import multiprocessing as mp
import pandas as pd
import os
import shutil
import bz2
import _pickle as cPickle
from pyentrp import entropy as ent
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import NearestNeighbors
import re
import subprocess
import datetime
import time
import glob 
import lhsmdu

from ..base import SwarmOptimizer
from ..utils import Reporter
from ..backend.handlers import BoundaryHandler, VelocityHandler
from ..backend.generators import create_multi_swarm
from ..backend.operators import compute_velocity, compute_position,compute_objective_function_multi

class MOPSO(SwarmOptimizer):
    def __init__(
        self,
        n_particles,
        dimensions,
        options,
        setup, ### BS ###
        bounds=None,
        bh_strategy="nearest",
        # bh_strategy="periodic",
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

        ### BS ###
        self.setup = setup
        self.performance_all_iter = pd.DataFrame()
        self.tof_all_iter = pd.DataFrame()
        self.particle_values_all_iter = pd.DataFrame()
        self.particle_values_converted_all_iter = pd.DataFrame()
        
    def optimize(self, objective_func, iters, fast=False,n_processes = None, **kwargs):
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

         # Setup Pool of processes for parallel evaluation
        pool = None if n_processes is None else mp.Pool(n_processes)

        self.swarm.pbest_cost = np.full((self.swarm_size[0],self.options["obj_dimensions"]), np.inf)

        # is this bit here necessary or can i push it into the for loop? otherwise would need to go through all the convert particle values / run batch fiel for petrel model steps beforehand
        # self.swarm.current_cost = objective_func(self.swarm.position, **kwargs)
        # self.swarm.update_archive()
        # self.swarm.pbest_pos = self.swarm.position
        # self.swarm.pbest_cost = self.swarm.current_cost
        for i in self.rep.pbar(iters, self.name):
            if not fast:
                sleep(0.01)


            # convert particle values to values suitable for model building
            self.convert_particle_values()

            # Built new geomodels (with batch files) in Petrel based upon converted particles.
            self.built_batch_file_for_petrel_models_uniform()

            # built multibat files to run petrel licences in parallel
            self.built_multibat_files()

            # run these batch files to built new geomodels 
            self.run_batch_file_for_petrel_models()

            # fmt: off
            # Compute cost for swarm current position, performance and personal best
            self.swarm.current_cost,self.LC, self.performance, self.setup= compute_objective_function_multi(
            self.swarm, objective_func,self.setup,i, pool=pool)  
            # self.swarm.current_cost = objective_func(self.swarm.position, **kwargs)
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

    def convert_particle_values(self):

            varminmax = self.setup["varminmax"]
            n_particles = self.setup["n_particles"]
            parameter_name = self.setup["columns"]
            continuous_discrete = self.setup["continuous_discrete"]
            n_parameters = len(parameter_name)

            # turn discrete parameters from continuous back into discrete parameters. Therefore have to set up the boundaries for each parameter
            converted_vals_range = varminmax
            # transpose particle values to make value conversion easier
            x_t = self.swarm.position.T.copy()

            # empty matrix to be filled in with the true converted values required for model building
            converted_vals = np.zeros(x_t.shape)

            # fill in correct values
            for index, value in enumerate(x_t,0):

                # continous values
                if continuous_discrete[index] == 0:
                    converted_vals[index] = np.around((value * (converted_vals_range[index,1] - converted_vals_range[index,0]) + converted_vals_range[index,0]),decimals= 10)
                # discrete values
                elif continuous_discrete[index] == 1:
                    converted_vals[index] = np.around((value * (converted_vals_range[index,1] - converted_vals_range[index,0]) + converted_vals_range[index,0]))

            # transpose back to initial setup
            self.swarm.position_converted = np.array(converted_vals.T).astype("float32")

            # # swap around parameters that work together and where min requires to be bigger than max. ohterwise wont do anzthing in petrel workflow.
            for i in range(0,n_particles):

                for j in range(0,n_parameters):
                    match_min = re.search("min",parameter_name[j],re.IGNORECASE)

                    if match_min:
                        match_max = re.search("max",parameter_name[j+1],re.IGNORECASE)

                        if match_max:
                            # if converted_particle_values[i,j] > converted_particle_values[i,j+1]:
                            #                     converted_particle_values[i,j],converted_particle_values[i,j+1] = converted_particle_values[i,j+1],converted_particle_values[i,j] 

                            if self.swarm.position_converted[i,j] > self.swarm.position_converted[i,j+1]:
                                self.swarm.position_converted[i,j],self.swarm.position_converted[i,j+1] = self.swarm.position_converted[i,j+1],self.swarm.position_converted[i,j] 

    def built_batch_file_for_petrel_models_uniform(self):
        # loading in settings that I set up on init_ABRM.py for this run

        seed = self.setup["set_seed"]
        n_modelsperbatch = self.setup["n_modelsperbatch"]
        runworkflow = self.setup["runworkflow"]
        n_particles = self.setup["n_particles"]
        n_parallel_petrel_licenses = self.setup["n_parallel_petrel_licenses"]
        parameter_type = self.setup["parameter_type"]
        parameter_name = self.setup["columns"]
        petrel_path = self.setup["petrel_path"]
        n_parameters = len(parameter_name)
        n_trainingimages = self.setup["n_trainingimages"]
        # base_path = self.setup["base_path"]
        #  Petrel has problems with batch files that get too long --> if I run
        #  20+ models at once. Therefore split it up in to sets of 3 particles / models
        #  per Petrel license and then run them in parallel. hard on CPU but
        #  shouldnt be too time expensive. Parameter iterations = number of
        #  models.
        particle = self.swarm.position_converted     # all particles together    
        particle_1d_array =  particle.reshape((particle.shape[0]*particle.shape[1]))    # all particles together                
        particlesperwf = np.linspace(0,n_modelsperbatch,n_parallel_petrel_licenses, endpoint = False,dtype = int) # this is how it should be. This is the name that each variable has per model in the petrel wf
        # particlesperwf = np.linspace(25,27,n_modelsperbatch, endpoint = True,dtype = int) # use 25,26,27 because of petrel wf. there the variables are named like that and cant bothered to change that.
        single_wf = [str(i) for i in np.tile(particlesperwf,n_particles)]
        single_particle_in_wf = [str(i) for i in np.arange(0,n_particles+1)]
        particle_str = np.asarray([str(i) for i in particle_1d_array]).reshape(particle.shape[0],particle.shape[1])
        parameter_name_str = np.asarray([parameter_name * n_particles]).reshape(particle.shape[0],particle.shape[1]) # not sure at all if this is working yet
        parameter_type_str = np.asarray([parameter_type * n_particles]).reshape(particle.shape[0],particle.shape[1])
        slicer_length = int(np.ceil(n_particles/n_modelsperbatch)) # always rounds up.
        slicer = np.arange(0,slicer_length,dtype = int)     # slicer = np.arange(0,(n_particles/n_modelsperbatch),dtype = int)

        # set up file path to petrel, petrel license and petrel projects and seed etc
        callpetrel = 'call "{}" ^'.format(petrel_path)
        license = '\n/licensePackage Standard ^'

        runworkflow = '\n/runWorkflow "{}" ^\n'.format(runworkflow)
        seed_petrel = '/nParm seed={} ^\n'.format(seed) 
        projectpath = []
        parallel_petrel_licenses = np.arange(0,n_parallel_petrel_licenses,1)
        for i in range(0,len(parallel_petrel_licenses)):
            path_petrel_projects = self.setup["base_path"] / "../Petrel_Projects/ABRM_"
            path = '\n"{}{}.pet"'.format(path_petrel_projects,parallel_petrel_licenses[i])
            projectpath.append(path)
        projectpath_repeat = projectpath * (len(slicer))    
        quiet = '/quiet ^' #wf wont pop up
        noshowpetrel = '\n/nosplashscreen ^' # petrel wont pop up
        exit = '\n/exit ^'  # exit petrel
        exit_2 = '\nexit' 	# exit bash file

        # set path for batch file to open and start writing into it
        for i in slicer:
            
            # path to petrel project
            path = projectpath_repeat[i]

            # path to batch file
            run_petrel_batch = self.setup["base_path"] / "../ABRM_functions/batch_files/run_petrel_{}.bat".format(i)

            # open batch file to start writing into it / updating it
            file = open(run_petrel_batch, "w+")

            # write petrelfilepath and licence part into file and seed
            file.write(callpetrel)
            file.write(license)
            file.write(runworkflow)
            file.write(seed_petrel)

            # generate n models per batch file / petrel license
            variables_per_model = np.arange((n_modelsperbatch*slicer[i]),(n_modelsperbatch*(i+1)))
            for _index_3, j in enumerate(variables_per_model):
                
                # parameter setup so that particles can be inserted into petrel workflow {} are place holders that will be fileld in with variable values,changing with each workflow
                Modelname = '/sparm ModelName_{}=M{} ^\n'.format(single_wf[j],single_particle_in_wf[j])
                file.write(Modelname)
                # for parameter name feature create something similar to singlewf or particle str feature as done above.

                for k in range(0,n_parameters):

                    # string parameters
                    if parameter_type_str[j,k] == 0:
                        parameter = '/sParm {}_{}={} ^\n'.format(parameter_name_str[j,k],single_wf[j],particle_str[j,k])
                        file.write(parameter)

                    # numeric parameters 
                    elif parameter_type_str[j,k] == 1:
                        parameter = '/nParm {}_{}={} ^\n'.format(parameter_name_str[j,k],single_wf[j],particle_str[j,k])
                        file.write(parameter)
                    
                    # training images
                    elif parameter_type_str[j,k] == 2:
                        for i in range(1,n_trainingimages+1):
            
                            if particle_str[j,k] == str(float(i)):
                                parameter = '/sParm {}_{}_{}={} ^\n'.format(parameter_name_str[j,k],str(i),single_wf[j],parameter_name_str[j,k])
                                file.write(parameter)
                            else:
                                parameter = '/sParm {}_{}_{}=None ^\n'.format(parameter_name_str[j,k],str(i),single_wf[j])
                                file.write(parameter)

                    # voronoi positioning --> not done in petrel therefore dont need to write in batch file
                    elif parameter_type_str[j,k] ==3:
                        pass

            # write into file
            file.write(quiet)
            file.write(noshowpetrel)
            file.write(exit)
            file.write(path)
            file.write(exit_2)

            # close file
            file.close()
    
    def built_multibat_files(self):
        #### built multi_batch file bat files that can launch several petrel licenses (run_petrel) at once

        n_particles = self.setup["n_particles"]
        n_modelsperbatch = self.setup["n_modelsperbatch"]
        n_parallel_petrel_licenses = self.setup["n_parallel_petrel_licenses"] 

        exit_bat = "\nexit"

        n_multibats = int(np.ceil(n_particles / n_modelsperbatch/n_parallel_petrel_licenses))
        run_petrel_ticker = 0 # naming of petrelfiles to run. problem: will alwazs atm write 3 files into multibatfile.

        for i in range(0,n_multibats):
            built_multibat = r'{}\batch_files\multi_bat_{}.bat'.format(self.setup["base_path"],i)
            file = open(built_multibat, "w+")

            for _j in range(0,n_parallel_petrel_licenses):

                run_petrel_bat = '\nStart {}/batch_files/run_petrel_{}.bat'.format(self.setup["base_path"],run_petrel_ticker)
                file.write(run_petrel_bat)
                run_petrel_ticker+=1

            file.write(exit_bat)
            file.close()
    
    def run_batch_file_for_petrel_models(self):

        petrel_on = self.setup["petrel_on"]
        n_particles = self.setup["n_particles"]
        n_modelsperbatch = self.setup["n_modelsperbatch"]
        n_parallel_petrel_licenses = self.setup["n_parallel_petrel_licenses"]    
        lock_files = str(self.setup["base_path"] /  "../Petrel_Projects/*.lock")
        kill_petrel =r'{}\batch_files\kill_petrel.bat'.format(self.setup["base_path"])

        if petrel_on == True:
            # initiate model by running batch file make sure that petrel has sufficient time to built the models and shut down again. 
            print(' Start building models',end = "\r")

            #how many multibat files to run
            n_multibats = int(np.ceil(n_particles / n_modelsperbatch/n_parallel_petrel_licenses))
            for i in range(0,n_multibats):
                run_multibat = r'{}\batch_files\multi_bat_{}.bat'.format(self.setup["base_path"],i)
                subprocess.call([run_multibat])
                # not continue until lock files are gone and petrel is finished.
                time.sleep(120)
                kill_timer = 1 # waits 2h before petrel project is shut down if it has a bug 
                while len(glob.glob(lock_files)) >= 1 or kill_timer > 7200:
                    kill_timer += 1
                    time.sleep(5)
                time.sleep(30)
                subprocess.call([kill_petrel]) # might need to add something that removes lock file here.


            print('Building models complete',end = "\r")

        else:
            print("dry run - no model building",end = "\r")
