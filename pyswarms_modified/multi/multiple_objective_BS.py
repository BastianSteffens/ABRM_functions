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
from ..backend.operators import compute_velocity, compute_position,compute_objective_function_multi, mutate

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

        # self.swarm.pbest_cost = np.full((self.swarm_size[0],self.options["obj_dimensions"]), np.inf)

        # is this bit here necessary or can i push it into the for loop? otherwise would need to go through all the convert particle values / run batch fiel for petrel model steps beforehand
        # self.swarm.current_cost = objective_func(self.swarm.position, **kwargs)
        # self.swarm.update_archive()
        # self.swarm.pbest_pos = self.swarm.position
        # self.swarm.pbest_cost = self.swarm.current_cost
        self.pre_iter_run(objective_func,self.setup,0,pool)

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

            # get output dfs 
            self.get_output_dfs(i)

            # calculate tof-based entropy of models in swarm that pass misfit criterion
            self.compute_tof_based_entropy_best_models()

             # calculate the gradient of entropy change of best models
            self.swarm.entropy_gradient = self.compute_best_model_diversity_gradient(i)

            # append outputdata from current iteration to all data
            self.performance_all_iter = self.performance_all_iter.append(self.performance, ignore_index = True)
            self.tof_all_iter = self.tof_all_iter.append(self.tof,ignore_index = True)
            self.particle_values_all_iter = self.particle_values_all_iter.append(self.particle_values, ignore_index = True)
            self.particle_values_converted_all_iter = self.particle_values_converted_all_iter.append(self.particle_values_converted, ignore_index = True)


            # save all reservoir models
            # self.save_data(i,iters)
            self.save_all_models()


            # save all outputdata from models, at beginning, the end and every 5 iteration for checkup
            if i == 0:
                self.save_data(i,iters)
            elif i == iters-1:
                self.save_data(i,iters)
            elif i % 4 == 0:
                self.save_data(i,iters)

            # Perform velocity and position updates
            self.swarm.velocity = compute_velocity(
                self.swarm, self.velocity_clamp, self.vh, self.bounds
            )
            self.swarm.position = compute_position(
                self.swarm, self.bounds, self.bh
            )

            # perform mutation on particle positions
            self.swarm.position = mutate(self.swarm,self.bounds,i,iters,0.5)
            

            ### BS ###
            # reset PSO if the entropy gradient of the best solutions is declining
            if self.swarm.entropy_gradient<0:
                print("entropy gradient is negative --> resetting PSO")
                # new init position
                self.init_pos = np.array(lhsmdu.sample(numDimensions = self.setup["n_particles"],numSamples = self.setup["n_parameters"]))
                # reset swarm
                self.reset()
                self.swarm.pbest_cost = np.full((self.swarm_size[0],self.options["obj_dimensions"]), np.inf)
                # reset inertia
                self.swarm.options["w"] = self.setup["initial_inertia"] 

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
        # particlesperwf = np.linspace(0,n_modelsperbatch,n_parallel_petrel_licenses, endpoint = False,dtype = int) # this is how it should be. This is the name that each variable has per model in the petrel wf
        particlesperwf = np.linspace(25,27,n_modelsperbatch, endpoint = True,dtype = int) # use 25,26,27 because of petrel wf. there the variables are named like that and cant bothered to change that.
        single_wf = [str(i) for i in np.tile(particlesperwf,n_particles)]
        single_particle_in_wf = [str(i) for i in np.arange(0,n_particles+1)]
        particle_str = np.asarray([str(i) for i in particle_1d_array]).reshape(particle.shape[0],particle.shape[1])
        parameter_name_str = np.asarray([parameter_name * n_particles]).reshape(particle.shape[0],particle.shape[1]) # not sure at all if this is working yet
        parameter_type_str = np.asarray([parameter_type * n_particles]).reshape(particle.shape[0],particle.shape[1])
        slicer_length = int(np.ceil(n_particles/n_modelsperbatch)) # always rounds up.
        slicer = np.arange(0,slicer_length,dtype = int)     # slicer = np.arange(0,(n_particles/n_modelsperbatch),dtype = int)

        # set up file path to petrel, petrel license and petrel projects and seed etc
        callpetrel = 'call "{}" ^'.format(petrel_path)
        # license = '\n/licensePackage Standard ^'
        license = '\n/licensePackage Petrel_112614311_MACKPHiBP8aUA ^'


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
    
    def get_output_dfs(self,i):
        """prepare dfs of whole swarm with output that is ready for postprocessing"""
        n_shedules = self.setup["n_shedules"]

        # raw data from FD
        self.tof = pd.DataFrame()
        for shedule_no in range(n_shedules):

            misfit = "misfit_" + str(shedule_no)
            tof = "tof_" + str(shedule_no)
            self.tof[tof] = self.performance[tof]
            self.tof[misfit] = self.performance[misfit]
            self.tof["iteration"] = self.performance["iteration"]
            self.tof["particle_no"] = self.performance["particle_no"]

        # self.swarm_performance_short = self.performance.iloc[::100,:].copy()
        
        # converted particles and raw particles tother with simulation outputs
        columns = self.setup["columns"]
        folder_path = self.setup["folder_path"]
        self.particle_values_converted = pd.DataFrame(data = self.swarm.position_converted,columns = columns)
        self.particle_values = pd.DataFrame(data = self.swarm.position,columns = columns)
        # add iteration to df
        self.particle_values_converted["iteration"] = i
        self.particle_values["iteration"] =i
        # add particle no to df
        particle_no = np.arange(self.swarm.position_converted.shape[0], dtype = int)
        self.particle_values_converted["particle_no"] = particle_no
        self.particle_values["particle_no"] = particle_no

        #add both misfits and LCs to df
        for shedule_no in range(n_shedules):

                misfit = "misfit_" + str(shedule_no)
                LC = "LC_" + str(shedule_no)
                self.particle_values[misfit] = self.swarm.current_cost[:,shedule_no]
                self.particle_values_converted[misfit] = self.swarm.current_cost[:,shedule_no]
                self.particle_values[LC] = self.LC[:,shedule_no].astype("float32")
                self.particle_values_converted[LC] = self.LC[:,shedule_no].astype("float32")

    def compute_tof_based_entropy_best_models(self):

        """ function to compute the entropy of the models that fulfill misfit criterion based on the time-of-flight"""
       
        n_shedules = self.setup["n_shedules"]

        #generate temporary all iter particle df
        temp_all_iter_particle_values = self.particle_values_all_iter.append(self.particle_values, ignore_index = True)
        temp_all_iter_tof = self.tof_all_iter.append(self.tof,ignore_index = True)

        particle_values_all_iter_best = temp_all_iter_particle_values.copy()
        # get models tht fulfil misfit criterion of all shedules
        for shedule_no in range(n_shedules):
            misfit = "misfit_" + str(shedule_no)
            particle_values_all_iter_best = particle_values_all_iter_best[particle_values_all_iter_best[misfit] <= self.setup["best_models"]]

        all_cells_entropy = []

        # check if more than 1 models exist that are better than misfit
        if particle_values_all_iter_best.shape[0] > 3:
            print(" got {} best models ...calculating entropy".format(particle_values_all_iter_best.shape[0]))
            
            tof_entropy_all_shedules = []
            for shedule_no in range(n_shedules):
                
                tof_column = "tof_" + str(shedule_no)
                # filter out tof for all best models and make it readable for clustering
                iteration = particle_values_all_iter_best.iteration.tolist()
                particle_no =  particle_values_all_iter_best.particle_no.tolist()  
                best_tof = pd.DataFrame(columns = np.arange(200*100*7))
        
                tof_all = pd.DataFrame()
                for i in range(particle_values_all_iter_best.shape[0]):

                    tof = temp_all_iter_tof[(temp_all_iter_tof.iteration == iteration[i]) & (temp_all_iter_tof.particle_no == particle_no[i])][tof_column]#.tof
                    tof.reset_index(drop=True, inplace=True)
                    tof_all = tof_all.append(tof,ignore_index = True)

                best_tof = tof_all
                best_tof["iteration"] = iteration
                best_tof["particle_no"] = particle_no
                best_tof.set_index(particle_values_all_iter_best.index.values,inplace = True)


                for i in range(best_tof.shape[1]-2):

                    cell = np.round(np.array(best_tof[i]).reshape(-1)/60/60/24/365.25)

                    #over 20 years tof is binend together.considered unswept.
                    cell_binned = np.digitize(cell,bins=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20])
            
                    # calculate entropy based upon clusters
                    cell_entropy = np.array(ent.shannon_entropy(cell_binned))
                    all_cells_entropy.append(cell_entropy)

                # sum up entropy for all cells
                tof_based_entropy_best_models = np.sum(np.array(all_cells_entropy))
                tof_entropy_all_shedules.append(tof_based_entropy_best_models)

            tof_entropy_all_shedules_sum = np.sum(np.array(tof_entropy_all_shedules))
            print("entropy: {}".format(tof_entropy_all_shedules_sum))


        else:
            print(" no best models")
            tof_entropy_all_shedules = []
            for shedule_no in range(n_shedules):
                tof_entropy_all_shedules.append(0)

            tof_entropy_all_shedules_sum = 0
        
        self.particle_values["tof_entropy_all_shedules_sum"] = tof_entropy_all_shedules_sum
        self.particle_values_converted["tof_entropy_all_shedules_sum"] = tof_entropy_all_shedules_sum

        for shedule_no in range(n_shedules):
            entropy_single_shedule = "tof_entropy_shedule_" + str(shedule_no)

            self.particle_values[entropy_single_shedule] = tof_entropy_all_shedules[shedule_no]
            self.particle_values_converted[entropy_single_shedule] = tof_entropy_all_shedules[shedule_no]

    def compute_best_model_diversity_gradient(self,i,delta = 5):

        """ check how if I am still producing new best models or  if they are just hovering around similar models.
            This is done by checking how the slope of tof_based_entropy changes over iterations. If that slope falls below 0
            the PSO should spread out again and reset the global/local best memory.
        """
        delta = 5
       # dont do this analysis in the beginning, if not enough data availabe
        if i < delta:
            slope = 0
        
        else:
            iterations_to_check = np.array(np.arange(i-delta,i))
            linear_regressor = LinearRegression()
            entropy = []
            for j in range(delta):
                entropy.append(np.array(self.particle_values_converted_all_iter[(self.particle_values_converted_all_iter["iteration"] ==iterations_to_check[j]) & (self.particle_values_converted_all_iter["particle_no"] ==0)].tof_entropy_all_shedules_sum))
            
            iterations_to_check = np.array(np.arange(i-delta,i)).reshape(-1,1)

            linear_regressor.fit(iterations_to_check,entropy)
            slope = linear_regressor.coef_.flatten()
        
        self.particle_values["entropy_slope"] = float(slope)
        self.particle_values_converted["entropy_slope"] = float(slope)

        print("entropy slope for last {} iterations : {}".format(delta,float(slope)))
        return slope

    def save_all_models(self):
        """ Save reservoir models to output folder """
        # loading in settings that I set up on init_ABRM.py for this run

        save_models = self.setup["save_all_models"]
        n_shedules = self.setup["n_shedules"]
        base_path = self.setup["base_path"]

        if save_models == True:
            destination_path = self.setup["folder_path"] / 'all_models'
            data_path = destination_path / "DATA"
            include_path = destination_path / "INCLUDE"
            permx_path = include_path / "PERMX"
            permy_path = include_path / "PERMY"
            permz_path = include_path / "PERMZ"
            poro_path = include_path / "PORO"

            n_particles = self.setup["n_particles"]
            source_path = base_path / "../FD_Models"

            if not os.path.exists(destination_path):

                # make folders and subfolders
                os.makedirs(destination_path)
                os.makedirs(data_path)
                os.makedirs(include_path)
                os.makedirs(permx_path)
                os.makedirs(permy_path)
                os.makedirs(permz_path)
                os.makedirs(poro_path)
                for shedule_no in range(n_shedules):
                #copy and paste generic files into Data]
                    schedule = "SCHEDULE_multi_" + str(shedule_no)
                    DP_pvt_src_path = source_path / "INCLUDE/DP_pvt.INC"
                    GRID_src_path = source_path / "INCLUDE/GRID.GRDECL"
                    ROCK_RELPERMS_src_path = source_path / "INCLUDE/ROCK_RELPERMS.INC"
                    SCHEDULE_src_path = source_path / "INCLUDE/{}.INC".format(schedule)
                    SOLUTION_src_path = source_path / "INCLUDE/SOLUTION.INC"
                    SUMMARY_src_path = source_path / "INCLUDE/SUMMARY.INC"

                    DP_pvt_dest_path = include_path / "DP_pvt.INC"
                    GRID_dest_path = include_path / "GRID.GRDECL"
                    ROCK_RELPERMS_dest_path = include_path / "ROCK_RELPERMS.INC"
                    SCHEDULE_dest_path = include_path / "{}.INC".format(schedule)
                    SOLUTION_dest_path = include_path / "SOLUTION.INC"
                    SUMMARY_dest_path = include_path / "SUMMARY.INC"
                    
                    shutil.copy(DP_pvt_src_path,DP_pvt_dest_path)
                    shutil.copy(GRID_src_path,GRID_dest_path)
                    shutil.copy(ROCK_RELPERMS_src_path,ROCK_RELPERMS_dest_path)
                    shutil.copy(SCHEDULE_src_path,SCHEDULE_dest_path)
                    shutil.copy(SOLUTION_src_path,SOLUTION_dest_path)
                    shutil.copy(SUMMARY_src_path,SUMMARY_dest_path)

            if os.path.exists(destination_path):

                PERMX_src_path = source_path / "INCLUDE/PERMX"
                PERMY_src_path = source_path / "INCLUDE/PERMY"
                PERMZ_src_path = source_path / "INCLUDE/PERMZ"
                PORO_src_path = source_path / "INCLUDE/PORO"

                PERMX_dest_path = include_path / "PERMX"
                PERMY_dest_path = include_path / "PERMY"
                PERMZ_dest_path = include_path / "PERMZ"
                PORO_dest_path = include_path / "PORO"

                model_id = 0

                for particle_id in range(0,n_particles):

                    # set path for Datafile
                    data_file_path = data_path / "M{}.DATA".format(model_id)
                    # getting higher model numbers for saving
                    while os.path.exists(data_file_path):
                        model_id += 1
                        data_file_path = data_path / "M{}.DATA".format(model_id)

                    # open datafile file to start writing into it / updating it
                    for shedule_no in range(n_shedules):
                        self.built_data_file(data_file_path,model_id,shedule_no)

                    #copy and paste permxyz and poro files to new location
                    permx_file_src_path = PERMX_src_path / "M{}.GRDECL".format(particle_id)  
                    permy_file_src_path = PERMY_src_path / "M{}.GRDECL".format(particle_id)  
                    permz_file_src_path = PERMZ_src_path / "M{}.GRDECL".format(particle_id)  
                    poro_file_src_path = PORO_src_path / "M{}.GRDECL".format(particle_id)

                    permx_file_dest_path = PERMX_dest_path / "M{}.GRDECL".format(model_id)  
                    permy_file_dest_path = PERMY_dest_path / "M{}.GRDECL".format(model_id)  
                    permz_file_dest_path = PERMZ_dest_path / "M{}.GRDECL".format(model_id)  
                    poro_file_dest_path = PORO_dest_path / "M{}.GRDECL".format(model_id) 

                    shutil.copy(permx_file_src_path,permx_file_dest_path)
                    shutil.copy(permy_file_src_path,permy_file_dest_path)
                    shutil.copy(permz_file_src_path,permz_file_dest_path)
                    shutil.copy(poro_file_src_path,poro_file_dest_path)

    def built_data_file(self,data_file_path,model_id,shedule_no):
        """ built data files that can be used for flow simulations or flow diagnostics """
        schedule = "SCHEDULE_multi_" + str(shedule_no)
        data_file = "RUNSPEC\n\nTITLE\nModel_{}_{}\n\nDIMENS\n--NX NY NZ\n200 100 7 /\n\n--Phases\nOIL\nWATER\n\n--DUALPORO\n--NODPPM\n\n--Units\nMETRIC\n\n--Number of Saturation Tables\nTABDIMS\n1 /\n\n--Maximum number of Wells\nWELLDIMS\n10 100 5 10 /\n\n--First Oil\nSTART\n1 OCT 2017 /\n\n--Memory Allocation\nNSTACK\n100 /\n\n--How many warnings allowed, but terminate after first error\nMESSAGES\n11*5000 1 /\n\n--Unified Output Files\nUNIFOUT\n\n--======================================================================\n\nGRID\n--Include corner point geometry model\nINCLUDE\n'..\INCLUDE\GRID.GRDECL'\n/\n\nACTNUM\n140000*1 /\n\n--Porosity\nINCLUDE\n'..\INCLUDE\PORO\M{}.GRDECL'\n/\n\n--Permeability\nINCLUDE\n'..\INCLUDE\PERMX\M{}.GRDECL'\n/\nINCLUDE\n'..\INCLUDE\PERMY\M{}.GRDECL'\n/\nINCLUDE\n'..\INCLUDE\PERMZ\M{}.GRDECL'\n/\n\n--Net to Gross\nNTG\n140000*1\n/\n\n--Output .INIT file to allow viewing of grid data in post proessor\nINIT\n\n--======================================================================\n\nPROPS\n\nINCLUDE\n'..\INCLUDE\DP_pvt.inc' /\n\nINCLUDE\n'..\INCLUDE\ROCK_RELPERMS.INC' /\n\n--======================================================================\n\nREGIONS\n\nEQLNUM\n140000*1\n/\nSATNUM\n140000*1\n/\nPVTNUM\n140000*1\n/\n\n--======================================================================\n\nSOLUTION\n\nINCLUDE\n'..\INCLUDE\SOLUTION.INC' /\n\n--======================================================================\n\nSUMMARY\n\nINCLUDE\n'..\INCLUDE\SUMMARY.INC' /\n\n--======================================================================\n\nSCHEDULE\n\nINCLUDE\n'..\INCLUDE\{}.INC' /\n\nEND".format(model_id,shedule_no,model_id,model_id,model_id,model_id,schedule)  
        
        file = open(data_file_path, "w+")
        # write petrelfilepath and licence part into file and seed
        file.write(data_file)

        # close file
        file.close()
    
    def save_data(self,i,iters):
        """ save df to csv files / pickle that contains all data used for postprocessing """
        print("Saving Data at Iteration {}/{}".format(i,iters-1))

        # filepath setup
        folder_path = self.setup["folder_path"]
        
        # output_file_performance = "swarm_performance_all_iter.csv"
        output_file_performance = "swarm_performance_all_iter.pbz2"
        output_file_partilce_values_converted = "swarm_particle_values_converted_all_iter.csv"
        output_file_partilce_values = "swarm_particle_values_all_iter.csv"
        tof_file = "tof_all_iter.pbz2"
        setup_file = "variable_settings.pickle"

        file_path_tof = folder_path / tof_file
        file_path_performance = folder_path / output_file_performance
        # file_path_performance_pickle = folder_path / output_file_performance_pickle

        file_path_particles_values_converted = folder_path / output_file_partilce_values_converted
        file_path_particles_values = folder_path / output_file_partilce_values
        file_path_setup = folder_path / setup_file

        # make folder
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        
        # save all
        # self.performance_all_iter.to_csv(file_path_performance,index=False)
        self.particle_values_converted_all_iter.to_csv(file_path_particles_values_converted,index=False)
        self.particle_values_all_iter.to_csv(file_path_particles_values,index=False)
        with bz2.BZ2File(file_path_tof,"w") as f:
            cPickle.dump(self.tof_all_iter,f,protocol= 4)
        with bz2.BZ2File(file_path_setup,"w") as f:
            cPickle.dump(self.setup,f)
        with bz2.BZ2File(file_path_performance,"w") as f:
            cPickle.dump(self.performance_all_iter,f, protocol= 4) # allow for larger datasets than 4 gb

    def pre_iter_run(self,objective_func,setup,iteration,pool):

        # convert particle values to values suitable for model building
        self.convert_particle_values()

        # Built new geomodels (with batch files) in Petrel based upon converted particles.
        self.built_batch_file_for_petrel_models_uniform()

        # built multibat files to run petrel licences in parallel
        self.built_multibat_files()

        # run these batch files to built new geomodels 
        self.run_batch_file_for_petrel_models()

        # Compute cost for swarm current position, performance and personal best
        self.swarm.current_cost,self.LC, self.performance, self.setup= compute_objective_function_multi(
        self.swarm, objective_func,self.setup,0, pool=pool)  

        self.swarm.update_archive()
        self.swarm.pbest_pos = self.swarm.position
        self.swarm.pbest_cost = self.swarm.current_cost