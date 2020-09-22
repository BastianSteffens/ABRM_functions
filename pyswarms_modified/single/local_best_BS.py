# -*- coding: utf-8 -*-

r"""
A Local-best Particle Swarm Optimization (lbest PSO) algorithm.

Similar to global-best PSO, it takes a set of candidate solutions,
and finds the best solution using a position-velocity update method.
However, it uses a ring topology, thus making the particles
attracted to its corresponding neighborhood.

The position update can be defined as:

.. math::

   x_{i}(t+1) = x_{i}(t) + v_{i}(t+1)

Where the position at the current timestep :math:`t` is updated using
the computed velocity at :math:`t+1`. Furthermore, the velocity update
is defined as:

.. math::

   v_{ij}(t + 1) = m * v_{ij}(t) + c_{1}r_{1j}(t)[y_{ij}(t) − x_{ij}(t)] + c_{2}r_{2j}(t)[\hat{y}_{j}(t) − x_{ij}(t)]

However, in local-best PSO, a particle doesn't compare itself to the
overall performance of the swarm. Instead, it looks at the performance
of its nearest-neighbours, and compares itself with them. In general,
this kind of topology takes much more time to converge, but has a more
powerful explorative feature.

In this implementation, a neighbor is selected via a k-D tree
imported from :code:`scipy`. Distance are computed with either
the L1 or L2 distance. The nearest-neighbours are then queried from
this k-D tree. They are computed for every iteration.

An example usage is as follows:

.. code-block:: python

    import pyswarms as ps
    from pyswarms.utils.functions import single_obj as fx

    # Set-up hyperparameters
    options = {'c1': 0.5, 'c2': 0.3, 'w': 0.9, 'k': 3, 'p': 2}

    # Call instance of LBestPSO with a neighbour-size of 3 determined by
    # the L2 (p=2) distance.
    optimizer = ps.single.LocalBestPSO(n_particles=10, dimensions=2,
                                       options=options)

    # Perform optimization
    stats = optimizer.optimize(fx.sphere, iters=100)

This algorithm was adapted from one of the earlier works of
J. Kennedy and R.C. Eberhart in Particle Swarm Optimization
[IJCNN1995]_ [MHS1995]_

.. [IJCNN1995] J. Kennedy and R.C. Eberhart, "Particle Swarm Optimization,"
    Proceedings of the IEEE International Joint Conference on Neural
    Networks, 1995, pp. 1942-1948.

.. [MHS1995] J. Kennedy and R.C. Eberhart, "A New Optimizer using Particle
    Swarm Theory,"  in Proceedings of the Sixth International
    Symposium on Micromachine and Human Science, 1995, pp. 39–43.
"""

import logging
import numpy as np
import pandas as pd
import multiprocessing as mp
import os
import shutil
import bz2
import _pickle as cPickle
from pyentrp import entropy as ent
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import NearestNeighbors
import re

from ..backend.operators import compute_pbest, compute_objective_function
from ..backend.topology import Ring
from ..backend.handlers import BoundaryHandler, VelocityHandler
from ..base import SwarmOptimizer
from ..utils.reporter import Reporter

class LocalBestPSO(SwarmOptimizer):
    def __init__(
        self,
        n_particles,
        dimensions,
        options,
        setup, ###BS###
        bounds=None,
        bh_strategy="nearest",
        velocity_clamp=None,
        vh_strategy="unmodified",
        center=1.00,
        ftol=-np.inf,
        init_pos=None,
        static=False,
    ):
        """Initialize the swarm

        Attributes
        ----------
        n_particles : int
            number of particles in the swarm.
        dimensions : int
            number of dimensions in the space.
        bounds : tuple of numpy.ndarray
            a tuple of size 2 where the first entry is the minimum bound
            while the second entry is the maximum bound. Each array must
            be of shape :code:`(dimensions,)`.
        bh_strategy : str
            a strategy for the handling of out-of-bounds particles.
        velocity_clamp : tuple (default is :code:`(0,1)`)
            a tuple of size 2 where the first entry is the minimum velocity
            and the second entry is the maximum velocity. It
            sets the limits for velocity clamping.
        vh_strategy : str
            a strategy for the handling of the velocity of out-of-bounds particles.
        center : list, optional
            an array of size :code:`dimensions`
        ftol : float
            relative error in objective_func(best_pos) acceptable for
            convergence. Default is :code:`-np.inf`
        options : dict with keys :code:`{'c1', 'c2', 'w', 'k', 'p'}`
            a dictionary containing the parameters for the specific
            optimization technique
                * c1 : float
                    cognitive parameter
                * c2 : float
                    social parameter
                * w : float
                    inertia parameter
                * k : int
                    number of neighbors to be considered. Must be a
                    positive integer less than :code:`n_particles`
                * p: int {1,2}
                    the Minkowski p-norm to use. 1 is the
                    sum-of-absolute values (or L1 distance) while 2 is
                    the Euclidean (or L2) distance.
        init_pos : numpy.ndarray, optional
            option to explicitly set the particles' initial positions. Set to
            :code:`None` if you wish to generate the particles randomly.
        static: bool
            a boolean that decides whether the Ring topology
            used is static or dynamic. Default is `False`
        """
        # Initialize logger
        self.logger = logging.getLogger(__name__)
        # Assign k-neighbors and p-value as attributes
        self.k, self.p = options["k"], options["p"]
        # Initialize parent class
        super(LocalBestPSO, self).__init__(
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
        # Initialize the topology
        self.top = Ring(static=static)
        self.bh = BoundaryHandler(strategy=bh_strategy)
        self.vh = VelocityHandler(strategy=vh_strategy)
        self.name = __name__

        ### BS ###
        self.setup = setup
        self.performance_all_iter = pd.DataFrame()
        self.tof_all_iter = pd.DataFrame()
        self.particle_values_all_iter = pd.DataFrame()
        self.particle_values_converted_all_iter = pd.DataFrame()
        

    def optimize(self, objective_func, iters, n_processes=None, **kwargs): ### BS ###

        """Optimize the swarm for a number of iterations

        Performs the optimization to evaluate the objective
        function :code:`f` for a number of iterations :code:`iter.`

        Parameters
        ----------
        objective_func : callable
            objective function to be evaluated
        setup : dict with all settings for mdoelling
        iters : int
            number of iterations
        n_processes : int
            number of processes to use for parallel particle evaluation (default: None = no parallelization)
        kwargs : dict
            arguments for the objective function

        Returns
        -------
        tuple
            the local best cost and the local best position among the
            swarm.
        """
        self.rep.log("Obj. func. args: {}".format(kwargs), lvl=logging.DEBUG)
        self.rep.log(
            "Optimize for {} iters with {}".format(iters, self.options),
            lvl=logging.INFO,
        )
        # Populate memory of the handlers
        self.bh.memory = self.swarm.position
        self.vh.memory = self.swarm.position

        # Setup Pool of processes for parallel evaluation
        pool = None if n_processes is None else mp.Pool(n_processes)

        self.swarm.pbest_cost = np.full(self.swarm_size[0], np.inf)

        for i in self.rep.pbar(iters, self.name):

            # convert particle values to values suitable for model building
            self.convert_particle_values()

            # Built new geomodels (with batch files) in Petrel based upon converted particles.
            self.built_batch_file_for_petrel_models_uniform()

            # built multibat files to run petrel licences in parallel
            self.built_multibat_files()

            # run these batch files to built new geomodels 
            self.run_batch_file_for_petrel_models()

            # Compute cost for swarm current position, performance and personal best
            self.swarm.current_cost,self.LC, self.performance, self.setup= compute_objective_function(
            self.swarm, objective_func,self.setup,i, pool=pool)       
            self.swarm.pbest_pos, self.swarm.pbest_cost = compute_pbest(self.swarm)
            best_cost_yet_found = np.min(self.swarm.best_cost)
            # Update gbest from neighborhood
            self.swarm.best_pos, self.swarm.best_cost = self.top.compute_gbest(
                self.swarm, p=self.p, k=self.k)

            ### BS ###

            # get output dfs 
            self.get_output_dfs(i)

            # calculate tof-based entropy of models in swarm that pass misfit criterion
            self.compute_tof_based_entropy_best_models()

            # calculate the gradient of entropy change of best models
            entropy_gradient = self.compute_best_model_diversity_gradient(i)

            # append outputdata from current iteration to all data
            self.performance_all_iter = self.performance_all_iter.append(self.performance, ignore_index = True)
            self.tof_all_iter = self.tof_all_iter.append(self.tof,ignore_index = True)
            self.particle_values_all_iter = self.particle_values_all_iter.append(self.particle_values, ignore_index = True)
            self.particle_values_converted_all_iter = self.particle_values_converted_all_iter.append(self.particle_values_converted, ignore_index = True)

            # save all reservoir models
            self.save_all_models()
            # self.save_data(i,iters)

            # save all outputdata from models, at beginning, the end and every 2 iteration for checkup
            if i == 0:
                self.save_data(i,iters)
            elif i == iters-1:
                self.save_data(i,iters)
            elif i % 2 == 0:
                self.save_data(i,iters)

            self.rep.hook(best_cost=np.min(self.swarm.best_cost))
            # Save to history
            hist = self.ToHistory(
                best_cost=self.swarm.best_cost,
                mean_pbest_cost=np.mean(self.swarm.pbest_cost),
                mean_neighbor_cost=np.mean(self.swarm.best_cost),
                position=self.swarm.position,
                velocity=self.swarm.velocity,
            )
            self._populate_history(hist)
            # Verify stop criteria based on the relative acceptable cost ftol
            relative_measure = self.ftol * (1 + np.abs(best_cost_yet_found))
            if (
                np.abs(self.swarm.best_cost - best_cost_yet_found)
                < relative_measure
            ):
                break

            # Perform position velocity update
            self.swarm.velocity = self.top.compute_velocity(
                self.swarm, self.velocity_clamp, self.vh, self.bounds
            )
            self.swarm.position = self.top.compute_position(
                self.swarm, self.bounds, self.bh
            )
            ### BS ###
            if entropy_gradient<0:
                break
            # self.reset()

        # Obtain the final best_cost and the final best_position
        final_best_cost = self.swarm.best_cost.copy()
        final_best_pos = self.swarm.pbest_pos[self.swarm.pbest_cost.argmin()].copy()
        # Write report in log and return final cost and position
        self.rep.log(
            "Optimization finished | best cost: {}, best pos: {}".format(
                final_best_cost, final_best_pos
            ),
            lvl=logging.INFO,
        )
        return (final_best_cost, final_best_pos)

    ### BS ###

    def compute_tof_based_entropy_best_models(self):
        """ function to compute the entropy of the models that fulfill misfit criterion based on the time-of-flight"""

        #generate temporary all iter particle df
        temp_all_iter_particle_values = self.particle_values_all_iter.append(self.particle_values, ignore_index = True)
        temp_all_iter_tof = self.tof_all_iter.append(self.tof,ignore_index = True)

        # get models taht fulfil misfit criterion
        particle_values_all_iter_best = temp_all_iter_particle_values[temp_all_iter_particle_values["misfit"] <= self.setup["best_models"]]

        all_cells_entropy = []

        # check if more than 1 models exist that are better than misfit
        if particle_values_all_iter_best.shape[0] > 3:
            print("got {} best models ...calculating entropy".format(particle_values_all_iter_best.shape[0]))
            
            # filter out tof for all best models and make it readable for clustering
            iteration = particle_values_all_iter_best.iteration.tolist()
            particle_no =  particle_values_all_iter_best.particle_no.tolist()  
            best_tof = pd.DataFrame(columns = np.arange(200*100*7))
      
            tof_all = pd.DataFrame()
            for i in range(particle_values_all_iter_best.shape[0]):

                tof = temp_all_iter_tof[(temp_all_iter_tof.iteration == iteration[i]) & (temp_all_iter_tof.particle_no == particle_no[i])].tof
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
            print("entropy: {}".format(tof_based_entropy_best_models))


        else:
            print("no best models")
            tof_based_entropy_best_models = 0
        
        self.particle_values["tof_based_entropy_best_models"] = tof_based_entropy_best_models
        self.particle_values_converted["tof_based_entropy_best_models"] = tof_based_entropy_best_models

    def compute_best_model_diversity_gradient(self,i,delta = 1):
        """ check how if I am still producing new best models or  if they are just hovering around similar models.
            This is done by checking how the slope of tof_based_entropy changes over iterations. If that slope falls below 0
            the PSO should spread out again and reset the global/local best memory.
        """
       # dont do this analysis in the beginning, if not enough data availabe
        if i < delta:
            slope = 0
        
        else:
            iterations_to_check = np.array(np.arange(i-delta,i))#.reshape(-1,1)
            linear_regressor = LinearRegression()
            entropy = []
            for j in range(delta):
                entropy.append(np.array(self.particle_values_converted_all_iter[(self.particle_values_converted_all_iter["iteration"] ==iterations_to_check[j]) & (self.particle_values_converted_all_iter["particle_no"] ==0)].tof_based_entropy_best_models))
            
            iterations_to_check = np.array(np.arange(i-delta,i)).reshape(-1,1)

            linear_regressor.fit(iterations_to_check,entropy)
            slope = linear_regressor.coef_.flatten()
        
        self.particle_values["entropy_slope"] = float(slope)
        self.particle_values_converted["entropy_slope"] = float(slope)

        return slope

    def built_data_file(self,data_file_path,model_id):
        """ built data files that can be used for flow simulations or flow diagnostics """
        schedule = self.setup["schedule"]
        data_file = "RUNSPEC\n\nTITLE\nModel_{}\n\nDIMENS\n--NX NY NZ\n200 100 7 /\n\n--Phases\nOIL\nWATER\n\n--DUALPORO\n--NODPPM\n\n--Units\nMETRIC\n\n--Number of Saturation Tables\nTABDIMS\n1 /\n\n--Maximum number of Wells\nWELLDIMS\n10 100 5 10 /\n\n--First Oil\nSTART\n1 OCT 2017 /\n\n--Memory Allocation\nNSTACK\n100 /\n\n--How many warnings allowed, but terminate after first error\nMESSAGES\n11*5000 1 /\n\n--Unified Output Files\nUNIFOUT\n\n--======================================================================\n\nGRID\n--Include corner point geometry model\nINCLUDE\n'..\INCLUDE\GRID.GRDECL'\n/\n\nACTNUM\n140000*1 /\n\n--Porosity\nINCLUDE\n'..\INCLUDE\PORO\M{}.GRDECL'\n/\n\n--Permeability\nINCLUDE\n'..\INCLUDE\PERMX\M{}.GRDECL'\n/\nINCLUDE\n'..\INCLUDE\PERMY\M{}.GRDECL'\n/\nINCLUDE\n'..\INCLUDE\PERMZ\M{}.GRDECL'\n/\n\n--Net to Gross\nNTG\n140000*1\n/\n\n--Output .INIT file to allow viewing of grid data in post proessor\nINIT\n\n--======================================================================\n\nPROPS\n\nINCLUDE\n'..\INCLUDE\DP_pvt.inc' /\n\nINCLUDE\n'..\INCLUDE\ROCK_RELPERMS.INC' /\n\n--======================================================================\n\nREGIONS\n\nEQLNUM\n140000*1\n/\nSATNUM\n140000*1\n/\nPVTNUM\n140000*1\n/\n\n--======================================================================\n\nSOLUTION\n\nINCLUDE\n'..\INCLUDE\SOLUTION.INC' /\n\n--======================================================================\n\nSUMMARY\n\nINCLUDE\n'..\INCLUDE\SUMMARY.INC' /\n\n--======================================================================\n\nSCHEDULE\n\nINCLUDE\n'..\INCLUDE\{}.INC' /\n\nEND".format(model_id,model_id,model_id,model_id,model_id,schedule)  
        
        file = open(data_file_path, "w+")
        # write petrelfilepath and licence part into file and seed
        file.write(data_file)

        # close file
        file.close()
        
    def save_all_models(self):
        """ Save reservoir models to output folder """
        # loading in settings that I set up on init_ABRM.py for this run

        save_models = self.setup["save_all_models"]
        schedule = self.setup["schedule"]
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

                #copy and paste generic files into Data
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
                    self.built_data_file(data_file_path,model_id)

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

    def save_data(self,i,iters):
        """ save df to csv files / pickle that contains all data used for postprocessing """
        print("Saving Data at Iteration {}/{}".format(i,iters-1))

        # filepath setup
        folder_path = self.setup["folder_path"]
        
        output_file_performance = "swarm_performance_all_iter.csv"
        output_file_partilce_values_converted = "swarm_particle_values_converted_all_iter.csv"
        output_file_partilce_values = "swarm_particle_values_all_iter.csv"
        tof_file = "tof_all_iter.pbz2"
        setup_file = "variable_settings.pickle"


        file_path_tof = folder_path / tof_file
        file_path_performance = folder_path / output_file_performance
        file_path_particles_values_converted = folder_path / output_file_partilce_values_converted
        file_path_particles_values = folder_path / output_file_partilce_values
        file_path_setup = folder_path / setup_file

        # make folder
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        
        # save all
        self.performance_all_iter.to_csv(file_path_performance,index=False)
        self.particle_values_converted_all_iter.to_csv(file_path_particles_values_converted,index=False)
        self.particle_values_all_iter.to_csv(file_path_particles_values,index=False)
        with bz2.BZ2File(file_path_tof,"w") as f:
            cPickle.dump(self.tof_all_iter,f)
        with bz2.BZ2File(file_path_setup,"w") as f:
            cPickle.dump(self.setup,f)

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

                        print("swapaarroo is hapening")

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
            print('Start building models',end = "\r")

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
        
        # raw data from FD
        self.tof = self.performance[["tof","misfit","iteration","particle_no"]].copy()
        self.swarm_performance_short = self.performance.iloc[::100,:].copy()
        
        # converted particles and raw particles tother with simulation outputs
        columns = self.setup["columns"]
        folder_path = self.setup["folder_path"]
        self.particle_values_converted = pd.DataFrame(data = self.swarm.position_converted,columns = columns)
        self.particle_values = pd.DataFrame(data = self.swarm.position,columns = columns)
        # add misfit to df
        self.particle_values_converted["misfit"]= self.swarm.current_cost
        self.particle_values["misfit"]= self.swarm.current_cost
        # add LC to df
        self.particle_values_converted["LC"] = self.LC.astype("float32")
        self.particle_values["LC"] = self.LC.astype("float32")
        # add iteration to df
        self.particle_values_converted["iteration"] = i
        self.particle_values["iteration"] =i
        # add particle no to df
        particle_no = np.arange(self.swarm.position_converted.shape[0], dtype = int)
        self.particle_values_converted["particle_no"] = particle_no
        self.particle_values["particle_no"] = particle_no