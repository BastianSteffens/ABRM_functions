############################################################################



####### Import required Packages #######
import pyswarms_modified as ps       # PSO package in Python
# import pyswarms as ps       # PSO package in Python

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pyswarms.utils.plotters import (plot_cost_history, plot_contour, plot_surface)
import random 
import pickle
import os
from os import path
import datetime
from datetime import date
from numpy.random import seed
from numpy.random import rand
import lhsmdu
###### Import required functions #######
# import ABRM_functions
from swarm import swarm
import pathlib

############################################################################

def init():
    
    # from pathlib import Path
    # print(Path(__file__).parent)



    # seed
    # set_seed = random.randint(0,10000000)
    set_seed = 4205825
    random.seed(set_seed)


    ###### Set hyperparmeters for PSO ######
    n_parameters = 30
    n_iters = 30
    n_particles = 30 # always pick multiple of 3. need to fix this 
    min_bound = 0 * np.ones(n_parameters)
    max_bound = 1 * np.ones(n_parameters)
    bounds = (min_bound, max_bound)
    social_component = 1.49618 #1.994 #2.05
    cognitive_component = 1.49618 # 2.05 ; 1.494 with 0 should all converge to the global minima,however good that is. 
    inertia = 0.9
    damping_factor = 0.99
    direction  = 1
    n_neighbors  =  5
    distance_measure  = 2 # 2 = euclidian 1 = manhatten
    dimensions = n_parameters
    options = {'c1': social_component, 'c2': cognitive_component, 'w':inertia, 'k':n_neighbors, 'p':distance_measure,'d':damping_factor, "direction": direction}
    max_velocity = 0.1
    min_velocity = -max_velocity
    velocity_clamp = (min_velocity,max_velocity)
    vh_strategy="invert" # velocity handler
    bh_strategy = "nearest" # boundary handler
    # static: bool
    #         a boolean that decides whether the Ring topology
    #         used is static or dynamic. Default is `False`
    # initialise position with latin hyper cube sampling 
    init_pos = np.array(lhsmdu.sample(numDimensions = n_particles,numSamples = n_parameters,randomSeed = set_seed))

    ###### Set modelling parameters for Petrelworkflows ######
   
    # if I want to set a varaible constant, just make the range = 0 e.g. varmin=varmax
    varminmax = np.array([[1,4],[1,200],[1,200],[1,100],[1,100],[1,7],[1,7],
                          [1,4],[1,200],[1,200],[1,100],[1,100],[1,7],[1,7],
                          [1,4],[1,200],[1,200],[1,100],[1,100],[1,7],[1,7],
                          [0,1],[0,1],[0,1],[1,1500],[1,100],[1,1500],
                          [1,100],[1,1500],[1,100]])
    # varminmax = np.array([[1,4],[1,4],[1,4],[1,200],[1,100],[1,200],[1,100],[1,200],
    #                       [1,100],[1,200],[1,100],[1,200],[1,100],[1,200],
    #                       [1,100],[1,200],[1,100],[1,200],[1,100],[1,200],[1,100],[1,200],[1,100],
    #                       [1,200],[1,100],[1,200],[1,100],[1,1000],[1,100],[1,1000],[1,100],[1,1000],[1,100]])
    # # varminmax = np.array([[0.001,1.5],[4,10],[0.1,3],[2.01,5],[1,100],[0,90],[0,90],[1,100],[0.00075,0.0000075],[0.000015,0.00000015]])    
    nx = 200
    ny = 100
    nz = 7
    n_voronoi = 0 #12
    n_voronoi_zones = 3
    # if continuoes = 0, if discrete = 1
    continuous_discrete = [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0]
    # continuous_discrete = [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0]

    # continuous_discrete = [0,1,0,0,0,0,0,0,0,0]
    # var names
    # columns = ["TI1","TI2","TI3","Voronoi_x_0","Voronoi_y_0","Voronoi_x_1","Voronoi_y_1",
    #            "Voronoi_x_2","Voronoi_y_2","Voronoi_x_3","Voronoi_y_3","Voronoi_x_4",
    #            "Voronoi_y_4","Voronoi_x_5","Voronoi_y_5","Voronoi_x_6","Voronoi_y_6",
    #            "Voronoi_x_7","Voronoi_y_7","Voronoi_x_8","Voronoi_y_8","Voronoi_x_9",
    #            "Voronoi_y_9","Voronoi_x_10","Voronoi_y_10","Voronoi_x_11","Voronoi_y_11","FracpermX",
    #            "MatrixpermX","FracpermY","MatrixpermY","FracpermZ","MatrixpermZ"]
    columns = ["TI1","F1_I_MIN","F1_I_MAX","F1_J_MIN","F1_J_MAX","F1_K_MIN","F1_K_MAX",
               "TI2","F2_I_MIN","F2_I_MAX","F2_J_MIN","F2_J_MAX","F2_K_MIN","F2_K_MAX",
               "TI3","F3_I_MIN","F3_I_MAX","F3_J_MIN","F3_J_MAX","F3_K_MIN","F3_K_MAX",
               "F1_Curve_Prob","F2_Curve_Prob","F3_Curve_Prob","FracpermX","MatrixpermX",
               "FracpermY","MatrixpermY","FracpermZ","MatrixpermZ"]     
    # columns = ["P32","n_sides","elongation_ratio","shape","scale","mean_dip",
    #            "mean_dip_azimuth","concentration","aperture_mean","aperture_std"]
    #  
    
    # var types str = 0, numeric =1, TI = 2, voronoi_coordinate = 3
    # parameter_type = [2,2,2,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,1,1,1,1,1,1]
    parameter_type = [2,1,1,1,1,1,1,2,1,1,1,1,1,1,2,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]
    # parameter_type = [1,1,1,1,1,1,1,1,1,1]

    n_trainingimages = 4
    # misfit values
    # create curve and save resultign desired LC
    Phi_points_target = np.linspace(0, 1, num=11, endpoint=True)
    F_points_target = np.array([0, 0.35, 0.6, 0.7, 0.8, 0.85, 0.90, 0.95, 0.97, 0.99, 1])
    
    # what schedule 5_spot or line_drive
    schedule = "5_spot"

    penalty = "linear"

    # misfit threshold to qualify as suitable model
    best_models = 0.15

    # seed
    set_seed = random.randint(0,10000000)
    random.seed(set_seed)
    # models per petrel workflow (limited to 3 atm)
    n_modelsperbatch = 3
    # how many potrel licenses to run at once
    n_parallel_petrel_licenses = 3
    # which workflow to run in petrel (atm onlz 1 wf)
    runworkflow = "WF_2020_04_16"   #"WF_2020_07_03" #"WF_2020_04_16"#"WF_2019_09_16", "WF_test" "WF_2020_05_08"
    # run with petrel or without for test
    petrel_on = True
    petrel_path = "C:/Program Files/Schlumberger/Petrel 2017/Petrel.exe"

    # if all models should be explicitly saved and not overwritten. 
    save_all_models = True


    setup = dict(varminmax = varminmax, columns = columns, set_seed = set_seed, parameter_type = parameter_type,
                 n_modelsperbatch = n_modelsperbatch, runworkflow = runworkflow, n_iters = n_iters,
                 n_particles = n_particles,n_parameters = n_parameters, Phi_points_target = Phi_points_target,
                 F_points_target =F_points_target, petrel_on = petrel_on,velocity_clamp = velocity_clamp, 
                 save_all_models = save_all_models, vh_strategy=vh_strategy,
                 bh_strategy = bh_strategy, n_parallel_petrel_licenses = n_parallel_petrel_licenses,
                 n_neighbors = n_neighbors,petrel_path = petrel_path, n_trainingimages = n_trainingimages,
                 continuous_discrete = continuous_discrete,schedule = schedule,penalty = penalty, best_models = best_models,
                 nx = nx, ny = ny, nz = nz, n_voronoi = n_voronoi,n_voronoi_zones = n_voronoi_zones,iter_ticker = 0, PSO_parameters = options)

    
    base_path = pathlib.Path(__file__).parent
    output_path = base_path / "../Output/"
    output_folder = str(datetime.datetime.today().replace(microsecond= 0, second = 0).strftime("%Y_%m_%d_%H_%M"))
    output_file_variables = "variable_settings_saved.pickle"
    folder_path = output_path / output_folder
    file_path = folder_path / output_file_variables
    setup["date"] = output_folder
    setup["path"] = file_path
    setup["folder_path"] = folder_path
    setup["base_path"] = base_path
    # print(setup["base_path"])
    #save variables to pickle file and load them into pso later. this also sets up folder structure to save rest of pso resutls in
    # ABRM_functions.save_variables_to_file(setup)
    ###### Initialize swarm ######

    # Call instance of PSO
    optimizer = ps.single.LocalBestPSO(n_particles=n_particles, dimensions=dimensions, options=options, setup = setup,
                                       bounds= bounds, velocity_clamp= velocity_clamp, vh_strategy=vh_strategy,
                                       bh_strategy = bh_strategy,init_pos= init_pos)

    # abrm_swarm = swarm(x_swarm,)
    # Perform optimization
        # cost, pos = optimizer.optimize(ABRM_functions.swarm, iters=n_iters, n_processes= 1)
    cost, pos = optimizer.optimize(swarm, iters=n_iters)
    # Plot the cost
    plot_cost_history(optimizer.cost_history)
    plt.show()

def main():
    init()

if __name__ == "__main__":
    main()


