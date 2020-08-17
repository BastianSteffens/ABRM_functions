
########################
import numpy as np
import pandas as pd
import subprocess
import time
import matlab.engine
import os
from os import path
import datetime
from datetime import date
import pickle
import bz2
import _pickle as cPickle
from scipy import interpolate
from sklearn.metrics import mean_squared_error
from sklearn.neighbors import NearestNeighbors
# from sklearn import preprocessing
import matplotlib.pyplot as plt
import shutil
import umap
import hdbscan
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import re
import pathlib
# from pathlib import Path
import glob
from pyentrp import entropy as ent
from skimage.util.shape import view_as_windows
from GRDECL2VTK import *
from geovoronoi import voronoi_regions_from_coords

import pyvista as pv
########################

def swarm(x_swarm):
 
    """
    Inputs
    ------
    x: numpy.ndarray of shape (n_particles, dimensions)
        The swarm that will perform the search

        Also builts the models for each particle to be looked at. This allows for multiple models to be built at once (cumbersome work in petrel)

    Returns
    -------
    numpy.ndarray of shape (n_particles, )
        The computed loss for each particle
    """

    ### 1 ###
    # convert particle values to values suitable for model building
    x_swarm_converted = convert_particle_values(x_swarm)
    
    ### 2 ###
    # Built new geomodels (with batch files) in Petrel based upon converted particles.
    built_batch_file_for_petrel_models_uniform(x_swarm_converted)

    ### 4 ###
    # built multibat files to run petrel licences in parallel
    built_multibat_files()
    
    ### 5 ###
    # run these batch files to built new geomodels 
    run_batch_file_for_petrel_models(x_swarm_converted)

    ### 6 ###
    # if working with voronoi tesselation for zonation. now its time to patch the previously built models together
    patch_voronoi_models(x_swarm_converted)

    ### 7 ###
    # built FD_Data files required for model evaluation
    built_FD_Data_files()

    ### 8 ###
    # evaluate model performance
    n_particles = x_swarm.shape[0]
    misfit_swarm = np.zeros(n_particles)
    entropy_swarm = np.zeros(n_particles)
    tof_upscaled_entropy_swarm = np.zeros(n_particles)
    # combined_misfit_entropy_swarm = np.zeros(n_particles)
    swarm_performance = pd.DataFrame()
    LC_swarm = np.zeros(n_particles)

    for i in range(n_particles):
        particle_misfit, particle_performance, particle_entropy = particle(x_swarm[i],i) # evaluate
        LC_swarm[i] = particle_performance.LC[0]
        misfit_swarm[i] = particle_misfit 
        entropy_swarm[i] = particle_entropy
        # tof_upscaled_entropy_swarm[i] = particle_tof_upscaled_entropy
        # combined_misfit_entropy_swarm[i] = particle_combined_misfit_entropy
        swarm_performance = swarm_performance.append(particle_performance) # store for data saving
    
    print('swarm misfit {}'.format(misfit_swarm))

    # calculate swarm diversity
    diversity_swarm = compute_diversity_swarm(swarm_performance)

    # save swarm_performance
    save_swarm_performance(swarm_performance)

    # calculate diversity best models
    diversity_best = compute_diversity_best()

    # calculate updated tof upscaled entropy swarm
    # tof_upscaled_entropy_swarm = compute_upscaled_model_entropy()

    # save swarm_particle values
    save_particle_values(x_swarm, x_swarm_converted,misfit_swarm,LC_swarm,entropy_swarm,diversity_swarm,diversity_best)
       
    # save all models
    save_all_models()
    
    return np.array(misfit_swarm)	

def built_batch_file_for_petrel_models_uniform(x):

    # loading in settings that I set up on init_ABRM.py for this run
    base_path = pathlib.Path(__file__).parent
    pickle_file = base_path / "../Output/variable_settings.pickle"
    with open(pickle_file, "rb") as f:
        setup = pickle.load(f)

    seed = setup["set_seed"]
    n_modelsperbatch = setup["n_modelsperbatch"]
    runworkflow = setup["runworkflow"]
    n_particles = setup["n_particles"]
    n_parallel_petrel_licenses = setup["n_parallel_petrel_licenses"]
    parameter_type = setup["parameter_type"]
    parameter_name = setup["columns"]
    petrel_path = setup["petrel_path"]
    n_parameters = len(parameter_name)
    n_trainingimages = setup["n_trainingimages"]
    #  Petrel has problems with batch files that get too long --> if I run
    #  20+ models at once. Therefore split it up in to sets of 3 particles / models
    #  per Petrel license and then run them in parallel. hard on CPU but
    #  shouldnt be too time expensive. Parameter iterations = number of
    #  models.
    particle = x    # all particles together    
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
        path_petrel_projects = base_path / "../Petrel_Projects/ABRM_"
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
        run_petrel_batch = base_path / "../ABRM_functions/batch_files/run_petrel_{}.bat".format(i)

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

def built_multibat_files():
    #### built multi_batch file bat files that can launch several petrel licenses (run_petrel) at once

    # loading in settings that I set up on init_ABRM.py for this run
    base_path = pathlib.Path(__file__).parent
    pickle_file = base_path / "../Output/variable_settings.pickle"
    with open(pickle_file, "rb") as f:
        setup = pickle.load(f)

    n_particles = setup["n_particles"]
    n_modelsperbatch = setup["n_modelsperbatch"]
    n_parallel_petrel_licenses = setup["n_parallel_petrel_licenses"] 

    exit_bat = "\nexit"

    n_multibats = int(np.ceil(n_particles / n_modelsperbatch/n_parallel_petrel_licenses))
    run_petrel_ticker = 0 # naming of petrelfiles to run. problem: will alwazs atm write 3 files into multibatfile.

    for i in range(0,n_multibats):
        built_multibat = r'{}\batch_files\multi_bat_{}.bat'.format(base_path,i)
        file = open(built_multibat, "w+")

        for _j in range(0,n_parallel_petrel_licenses):

            run_petrel_bat = '\nStart {}/batch_files/run_petrel_{}.bat'.format(base_path,run_petrel_ticker)
            file.write(run_petrel_bat)
            run_petrel_ticker+=1

        file.write(exit_bat)
        file.close()

def run_batch_file_for_petrel_models(x):

    # loading in settings that I set up on init_ABRM.py for this run
    base_path = pathlib.Path(__file__).parent
    pickle_file = base_path / "../Output/variable_settings.pickle"    
    with open(pickle_file, "rb") as f:
        setup = pickle.load(f)

    petrel_on = setup["petrel_on"]
    n_particles = setup["n_particles"]
    n_modelsperbatch = setup["n_modelsperbatch"]
    n_parallel_petrel_licenses = setup["n_parallel_petrel_licenses"]    
    lock_files = str(base_path /  "../Petrel_Projects/*.lock")
    kill_petrel =r'{}\batch_files\kill_petrel.bat'.format(base_path)

    if petrel_on == True:
        # initiate model by running batch file make sure that petrel has sufficient time to built the models and shut down again. 
        print(' Start building models')

        #how many multibat files to run
        n_multibats = int(np.ceil(n_particles / n_modelsperbatch/n_parallel_petrel_licenses))
        for i in range(0,n_multibats):
            run_multibat = r'{}\batch_files\multi_bat_{}.bat'.format(base_path,i)
            subprocess.call([run_multibat])
            # not continue until lock files are gone and petrel is finished.
            time.sleep(120)
            kill_timer = 1 # waits 2h before petrel project is shut down if it has a bug 
            while len(glob.glob(lock_files)) >= 1 or kill_timer > 7200:
                kill_timer += 1
                time.sleep(5)
            time.sleep(30)
            subprocess.call([kill_petrel]) # might need to add something that removes lock file here.


        print('Building models complete')

    else:
        print(" dry run - no model building")

def particle(x,i):

    ### 8 ###
    # Objective Function run flow diagnostics
    particle_performance = obj_fkt_FD(i)

    ### 9 ###
    # Compute Performance
    particle_misfit = misfit_fkt_F_Phi_curve(particle_performance["F"],particle_performance["Phi"])
    print('misfit {}'.format(particle_misfit))

    ### 10 ###
    # Compute particle parameter entropy
    particle_entropy = compute_particle_paramter_entropy(x)

    ### 11 ###
    # compute upscaled model entropy
    #particle_tof_upscaled_entropy = compute_upscaled_model_entropy(particle_performance["tof"])

    ### 12 ###
    # Compute entropy and misfit combined
    # particle_combined_misfit_entropy = compute_combined_misfit_entropy(particle_misfit,particle_entropy,particle_tof_upscaled_entropy )

    # store misfit and particle no in dataframe
    particle_performance["particle_no"] = i
    particle_performance["misfit"] = particle_misfit

    return particle_misfit,particle_performance, particle_entropy#,particle_tof_upscaled_entropy#, particle_combined_misfit_entropy

def save_all_models():
    
    # loading in settings that I set up on init_ABRM.py for this run
    base_path = pathlib.Path(__file__).parent
    pickle_file = base_path / "../Output/variable_settings.pickle"    
    with open(pickle_file, "rb") as f:
        setup = pickle.load(f)
    save_models = setup["save_all_models"]
    schedule = setup["schedule"]

    if save_models == True:
        destination_path = setup["folder_path"] + '/all_models/'
        data_path = destination_path + "DATA/"
        include_path = destination_path + "INCLUDE/"
        permx_path = include_path + "PERMX"
        permy_path = include_path + "PERMY"
        permz_path = include_path + "PERMZ"
        poro_path = include_path + "PORO"

        n_particles = setup["n_particles"]
        source_path = str(base_path / "../FD_Models/")
        source_path = "{}\..\FD_Models".format(base_path)

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
            DP_pvt_src_path = source_path + "\INCLUDE\DP_pvt.INC"
            GRID_src_path = source_path + "\INCLUDE\GRID.GRDECL"
            ROCK_RELPERMS_src_path = source_path + "\INCLUDE\ROCK_RELPERMS.INC"
            SCHEDULE_src_path = source_path + "\INCLUDE\{}.INC".format(schedule)
            SOLUTION_src_path = source_path + "\INCLUDE\SOLUTION.INC"
            SUMMARY_src_path = source_path + "\INCLUDE\SUMMARY.INC"

            DP_pvt_dest_path = include_path + "\DP_pvt.INC"
            GRID_dest_path = include_path + "\GRID.GRDECL"
            ROCK_RELPERMS_dest_path = include_path + "\ROCK_RELPERMS.INC"
            SCHEDULE_dest_path = include_path + "\{}.INC".format(schedule)
            SOLUTION_dest_path = include_path + "\SOLUTION.INC"
            SUMMARY_dest_path = include_path + "\SUMMARY.INC"
            
            shutil.copy(DP_pvt_src_path,DP_pvt_dest_path)
            shutil.copy(GRID_src_path,GRID_dest_path)
            shutil.copy(ROCK_RELPERMS_src_path,ROCK_RELPERMS_dest_path)
            shutil.copy(SCHEDULE_src_path,SCHEDULE_dest_path)
            shutil.copy(SOLUTION_src_path,SOLUTION_dest_path)
            shutil.copy(SUMMARY_src_path,SUMMARY_dest_path)

        if os.path.exists(destination_path):

            PERMX_src_path = source_path + "\INCLUDE/PERMX/"
            PERMY_src_path = source_path + "\INCLUDE/PERMY/"
            PERMZ_src_path = source_path + "\INCLUDE/PERMZ/"
            PORO_src_path = source_path + "\INCLUDE/PORO/"

            PERMX_dest_path = include_path + "PERMX/"
            PERMY_dest_path = include_path + "PERMY/"
            PERMZ_dest_path = include_path + "PERMZ/"
            PORO_dest_path = include_path + "PORO/"

            model_id = 0

            for particle_id in range(0,n_particles):

                # set path for Datafile
                data_file_path = data_path + "M{}.DATA".format(model_id)
                while os.path.exists(data_file_path):
                    model_id += 1
                    data_file_path = data_path + "M{}.DATA".format(model_id)

                # print(model_id)
                # print(particle_id)

                # open datafile file to start writing into it / updating it
                built_data_file(data_file_path,model_id)

                #copy and paste permxyz and poro files to new location
                permx_file_src_path = PERMX_src_path + "M{}.GRDECL".format(particle_id)  
                permy_file_src_path = PERMY_src_path + "M{}.GRDECL".format(particle_id)  
                permz_file_src_path = PERMZ_src_path + "M{}.GRDECL".format(particle_id)  
                poro_file_src_path = PORO_src_path + "M{}.GRDECL".format(particle_id)

                permx_file_dest_path = PERMX_dest_path + "M{}.GRDECL".format(model_id)  
                permy_file_dest_path = PERMY_dest_path + "M{}.GRDECL".format(model_id)  
                permz_file_dest_path = PERMZ_dest_path + "M{}.GRDECL".format(model_id)  
                poro_file_dest_path = PORO_dest_path + "M{}.GRDECL".format(model_id) 

                shutil.copy(permx_file_src_path,permx_file_dest_path)
                shutil.copy(permy_file_src_path,permy_file_dest_path)
                shutil.copy(permz_file_src_path,permz_file_dest_path)
                shutil.copy(poro_file_src_path,poro_file_dest_path)


def patch_voronoi_models(x_swarm_converted):

    # loading in settings that I set up on init_ABRM.py for this run
    base_path = pathlib.Path(__file__).parent
    pickle_file = base_path / "../Output/variable_settings.pickle"    
    with open(pickle_file, "rb") as f:
        setup = pickle.load(f)

    n_particles = setup["n_particles"]
    n_voronoi = setup["n_voronoi"]
    n_voronoi_zones = setup["n_voronoi_zones"]
    parameter_type = setup["parameter_type"]
    n_parameters = setup["n_parameters"]
    parameter_name = setup["columns"]
    nx = setup["nx"]
    ny = setup["ny"]
    nz = setup["nz"]
    iter_ticker = setup["iter_ticker"]

    n_neighbors = np.int(n_voronoi /n_voronoi_zones)
    

    for i in range(n_particles):
        # first figure out which points I am interested and append them to new list
        voronoi_x = []
        voronoi_y = []
        # voronoi_z = []
        for j in range(n_parameters):

            # find voronoi positions
            if parameter_type[j] == 3:
                if "x" in parameter_name[j]:
                    voronoi_x_temp = x_swarm_converted[i,j]
                    voronoi_x.append(voronoi_x_temp)
                elif "y" in parameter_name[j]:
                    voronoi_y_temp = x_swarm_converted[i,j]
                    voronoi_y.append(voronoi_y_temp)
                # elif "z" in parameter_name[j]:
                #     voronoi_z_temp = x_swarm_converted[i,j]
                #     voronoi_z.append(voronoi_z_temp)

        # use these points to built a voronoi tesselation
        voronoi_x = np.array(voronoi_x)
        voronoi_y = np.array(voronoi_y)
        voronoi_points = np.vstack((voronoi_x,voronoi_y)).T
        # voronoi_z = np.array(voronoi_z)



        # #define grid and  position initianinon points of n polygons
        grid = Polygon([(0, 0), (0, ny), (nx, ny), (nx, 0)])

        # generate 2D mesh
        x = np.arange(0,nx+1,1,)
        y = np.arange(0,ny+1,1)
        x_grid, y_grid = np.meshgrid(x,y)

        #get cell centers of mesh
        x_cell_center = x_grid[:-1,:-1]+0.5
        y_cell_center = y_grid[:-1,:-1]+0.5

        cell_center_which_polygon = np.zeros(len(x_cell_center.flatten()))

        # array to assign polygon id [ last column] to cell id [first 2 columns]
        all_cell_center = np.column_stack((x_cell_center.flatten(),y_cell_center.flatten(),cell_center_which_polygon))

        # get voronoi regions
        poly_shapes, pts, poly_to_pt_assignments = voronoi_regions_from_coords(voronoi_points, grid,farpoints_max_extend_factor = 30)

        # assign cells to a zone in first iteration stick to taht assingment.
        if iter_ticker == 0:

            # find centroids of vornoi polygons
            voronoi_centroids = []
            for j in range(n_voronoi):
                voronoi_centroids.append(np.array(poly_shapes[j].centroid))
            voronoi_centroids = np.array(voronoi_centroids)

            # assign each vornoi polygone to one of the n voronoi polygon zones with KNN
            knn = NearestNeighbors(n_neighbors= n_neighbors, algorithm='auto',p=2)

            assign_voronoi_zone = np.empty(n_voronoi)
            assign_voronoi_zone[:] = np.nan
            points_to_pick_from = voronoi_centroids
            for j in range(n_voronoi_zones):

                #randomly pick starting point of zone
                init_point = np.random.choice(len(points_to_pick_from))
                # find nearest points for starting point zone
                knn.fit(points_to_pick_from)
                _, indices = knn.kneighbors(points_to_pick_from[init_point].reshape(1,-1))

                for k in range(n_neighbors):    
                    # assing these points to a zone
                    assigner = np.where(voronoi_centroids == points_to_pick_from[indices[0,k]],1,0)
                    assigner = np.sum(assigner,axis = 1)
                    assigner = np.where(assigner ==2)
                    assign_voronoi_zone[assigner[0][0]] = j
                # remove selected points from array to choose from
                points_to_pick_from = np.delete(points_to_pick_from,indices,axis = 0)


            setup["assign_voronoi_zone_" +str(i)] = assign_voronoi_zone
            with open(pickle_file,'wb') as f:
                pickle.dump(setup,f)

        else:
        # load voronoi zone assignemnt
            with open(pickle_file, "rb") as f:
                setup = pickle.load(f)
            assign_voronoi_zone = setup["assign_voronoi_zone_" +str(i)]

        # in what voronoi zone and vornoi polygon do cell centers plot
        for j in range(len(all_cell_center)):
            for voronoi_polygon_id in range(n_voronoi):
                
                polygon = poly_shapes[voronoi_polygon_id]
                cell_id = Point(all_cell_center[j,0],all_cell_center[j,1])
                
                if polygon.intersects(cell_id):
                    all_cell_center[j,2] = assign_voronoi_zone[voronoi_polygon_id]
        
        # for j in range(len(all_cell_center)):
        #     for voronoi_polygon_id in range(n_voronoi):
                
        #         voronoi_polygon = poly_shapes[voronoi_polygon_id]
        #         cell_id = Point(all_cell_center[j,0],all_cell_center[j,1])
                
        #         if voronoi_polygon.intersects(cell_id):
        #             all_cell_center[j,2] = voronoi_polygon_id


        # print("test_ticker: {}".format(test_ticker))
        # plt.scatter(x_cell_center,y_cell_center, c = all_cell_center[:,2])
        # plt.xlim([0,nx])
        # plt.ylim([0,ny])
        # plt.show()
        
        # load and assign correct grdecl files to each polygon zone and patch togetehr to new model
        #output from reservoir modelling
        cell_vornoi_combination = np.tile(all_cell_center[:,2],nz).reshape((nx,ny,nz))
        cell_vornoi_combination_flatten = cell_vornoi_combination.flatten()

        all_model_values_permx = np.zeros((n_voronoi_zones,len(cell_vornoi_combination_flatten)))
        all_model_values_permy = np.zeros((n_voronoi_zones,len(cell_vornoi_combination_flatten)))
        all_model_values_permz = np.zeros((n_voronoi_zones,len(cell_vornoi_combination_flatten)))
        all_model_values_poro = np.zeros((n_voronoi_zones,len(cell_vornoi_combination_flatten)))

        geomodel_path = str(base_path / "../FD_Models/INCLUDE/GRID.grdecl")
        Model = GeologyModel(filename = geomodel_path)
        data_file_path = base_path / "../FD_Models/DATA/M_FD_{}.DATA".format(i)

        for j in range(n_voronoi_zones):
            temp_model_path_permx = base_path / '../FD_Models/INCLUDE/Voronoi/Patch_{}/PERMX/M{}.GRDECL'.format(j,i)
            temp_model_path_permy = base_path / '../FD_Models/INCLUDE/Voronoi/Patch_{}/PERMY/M{}.GRDECL'.format(j,i)
            temp_model_path_permz = base_path / '../FD_Models/INCLUDE/Voronoi/Patch_{}/PERMZ/M{}.GRDECL'.format(j,i)
            temp_model_path_poro = base_path / '../FD_Models/INCLUDE/Voronoi/Patch_{}/PORO/M{}.GRDECL'.format(j,i)
            temp_model_permx = Model.LoadCellData(varname="PERMX",filename=temp_model_path_permx)
            temp_model_permy = Model.LoadCellData(varname="PERMY",filename=temp_model_path_permy)
            temp_model_permz = Model.LoadCellData(varname="PERMZ",filename=temp_model_path_permz)
            temp_model_poro = Model.LoadCellData(varname="PORO",filename=temp_model_path_poro)

            all_model_values_permx[j] = temp_model_permx
            all_model_values_permy[j] = temp_model_permy
            all_model_values_permz[j] = temp_model_permz
            all_model_values_poro[j] = temp_model_poro

        # patch things together
        patch_permx  = []
        patch_permy  = []
        patch_permz  = []
        patch_poro  = []
        for j in range(len(cell_vornoi_combination_flatten)):
            for k in range(n_voronoi_zones):
            
                if cell_vornoi_combination_flatten[j] == k:
                    permx = all_model_values_permx[k,j]
                    permy = all_model_values_permy[k,j]
                    permz = all_model_values_permz[k,j]
                    poro = all_model_values_poro[k,j]
                    patch_permx.append(permx)
                    patch_permy.append(permy)
                    patch_permz.append(permz)
                    patch_poro.append(poro)

        # plt.hist(patch_poro)
        # plt.show()
        # values = np.array(patch_poro).reshape((nx,ny,nz))
        # #     # # grid.plot()
        # grid = pv.UniformGrid()
        # grid.dimensions = np.array(values.shape) + 1
        # grid.origin = (0, 0, 0)  # The bottom left corner of the data set

        # grid.spacing = (1, 1, 1)  # These are the cell sizes along each axis
        # grid.cell_arrays["values"] =values.flatten("K")
        # grid.plot(show_edges=False,notebook = False)
 


        file_permx_beginning = "FILEUNIT\nMETRIC /\n\nPERMX\n"
        permx_file_path = base_path / "../FD_Models/INCLUDE/PERMX/M{}.GRDECL".format(i)
        patch_permx[-1] = "{} /".format(patch_permx[-1])
        with open(permx_file_path,"w+") as f:
            f.write(file_permx_beginning)
            newline_ticker = 0
            for item in patch_permx:
                newline_ticker += 1
                if newline_ticker == 50:
                    f.write("\n")
                    newline_ticker = 0
                f.write("{} ".format(item))
            f.close()

        file_permy_beginning = "FILEUNIT\nMETRIC /\n\nPERMY\n"
        permy_file_path = base_path / "../FD_Models/INCLUDE/PERMY/M{}.GRDECL".format(i)
        patch_permy[-1] = "{} /".format(patch_permy[-1])
        with open(permy_file_path,"w+") as f:
            f.write(file_permy_beginning)
            newline_ticker = 0
            for item in patch_permy:
                newline_ticker += 1
                if newline_ticker == 50:
                    f.write("\n")
                    newline_ticker = 0
                f.write("{} ".format(item))
            f.close()

        file_permz_beginning = "FILEUNIT\nMETRIC /\n\nPERMZ\n"
        permz_file_path = base_path / "../FD_Models/INCLUDE/PERMZ/M{}.GRDECL".format(i)
        patch_permz[-1] = "{} /".format(patch_permz[-1])
        with open(permz_file_path,"w+") as f:
            f.write(file_permz_beginning)
            newline_ticker = 0
            for item in patch_permz:
                newline_ticker += 1
                if newline_ticker == 50:
                    f.write("\n")
                    newline_ticker = 0
                f.write("{} ".format(item))
            f.close()

        file_poro_beginning = "FILEUNIT\nMETRIC /\n\nPORO\n"
        poro_file_path = base_path / "../FD_Models/INCLUDE/PORO/M{}.GRDECL".format(i)
        patch_poro[-1] = "{} /".format(patch_poro[-1])
        with open(poro_file_path,"w+") as f:
            f.write(file_poro_beginning)
            newline_ticker = 0
            for item in patch_poro:
                newline_ticker += 1
                if newline_ticker == 50:
                    f.write("\n")
                    newline_ticker = 0
                f.write("{} ".format(item))
            f.close()

        # file_poro = "FILEUNIT\n\METRIC /\nPORO\n{} /".format(patch_poro)
        # poro_file_path = base_path / "../FD_Models/INCLUDE/PORO/M{}.GRDECL".format(i)
        # file = open(poro_file_path, "w+")
        # file.write(file_poro)\\\\\\\\\\\\\s
        # file.close()

    iter_ticker+=1
    setup["iter_ticker"] = iter_ticker
    with open(pickle_file,'wb') as f:
        pickle.dump(setup,f)


    print ("Voronoi-Patching done")
def built_FD_Data_files():

    # loading in settings that I set up on init_ABRM.py for this run
    base_path = pathlib.Path(__file__).parent
    pickle_file = base_path / "../Output/variable_settings.pickle"    
    with open(pickle_file, "rb") as f:
        setup = pickle.load(f)

    n_particles = setup["n_particles"]
    schedule = setup["schedule"]

    for i in range (0,n_particles+1):
        
        data_file = "RUNSPEC\n\nTITLE\nModel_{}\n\nDIMENS\n--NX NY NZ\n200 100 7 /\n\n--Phases\nOIL\nWATER\n\n--DUALPORO\n--NODPPM\n\n--Units\nMETRIC\n\n--Number of Saturation Tables\nTABDIMS\n1 /\n\n--Maximum number of Wells\nWELLDIMS\n10 100 5 10 /\n\n--First Oil\nSTART\n1 OCT 2017 /\n\n--Memory Allocation\nNSTACK\n100 /\n\n--How many warnings allowed, but terminate after first error\nMESSAGES\n11*5000 1 /\n\n--Unified Output Files\nUNIFOUT\n\n--======================================================================\n\nGRID\n--Include corner point geometry model\nINCLUDE\n'..\INCLUDE\GRID.GRDECL'\n/\n\nACTNUM\n140000*1 /\n\n--Porosity\nINCLUDE\n'..\INCLUDE\PORO\M{}.GRDECL'\n/\n\n--Permeability\nINCLUDE\n'..\INCLUDE\PERMX\M{}.GRDECL'\n/\nINCLUDE\n'..\INCLUDE\PERMY\M{}.GRDECL'\n/\nINCLUDE\n'..\INCLUDE\PERMZ\M{}.GRDECL'\n/\n\n--Net to Gross\nNTG\n140000*1\n/\n\n--Output .INIT file to allow viewing of grid data in post proessor\nINIT\n\n--======================================================================\n\nPROPS\n\nINCLUDE\n'..\INCLUDE\DP_pvt.inc' /\n\nINCLUDE\n'..\INCLUDE\ROCK_RELPERMS.INC' /\n\n--======================================================================\n\nREGIONS\n\nEQLNUM\n140000*1\n/\nSATNUM\n140000*1\n/\nPVTNUM\n140000*1\n/\n\n--======================================================================\n\nSOLUTION\n\nINCLUDE\n'..\INCLUDE\SOLUTION.INC' /\n\n--======================================================================\n\nSUMMARY\n\nINCLUDE\n'..\INCLUDE\SUMMARY.INC' /\n\n--======================================================================\n\nSCHEDULE\n\nINCLUDE\n'..\INCLUDE\{}.INC' /\n\nEND".format(i,i,i,i,i,schedule)  
        data_file_path = base_path / "../FD_Models/DATA/M_FD_{}.DATA".format(i)

        file = open(data_file_path, "w+")
        # write petrelfilepath and licence part into file and seed
        file.write(data_file)

        # close file
        file.close()

def built_data_file(data_file_path,model_id):

    # loading in settings that I set up on init_ABRM.py for this run
    base_path = pathlib.Path(__file__).parent
    pickle_file = base_path / "../Output/variable_settings.pickle"    
    with open(pickle_file, "rb") as f:
        setup = pickle.load(f)

    schedule = setup["schedule"]
    data_file = "RUNSPEC\n\nTITLE\nModel_{}\n\nDIMENS\n--NX NY NZ\n200 100 7 /\n\n--Phases\nOIL\nWATER\n\n--DUALPORO\n--NODPPM\n\n--Units\nMETRIC\n\n--Number of Saturation Tables\nTABDIMS\n1 /\n\n--Maximum number of Wells\nWELLDIMS\n10 100 5 10 /\n\n--First Oil\nSTART\n1 OCT 2017 /\n\n--Memory Allocation\nNSTACK\n100 /\n\n--How many warnings allowed, but terminate after first error\nMESSAGES\n11*5000 1 /\n\n--Unified Output Files\nUNIFOUT\n\n--======================================================================\n\nGRID\n--Include corner point geometry model\nINCLUDE\n'..\INCLUDE\GRID.GRDECL'\n/\n\nACTNUM\n140000*1 /\n\n--Porosity\nINCLUDE\n'..\INCLUDE\PORO\M{}.GRDECL'\n/\n\n--Permeability\nINCLUDE\n'..\INCLUDE\PERMX\M{}.GRDECL'\n/\nINCLUDE\n'..\INCLUDE\PERMY\M{}.GRDECL'\n/\nINCLUDE\n'..\INCLUDE\PERMZ\M{}.GRDECL'\n/\n\n--Net to Gross\nNTG\n140000*1\n/\n\n--Output .INIT file to allow viewing of grid data in post proessor\nINIT\n\n--======================================================================\n\nPROPS\n\nINCLUDE\n'..\INCLUDE\DP_pvt.inc' /\n\nINCLUDE\n'..\INCLUDE\ROCK_RELPERMS.INC' /\n\n--======================================================================\n\nREGIONS\n\nEQLNUM\n140000*1\n/\nSATNUM\n140000*1\n/\nPVTNUM\n140000*1\n/\n\n--======================================================================\n\nSOLUTION\n\nINCLUDE\n'..\INCLUDE\SOLUTION.INC' /\n\n--======================================================================\n\nSUMMARY\n\nINCLUDE\n'..\INCLUDE\SUMMARY.INC' /\n\n--======================================================================\n\nSCHEDULE\n\nINCLUDE\n'..\INCLUDE\{}.INC' /\n\nEND".format(model_id,model_id,model_id,model_id,model_id,schedule)  
    
    file = open(data_file_path, "w+")
    # write petrelfilepath and licence part into file and seed
    file.write(data_file)

    # close file
    file.close()

def save_swarm_performance(swarm_performance):
    
    print("start saving swarm iteration")

    # loading in settings that I set up on init_ABRM.py for this run
    base_path = pathlib.Path(__file__).parent
    pickle_file = base_path / "../Output/variable_settings.pickle"    
    with open(pickle_file, "rb") as f:
        setup = pickle.load(f)

    folder_path = setup["folder_path"]
    output_file_performance = "/swarm_performance_all_iter.csv"
    tof_file = "/tof_all_iter.pbz2"
    tof_file_path = folder_path + tof_file
    file_path = folder_path + output_file_performance

    #take out tof to save as zipped pickle
    tof = swarm_performance[["tof","misfit","particle_no"]]

    #cut down files that I save to every 100th value. thats enough for plotting.
    swarm_performance_short = swarm_performance.iloc[::100,:].copy()

    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    if os.path.exists(file_path):

        swarm_performance_all_iter = pd.read_csv(file_path)
        
        #update iterations
        iteration = swarm_performance_all_iter.iteration.max() + 1
        swarm_performance_short["iteration"] = iteration

        # appends the performance of the swarm at a single iteration to the swarm performance of all previous iterations
        swarm_performance_all_iter = swarm_performance_all_iter.append(swarm_performance_short)
        
        # save again
        swarm_performance_all_iter.to_csv(file_path,index=False)

    else:
        # this should only happen in first loop.

        # number of iterations
        swarm_performance_short["iteration"] = 0

        swarm_performance_short.to_csv(file_path,index=False)

    if os.path.exists(tof_file_path):

        #load compressed pickle file
        data = bz2.BZ2File(tof_file_path,"rb")
        tof_all_iter = cPickle.load(data)

        #update iterations
        iteration = tof_all_iter.iteration.max() + 1
        tof["iteration"] = iteration

        # appends the tof of the swarm at a single iteration to the swarm tof of all previous iterations
        tof_all_iter = tof_all_iter.append(tof)

        # save again
        with bz2.BZ2File(tof_file_path,"w") as f:
            cPickle.dump(tof_all_iter,f)

    else:
        # this should only happen in first loop.

        # number of iterations
        tof["iteration"] = 0

        with bz2.BZ2File(tof_file_path,"w") as f:
            cPickle.dump(tof,f)

def save_particle_values(x_swarm, x_swarm_converted,misfit_swarm,LC_swarm,entropy_swarm,diversity_swarm,diversity_best):

    # loading in settings that I set up on init_ABRM.py for this run
    base_path = pathlib.Path(__file__).parent
    pickle_file = base_path / "../Output/variable_settings.pickle"
    with open(pickle_file, "rb") as f:
        setup = pickle.load(f)

    columns = setup["columns"]
    folder_path = setup["folder_path"]

    particle_values_converted = pd.DataFrame(data = x_swarm_converted, columns= columns)
    particle_values = pd.DataFrame(data = x_swarm, columns= columns)

    # add misfit to df
    particle_values_converted["misfit"]= misfit_swarm
    particle_values["misfit"]= misfit_swarm

    # add particle no to df
    particle_no = np.arange(x_swarm_converted.shape[0], dtype = int)
    particle_values_converted["particle_no"] = particle_no
    particle_values["particle_no"] = particle_no

    # add LC to df
    particle_values_converted["LC"] = LC_swarm
    particle_values["LC"] = LC_swarm

    # add entropy to df
    particle_values_converted["entropy_swarm"] = entropy_swarm
    particle_values["entropy_swarm"] = entropy_swarm

    # add diversity_swarm to df
    particle_values_converted["diversity_swarm"] = diversity_swarm
    particle_values["diversity_swarm"] = diversity_swarm

    # add diversity of best models to df
    particle_values_converted["diversity_best"] = diversity_best
    particle_values["diversity_best"] = diversity_best

    # add combined entropy_misfit to df
    # particle_values_converted["combined_misfit_entropy_swarm"] = combined_misfit_entropy_swarm
    # particle_values["combined_misfit_entropy_swarm"] = combined_misfit_entropy_swarm

    # filepath setup
    output_file_partilce_values_converted = "/swarm_particle_values_converted_all_iter.csv"
    output_file_partilce_values = "/swarm_particle_values_all_iter.csv"

    file_path_particles_values_converted = folder_path + output_file_partilce_values_converted
    file_path_particles_values = folder_path + output_file_partilce_values

    # check if folder exists
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    if os.path.exists(file_path_particles_values_converted):

        swarm_particle_values_converted_all_iter = pd.read_csv(file_path_particles_values_converted)
        swarm_particle_values_all_iter = pd.read_csv(file_path_particles_values)

        #update iterations
        iteration = swarm_particle_values_converted_all_iter.iteration.max() + 1
        particle_values_converted["iteration"] = iteration
        particle_values["iteration"] = iteration

        # appends the performance of the swarm at a single iteration to the swarm performance of all previous iterations
        swarm_particle_values_converted_all_iter = swarm_particle_values_converted_all_iter.append(particle_values_converted)
        swarm_particle_values_all_iter = swarm_particle_values_all_iter.append(particle_values)

        # save again
        swarm_particle_values_converted_all_iter.to_csv(file_path_particles_values_converted,index=False)
        swarm_particle_values_all_iter.to_csv(file_path_particles_values,index=False)

    else:
        # this should only happen in first loop.

        # number of iterations
        particle_values_converted["iteration"] = 0
        particle_values["iteration"] = 0

        particle_values_converted.to_csv(file_path_particles_values_converted,index=False) 
        particle_values.to_csv(file_path_particles_values,index=False) 

def obj_fkt_FD(x):

    eng = matlab.engine.start_matlab()

    # run matlab and mrst
    eng.matlab_starter(nargout = 0)
    # print('sucessfully launched MRST.')

    # run FD and output dictionary
    FD_data = eng.FD_BS(x)
    # print('sucessfully ran FD.')
    # split into Ev tD F Phi and LC and tof column
    FD_data = np.array(FD_data._data).reshape((6,len(FD_data)//6))
    FD_performance = pd.DataFrame()
    
    FD_performance["EV"] = FD_data[0]
    FD_performance["tD"] = FD_data[1]
    FD_performance["F"] = FD_data[2]
    FD_performance["Phi"] = FD_data[3]
    FD_performance["LC"] = FD_data[4]
    FD_performance["tof"] = FD_data[5]

    return FD_performance

def convert_particle_values(x):

    # loading in settings that I set up on init_ABRM.py for this run
    base_path = pathlib.Path(__file__).parent
    pickle_file = base_path / "../Output/variable_settings.pickle"
    with open(pickle_file, "rb") as f:
        setup = pickle.load(f)

    varminmax = setup["varminmax"]
    n_particles = setup["n_particles"]
    parameter_name = setup["columns"]
    continuous_discrete = setup["continuous_discrete"]
    n_parameters = len(parameter_name)

    ### 1 ###
    # turn discrete hyperparameters from continuous back into discrete hyperparameters. Therefore have to set up the boundaries for each parameter
        #                       type          range    position
    # windowsize:           discrete     3 - 21    x[0]

    converted_vals_range = varminmax
    # transpose particle values to make value conversion easier
    x_t = x.T

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
    converted_particle_values = np.array(converted_vals.T)
    # print(converted_particle_values)

    # swap around parameters that work together and where min requires to be bigger than max. ohterwise wont do anzthing in petrel workflow.
    for i in range(0,n_particles):

        for j in range(0,n_parameters):
            match_min = re.search("min",parameter_name[j],re.IGNORECASE)

            if match_min:
                match_max = re.search("max",parameter_name[j+1],re.IGNORECASE)

                if match_max:
                    if converted_particle_values[i,j] > converted_particle_values[i,j+1]:
                        converted_particle_values[i,j],converted_particle_values[i,j+1] = converted_particle_values[i,j+1],converted_particle_values[i,j] 
  
    return converted_particle_values

def misfit_fkt(x):
    
    # loading in settings that I set up on init_ABRM.py for this run
    base_path = pathlib.Path(__file__).parent
    pickle_file = base_path / "../Output/variable_settings.pickle"
    with open(pickle_file, "rb") as f:
        setup = pickle.load(f)

    LC_target = setup["LC_target"]

    misfit = abs(LC_target - x)

    return misfit

def misfit_fkt_F_Phi_curve(F,Phi):
    
    # loading in settings that I set up on init_ABRM.py for this run
    base_path = pathlib.Path(__file__).parent
    pickle_file = base_path / "../Output/variable_settings.pickle"
    with open(pickle_file, "rb") as f:
        setup = pickle.load(f)
    
    F_points_target = setup["F_points_target"]
    Phi_points_target = setup["Phi_points_target"]
    # interpolate F-Phi curve from imput points with spline
    tck = interpolate.splrep(Phi_points_target,F_points_target, s = 0)
    Phi_interpolated = np.linspace(0,1,num = len(Phi),endpoint = True)
    F_interpolated = interpolate.splev(Phi_interpolated,tck,der = 0) # here can easily get first and second order derr.

    # calculate first order derivate of interpolated F-Phi curve and modelled F-Phi curve
    F_interpolated_first_derr = np.gradient(F_interpolated)
    F_first_derr = np.gradient(F)
    F_interpolated_second_derr = np.gradient(F_interpolated_first_derr)
    F_second_derr = np.gradient(F_first_derr)

    # calculate LC for interpolatd F-Phi curve and modelled F-Phi curve
    LC_interpolated = compute_LC(F_interpolated,Phi_interpolated)
    #print("LC_check")
    #print(LC_interpolated)
    LC = compute_LC(F,Phi)
    #print(LC)
    # calculate rmse for each curve and LC
    rmse_0 = mean_squared_error(F_interpolated,F,squared=False)
    rmse_1 = mean_squared_error(F_interpolated_first_derr,F_first_derr,squared=False)
    rmse_2 = mean_squared_error(F_interpolated_second_derr,F_second_derr,squared=False)
    LC_error = abs(LC-LC_interpolated)

    # calcluate quantile loss for each curve
    # qt_loss_0 = quantile_loss(0.1,F_interpolated,F)
    # qt_loss_1 = quantile_loss(0.1,F_interpolated_first_derr,F_first_derr)
    # qt_loss_2 = quantile_loss(0.1,F_interpolated_second_derr,F_second_derr)

    # calculate misfit - RMSE if i turn squared to True it will calculate MSE
    # misfit = 1/3*rmse_0 + 1/3*rmse_1 + 1/3*rmse_2
    misfit = rmse_0 + rmse_1 + rmse_2 + LC_error

    # plt.plot(Phi,F_interpolated_first_derr,Phi,F_first_derr)
    # plt.show()

    return misfit

def obj_fkt_simple_test(x):

    eng = matlab.engine.start_matlab()
    eng.matlab_starter(nargout=0)

    output = eng.diagnostic_BS(x)
 
    return output

def save_variables_to_file(setup):
    # with help of GUI generate file that contains all the variables that I want to use in my swarm function. save them and them load them into the swarm
    # save these settings to my folder so that I can recreate things within each subfunction
    # aslo set up entire folder structure and check if folders already exist to prevent overwriting
    # type(Path(__file__).resolve())
    # print(Path(__file__).parent)
    # print(type(__file__))
    # print(__file__)
    # BASE_DIR = Path(__file__).resolve().parent.parent
    # print(BASE_DIR)
    # from pathlib import Path

    base_path = str(pathlib.Path(__file__).parent)
    # output_path = str(base_path / "../Output/")
    output_path = base_path + "/" + "../Output/"

    #folder name will be current date and time
    output_folder = str(datetime.datetime.today().replace(microsecond= 0, second = 0).strftime("%Y_%m_%d_%H_%M"))
    output_file_variables = "/variable_settings_saved.pickle"
    folder_path = output_path + "/" + output_folder
    file_path = folder_path + output_file_variables

    setup["date"] = output_folder
    setup["path"] = file_path
    setup["folder_path"] = folder_path

    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        file_path = folder_path + output_file_variables
        with open(file_path,'wb') as f:
            pickle.dump(setup,f)
    
    
    # also save to another file so that will always get overwritten where the filepath stays the same
    file_path_constant = base_path + "/" + "../Output/variable_settings.pickle"
    with open(file_path_constant,'wb') as f:
        pickle.dump(setup,f)

def quantile_loss(q, y, f):
  # q: Quantile to be evaluated, e.g., 0.5 for median.
  # y: True value.
  # f: Fitted (predicted) value.
  e = (y - f)
  return np.mean(np.maximum(q * e, (q - 1) * e))

def compute_LC(F,Phi):
    v = np.diff(Phi,1)
    
    LC = 2*(np.sum(((np.array(F[0:-1]) + np.array(F[1:]))/2*v))-0.5)
    return LC

def compute_particle_paramter_entropy(x):
    # the idea for this function is the following: for each particle, open up the csv file with the particle_parameter values. 
    # add the current particles values to that list and calculate the entropy for each parameter column. 
    # add the entropy for each column up to total entropy
    # the aim is to maximize this value. 
    # have to check if there is a way to combine this maximisation with the minimisation of the rmse. mzaybe take some sort of inverse 
    # Another thing to keep in mind: should I include all values from the proceeding n iterations into that entropy calculations or limit it to n iterations prior?

    # loading in settings that I set up on init_ABRM.py for this run
    base_path = pathlib.Path(__file__).parent
    pickle_file = base_path / "../Output/variable_settings.pickle"
    with open(pickle_file, "rb") as f:
        setup = pickle.load(f)

    columns = setup["columns"]
    folder_path = setup["folder_path"]
    penalty = setup["penalty"]
    n_particles = setup["n_particles"]
    n_parameters = setup["n_parameters"]
    # filepath 
    output_file_partilce_values = "/swarm_particle_values_all_iter.csv"
    file_path = folder_path + output_file_partilce_values

    # check if file exists (after first iteration it should)
    if os.path.exists(file_path):
        
        swarm_particle_values_all_iter = pd.read_csv(file_path)

        # only look at particle values, not at LC,misfit etc.
        swarm_particle_values_all_iter = swarm_particle_values_all_iter[columns].copy()

        # appends the performance of the swarm at a single iteration to the swarm performance of all previous iterations
        x_list = list(x)
        df_x = pd.DataFrame(columns= columns )
        df_x.loc[len(df_x)] = x_list
        swarm_particle_values_all_iter = swarm_particle_values_all_iter.append(df_x, ignore_index = True)
        
        if penalty == "power_2":

            all_parameters_entropy = []
            parameter_entropy = []

            # iterate through columns and calculate entropy power 2
            for i in range(0,len(columns)):
                
                if swarm_particle_values_all_iter.shape[0] <= 100:
                    parameter = np.round(swarm_particle_values_all_iter[columns[i]],2)
                    parameter_entropy = np.power(ent.shannon_entropy(parameter),2)
                    all_parameters_entropy.append(parameter_entropy)
                
                elif swarm_particle_values_all_iter.shape[0] > 100:
                    parameter = np.round(swarm_particle_values_all_iter[columns[i]].iloc[-100:],2)
                    parameter_entropy = np.power(ent.shannon_entropy(parameter),2)
                    all_parameters_entropy.append(parameter_entropy)

        elif penalty == "exponential":

            all_parameters_entropy = []
            parameter_entropy = []

            # iterate through columns and calculate exp(entropy)
            for i in range(0,len(columns)):
                
                if swarm_particle_values_all_iter.shape[0] <= 100:
                    parameter = np.round(swarm_particle_values_all_iter[columns[i]],2)
                    parameter_entropy = np.exp(ent.shannon_entropy(parameter))
                    all_parameters_entropy.append(parameter_entropy)
                
                elif swarm_particle_values_all_iter.shape[0] > 100:
                    parameter = np.round(swarm_particle_values_all_iter[columns[i]].iloc[-100:],2)
                    parameter_entropy = np.exp(ent.shannon_entropy(parameter))
                    all_parameters_entropy.append(parameter_entropy)


        elif penalty == "linear":
            
            all_parameters_entropy = []
            parameter_entropy = []

            # iterate through columns and calculate entropy for last 
            for i in range(0,len(columns)):

                if swarm_particle_values_all_iter.shape[0] <= 100:
                    parameter = np.round(swarm_particle_values_all_iter[columns[i]],2)
                    parameter_entropy = ent.shannon_entropy(parameter)
                    all_parameters_entropy.append(parameter_entropy)
                
                elif swarm_particle_values_all_iter.shape[0] > 100:
                    parameter = np.round(swarm_particle_values_all_iter[columns[i]].iloc[-100:],2)
                    parameter_entropy = ent.shannon_entropy(parameter)
                    all_parameters_entropy.append(parameter_entropy)

        # sum up entropy for that particle
        particle_entropy = np.sum(all_parameters_entropy)
    else:
        # as I am using lating hypercube sampling in the first iteration there should be 0 overlap between the values.
        # therefore maxentropy that is possible for number of particles used times the nubmer of parameters is the entropy
        particle_entropy = n_parameters * np.round(ent.shannon_entropy(np.arange(0,n_particles)),2)

    return particle_entropy

def compute_diversity_swarm(swarm_performance):
    # calculate the entropy of larger blocks of the reservoir model 
    # for a single iteration for the entire swarm.
    # then divide by 1 over swarmsize and n_cell blocks

    # loading in settings that I set up on init_ABRM.py for this run
    base_path = pathlib.Path(__file__).parent
    pickle_file = base_path / "../Output/variable_settings.pickle"
    with open(pickle_file, "rb") as f:
        setup = pickle.load(f)

    folder_path = setup["folder_path"]
    penalty = setup["penalty"]
    n_particles = setup["n_particles"]
    n_parameters = setup["n_parameters"]


    # upscale and calculate mean for upscaled cell, then transpose and append to new df where one column equals one cell.
    window_shape = (10,10,7)
    step_size = 10
    df_upscaled_tof = pd.DataFrame(columns = np.arange(20*10*1))

    for i in range(0,n_particles):
        particle_no = i
        tof_single_particle = np.array(swarm_performance[(swarm_performance.particle_no == particle_no)].tof)
        tof_single_particle_3d = tof_single_particle.reshape((200,100,7))
        tof_single_particle_moving_window = view_as_windows(tof_single_particle_3d, window_shape, step= step_size)
        tof_single_particle_upscaled = []
        for i in range(0,20):
            for j in range(0,10):
                for k in range(0,1):
                    single_cell_temp = np.mean(tof_single_particle_moving_window[i,j,k])
                                        # single_cell_temp = np.round(np.mean(tof_single_particle_moving_window[i,j,k]),2)

                    tof_single_particle_upscaled.append(single_cell_temp)

        df_tof_single_particle_upscaled = pd.DataFrame(np.array(tof_single_particle_upscaled))
        df_tof_single_particle_upscaled_transposed = df_tof_single_particle_upscaled.T
        df_upscaled_tof = df_upscaled_tof.append(df_tof_single_particle_upscaled_transposed)

    all_cells_entropy = []

    # calculate entropy for each column of newly created df
    for i in range(0,df_upscaled_tof.shape[1]):

        # single cell in all models    
        cell = np.array(np.round(df_upscaled_tof[i])).reshape(-1,1)
        # print(cell)
        # scale tof
        # scaler = preprocessing.MinMaxScaler(feature_range=(-1,1))
        # cell = scaler.fit_transform(cell)#.reshape(-1,1))l
        # print(cell_scaled)
        # discretize with the help of HDBSCAN
        # Create HDBSCAN clusters
        hdb = hdbscan.HDBSCAN(min_cluster_size=2, cluster_selection_epsilon = 10000000, min_samples = 2, cluster_selection_method = "leaf")#,min_samples  =1)
        scoreTitles = hdb.fit(cell)
        cell_cluster_id = scoreTitles.labels_
        # unclustered cells get unique value
        for i in range(0,len(cell_cluster_id)):
            if cell_cluster_id[i] == -1:
                cell_cluster_id[i] = np.max(cell_cluster_id) + 1

        parameter_entropy = np.array(ent.shannon_entropy(cell_cluster_id))
        all_cells_entropy.append(parameter_entropy)

    # sum up entropy for that particle
    swarm_upscaled_entropy = np.sum(all_cells_entropy)
    print("swarm_diverstiy")
    print(swarm_upscaled_entropy)
    # diversity
    n_cells = df_upscaled_tof.shape[1]
    diversity_swarm = (1/(n_particles*n_cells)) * swarm_upscaled_entropy
    print(diversity_swarm)

    return diversity_swarm

def compute_diversity_best():
    # calculate entropy of large blocks of best models

    # loading in settings that I set up on init_ABRM.py for this run
    base_path = pathlib.Path(__file__).parent
    pickle_file = base_path / "../Output/variable_settings.pickle"
    with open(pickle_file, "rb") as f:
        setup = pickle.load(f)

    folder_path = setup["folder_path"]
    n_particles = setup["n_particles"]
    best_models = setup["best_models"]

    # filepath 
    tof_all_iter = "/tof_all_iter.pbz2."
    tof_file_path = folder_path + tof_all_iter
 
    # check if file exists (after first iteration it should)
    if os.path.exists(tof_file_path):

        #load compressed pickle file
        data = bz2.BZ2File(tof_file_path,"rb")
        tof_all_iter = cPickle.load(data)

        # if best models exist best models
        tof_best_models_check = tof_all_iter[(tof_all_iter.misfit <= best_models)].tof

        if tof_best_models_check.shape[0] > 140000:
            # print("best modesl exists")
            tof_best_models = tof_all_iter[(tof_all_iter.misfit <= best_models)]
            df_best_tof_upscaled = pd.DataFrame(columns = np.arange(20*10*1))

            iterations = tof_best_models["iteration"].unique().tolist()

            #upscale and calculate mean for upscaled cell, then transpose and append to new df where one column equals one cell.
            window_shape = (10,10,7)
            step_size = 10

            for i in range(0,len(iterations)):
                iteration = iterations[i]
                particle_no = tof_best_models[(tof_best_models.iteration == iteration)].particle_no.unique().tolist()
                for j in range(0,len(particle_no)):
                    particle = particle_no[j]
                    tof_single_particle = np.array(tof_best_models[(tof_best_models.iteration == iteration) & (tof_best_models.particle_no == particle)].tof)
                    tof_single_particle_3d = tof_single_particle.reshape((200,100,7))
                    tof_single_particle_moving_window = view_as_windows(tof_single_particle_3d, window_shape, step= step_size)
                    tof_single_particle_upscaled = []
                    for k in range(0,20):
                        for l in range(0,10):
                            for m in range(0,1):
                                single_cell_temp = np.round(np.mean(tof_single_particle_moving_window[k,l,m]),2)
                                tof_single_particle_upscaled.append(single_cell_temp)

                    df_tof_single_particle_upscaled = pd.DataFrame(np.array(tof_single_particle_upscaled))
                    df_tof_single_particle_upscaled_transposed = df_tof_single_particle_upscaled.T
                    # df_tof_single_particle_upscaled_transposed["particle_no"] = particle
                    # df_tof_single_particle_upscaled_transposed["iteration"] = iteration
                    df_best_tof_upscaled = df_best_tof_upscaled.append(df_tof_single_particle_upscaled_transposed)
            
            
            all_cells_entropy = []
            # calculate entropy for each column of newly created df
            for i in range(0,df_best_tof_upscaled.shape[1]):
            # for i in range(0,10):
                # single cell in all models    
                cell = np.array(df_best_tof_upscaled[i])
                cell = cell.reshape(-1,1)
                # print(cell)
                # scale tof
                # scaler = preprocessing.MinMaxScaler(feature_range=(-1,1))
                # cell = scaler.fit_transform(cell)#.reshape(-1,1))l
            #     print(cell_scaled)
                # discretize with the help of HDBSCAN
                # Create HDBSCAN clusters
                hdb = hdbscan.HDBSCAN(min_cluster_size=2,
                                    min_samples=2, 
                                    cluster_selection_epsilon=10000000,
                                    cluster_selection_method = "leaf")#, cluster_selection_epsilon = 0.1)#,min_samples  =1)
                scoreTitles = hdb.fit(cell)
                cell_cluster_id = scoreTitles.labels_
                
                # unclustered cells get unique value
                for i in range(0,len(cell_cluster_id)):
                    if cell_cluster_id[i] == -1:
                        cell_cluster_id[i] = np.max(cell_cluster_id) + 1
                                
            #     print(cell_cluster_id)

                parameter_entropy = np.array(ent.shannon_entropy(cell_cluster_id))
                all_cells_entropy.append(parameter_entropy)

            # sum up entropy for that particle
            swarm_upscaled_entropy = np.sum(all_cells_entropy)
            print("swarm_diverstiy")
            print(swarm_upscaled_entropy)
            # diversity
            n_cells = df_best_tof_upscaled.shape[1]
            diversity_best = (1/(n_particles*n_cells)) * swarm_upscaled_entropy
            print(diversity_best)

        #     all_cells_entropy = []

        #     # calculate entropy for each column of newly created df
        #     for i in range(0,df_best_tof_upscaled.shape[1]):
                    
        #         cell = np.round(df_best_tof_upscaled[i],2)
        #         parameter_entropy = np.array(ent.shannon_entropy(cell))
        #         all_cells_entropy.append(parameter_entropy)

        #     # sum up entropy for that particle
        #     best_upscaled_entropy = np.sum(all_cells_entropy)
        #     print(best_upscaled_entropy)
        #     # diversity
        #     n_best_models = df_best_tof_upscaled.shape[0]
        #     n_cells = df_best_tof_upscaled.shape[1]
        #     diversity_best = 1/(n_best_models*n_cells) * best_upscaled_entropy
        #     print(diversity_best)
        else:
            diversity_best = 0
    
    return diversity_best


def compute_upscaled_model_entropy():
    # calculate + entropy of larger blocks (of mean or sum or median) (upscaled) 
    # inbetween iterations and then of each model take the similar block, and
    # calculate their entropy. use that as a penalty function. want that to be high.

    # loading in settings that I set up on init_ABRM.py for this run
    base_path = pathlib.Path(__file__).parent
    pickle_file = base_path / "../Output/variable_settings.pickle"
    with open(pickle_file, "rb") as f:
        setup = pickle.load(f)

    folder_path = setup["folder_path"]
    penalty = setup["penalty"]
    n_particles = setup["n_particles"]
    n_parameters = setup["n_parameters"]

    # filepath 
    tof_all_iter = "/tof_all_iter.pbz2."
    tof_file_path = folder_path + tof_all_iter

    # particle_tof = np.array(x)
 
    # check if file exists (after first iteration it should)
    if os.path.exists(tof_file_path):

        #load compressed pickle file
        data = bz2.BZ2File(tof_file_path,"rb")
        tof_all_iter = cPickle.load(data)
        
        #upscale and calculate mean for upscaled cell, then transpose and append to new df where one column equals one cell.
        window_shape = (10,10,7)
        step_size = 10

        df_upscaled_models = pd.DataFrame(columns = np.arange(20*10*1))

        for iteration in range (0,tof_all_iter.iteration.max()+1):
            for particle_no in range(0,tof_all_iter.particle_no.max()+1):
                tof_single_particle = np.array(tof_all_iter[(tof_all_iter.iteration == iteration) & (tof_all_iter.particle_no == particle_no)].tof)
                tof_single_particle_3d = tof_single_particle.reshape((200,100,7))
                tof_single_particle_moving_window = view_as_windows(tof_single_particle_3d, window_shape, step= step_size)
                tof_single_particle_upscaled = []
                for i in range(0,20):
                    for j in range(0,10):
                        for k in range(0,1):
                            single_cell_temp = np.round(np.mean(tof_single_particle_moving_window[i,j,k]),2)
                            tof_single_particle_upscaled.append(single_cell_temp)

                df_tof_single_particle_upscaled = pd.DataFrame(np.log10(np.array(tof_single_particle_upscaled)))
                df_tof_single_particle_upscaled_transposed = df_tof_single_particle_upscaled.T
                df_upscaled_models = df_upscaled_models.append(df_tof_single_particle_upscaled_transposed)

    #    # do the same for new particle
    #     particle_tof_3d = particle_tof.reshape((200,100,7))
    #     particle_tof_moving_window = view_as_windows(particle_tof_3d, window_shape, step = step_size)
    #     particle_tof_upscaled = []
    #     for i in range(0,20):
    #         for j in range(0,10):
    #             for k in range(0,1):
    #                 single_cell_temp = np.round((np.mean(particle_tof_moving_window[i,j,k]),2))
    #                 particle_tof_upscaled.append(single_cell_temp)
                    
    #     df_particle_tof_upscaled = pd.DataFrame(np.log10(np.array(particle_tof_upscaled)))
    #     df_particle_tof_upscaled_transposed = df_particle_tof_upscaled.T
    #     df_upscaled_models = df_upscaled_models.append(df_particle_tof_upscaled_transposed)


        if penalty == "exponential":
            all_cells_entropy = []

            # calculate entropy for each column of newly created df
            for i in range(0,df_upscaled_models.shape[1]):
                    
                cell = np.round(df_upscaled_models[i],2)
                parameter_entropy = np.exp(np.array(ent.shannon_entropy(cell)))
                all_cells_entropy.append(parameter_entropy)

        elif penalty == "linear":
            all_cells_entropy = []

            # calculate entropy for each column of newly created df
            for i in range(0,df_upscaled_models.shape[1]):
                    
                cell = np.round(df_upscaled_models[i],2)
                parameter_entropy = np.array(ent.shannon_entropy(cell))
                all_cells_entropy.append(parameter_entropy)
        # sum up entropy for that particle
        swarm_upscaled_entropy = np.sum(all_cells_entropy)

    else:
    # as I am using lating hypercube sampling in the first iteration there should be 0 overlap between the values.
    # therefore maxentropy that is possible for number of particles used times the nubmer of parameters is the entropy
        swarm_upscaled_entropy = n_parameters * np.round(ent.shannon_entropy(np.arange(0,n_particles)),2)


    print("upscaled entropy")
    print(swarm_upscaled_entropy)
    return swarm_upscaled_entropy
 
def scale_min_max(X,xmin, xmax, lmin=0, lmax=1):
    """Scale every value of a list between lmin and lmax"""
    dx = xmin - xmax
    dl = lmin - lmax
    L = lmax-((xmax-X)*dl/dx)

    return(L)

def compute_combined_misfit_entropy(particle_misfit,particle_entropy):

    # loading in settings that I set up on init_ABRM.py for this run
    base_path = pathlib.Path(__file__).parent
    pickle_file = base_path / "../Output/variable_settings.pickle"
    with open(pickle_file, "rb") as f:
        setup = pickle.load(f)

    # n_particles = setup["n_particles"]
    # n_iterations = setup["n_iterations"]
    n_parameters = setup["n_parameters"]
    penalty = setup["penalty"]

    if penalty == "power_2":
        # uniform sampling should give me the maximum possible sampling. with np.round(x,2) that is around 6.65
        max_entropy_parameter = ent.shannon_entropy(np.arange(0,1000))
        max_entropy_parameter = np.power(max_entropy_parameter,2)
        min_entropy_parameter = 0

        max_entropy_particle = max_entropy_parameter * 200 #n_parameters
        min_entropy_particle = min_entropy_parameter * 200 #n_parameters

    elif penalty == "exponential":
        # uniform sampling should give me the maximum possible sampling. with np.round(x,2) that is around 6.65
        max_entropy_parameter = ent.shannon_entropy(np.arange(0,1000))
        max_entropy_parameter = np.exp(max_entropy_parameter)
        min_entropy_parameter = 0

        max_entropy_particle = max_entropy_parameter *200 # n_parameters
        min_entropy_particle = min_entropy_parameter * 200 #n_parameters

    elif penalty == "linear":
        # uniform sampling should give me the maximum possible sampling. with np.round(x,2) that is around 6.65
        max_entropy_parameter = ent.shannon_entropy(np.arange(0,1000))
        min_entropy_parameter = 0

        max_entropy_particle = max_entropy_parameter *200 # n_parameters
        min_entropy_particle = min_entropy_parameter * 200 #n_parameters


    # scale particle_entropy
    # particle_entropy_scaled = scale_min_max(swarm_upscaled_entropy,xmin = min_entropy_particle, xmax = max_entropy_particle)
    particle_entropy_scaled_inverted = 1-particle_entropy_scaled

    # combine misfit with scaled inverted entropy
    combined_misfit_entropy =particle_entropy_scaled_inverted + particle_misfit 
    return combined_misfit_entropy

### Postprocessing functions ### 

def read_data(data_to_process):
    
    base_path = pathlib.Path(__file__).parent

    # how many different datasets:
    n_datasets = len(data_to_process)

    # df to store all datasets in
    df_position = pd.DataFrame()
    df_performance = pd.DataFrame()
    df_tof = pd.DataFrame()
    setup_all = dict()
    FD_targets = dict()
    Phi_interpolated = []
    F_interpolated = []
    LC_interpolated = []
    
    # read individual datasets and conconate them to one big df. 
    for i in range(0,n_datasets):
        path = str(base_path / "../Output/")+ "/"+ data_to_process[i] + "/"
        performance = "swarm_performance_all_iter.csv"
        position = "swarm_particle_values_converted_all_iter.csv"
        setup = "variable_settings_saved.pickle"
        tof = "tof_all_iter.pbz2"

        performance_path = path + performance
        position_path = path + position
        setup_path = path + setup
        tof_path = path + tof

        # load data
        df_performance_single = pd.read_csv(performance_path)
        df_position_single = pd.read_csv(position_path)
        with open(setup_path,'rb') as f:
            setup_single = pickle.load(f)      
        
        #load compressed pickle file
        data = bz2.BZ2File(tof_path,"rb")
        df_tof_single = cPickle.load(data)
  
        
        df_position_single["dataset"] = data_to_process[i]
        df_performance_single["dataset"] = data_to_process[i]
        df_tof_single["dataset"] = data_to_process[i]
        
        # get F Phi curve and LC for interpolation
        Phi_points_target = setup_single["Phi_points_target"]
        F_points_target = setup_single["F_points_target"]
        
        # interpolate F-Phi curve from input points with spline
        tck = interpolate.splrep(Phi_points_target,F_points_target, s = 0)
        Phi_interpolated = np.linspace(0,1,num = 
                                   len(df_performance_single.loc[(df_performance_single.iteration == 0) 
                                    & (df_performance_single.particle_no == 0), "Phi"]),
                                    endpoint = True)
        F_interpolated = interpolate.splev(Phi_interpolated,tck,der = 0)
        LC_interpolated = compute_LC(F_interpolated,Phi_interpolated)
                
        # Concate all data together
        df_position = df_position.append(df_position_single)
        df_performance = df_performance.append(df_performance_single)
        df_tof = df_tof.append(df_tof_single)
        setup_all[data_to_process[i]] = setup_single
        FD_targets[data_to_process[i]] = dict(Phi_interpolated = Phi_interpolated,F_interpolated =F_interpolated,LC_interpolated = LC_interpolated)

        # uniform index
        df_position.reset_index(drop = True,inplace = True)
        df_performance.reset_index(drop = True,inplace = True)
        df_tof.reset_index(drop = True, inplace = True)
    
    return df_performance,df_position,df_tof,setup_all,FD_targets

def best_model_selection_UMAP_HDBSCAN(df_position,df_tof,cluster_parameter,setup_all,dataset,n_neighbors,min_cluster_size,misfit_tolerance,use_UMAP = True):
    # at some point make n_eighbours and min cluster size nad misfit tolernaze sliding scales
    # model parameters that generate lowest misfit
    if cluster_parameter == "particle_parameters":
        # particle_parameters used for clustering
        columns = setup_all[dataset[0]]["columns"]
        df_best =df_position[(df_position.misfit <= misfit_tolerance)].copy()
        df_best_for_clustering = df_best[columns].copy()
        df_best_for_clustering["LC"] = df_best.LC.copy()

    elif cluster_parameter == "tof":
        #cluster based upon tof.
        #upscale and calculate mean for upscaled cell, then transpose and append to new df where one column equals one cell.

        window_shape = (1,1,1)
        step_size = 1

        # df_best_index =df_position[(df_position.misfit <= misfit_tolerance)].copy()
        # index = df_best_index.index
        df_best_for_clustering = pd.DataFrame(columns = np.arange(200*100*7))
        df_best_temp =df_tof[(df_tof.misfit <= misfit_tolerance)].copy()
        iterations = df_best_temp["iteration"].unique().tolist()

        for i in range(0,len(iterations)):
            iteration = iterations[i]
            particle_no = df_best_temp[ df_best_temp.iteration == iteration].particle_no.unique().tolist()
            for j in range(0,len(particle_no)):
                particle = particle_no[j]
                tof_single_particle = np.array(df_best_temp[(df_best_temp.iteration == iteration) & (df_best_temp.particle_no == particle)].tof)
                tof_single_particle_3d = tof_single_particle.reshape((200,100,7))
                tof_single_particle_moving_window = view_as_windows(tof_single_particle_3d, window_shape, step= step_size)
                tof_single_particle_upscaled = []
                for k in range(0,200):
                    for l in range(0,100):
                        for m in range(0,7):
                            single_cell_temp = np.round(np.mean(tof_single_particle_moving_window[k,l,m]),2)
                            tof_single_particle_upscaled.append(single_cell_temp)

                df_tof_single_particle_upscaled = pd.DataFrame(np.log10(np.array(tof_single_particle_upscaled)))
                df_tof_single_particle_upscaled_transposed = df_tof_single_particle_upscaled.T
                df_tof_single_particle_upscaled_transposed["particle_no"] = particle
                df_tof_single_particle_upscaled_transposed["iteration"] = iteration
                df_best_for_clustering = df_best_for_clustering.append(df_tof_single_particle_upscaled_transposed)

        df_best = pd.DataFrame()
        df_best =df_position[(df_position.misfit <= misfit_tolerance)].copy()

        # df_best = df_best_for_clustering.copy()
        # df_best.index = index
        df_best_for_clustering.drop(columns = ["particle_no","iteration"], inplace = True)



    if use_UMAP == True:

        # Create UMAP reducer
        reducer    = umap.UMAP(n_neighbors=n_neighbors)
        embeddings = reducer.fit_transform(df_best_for_clustering)

        # Create HDBSCAN clusters
        hdb = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size)
        scoreTitles = hdb.fit(embeddings)

        df_best["cluster_prob"] = scoreTitles.probabilities_
        df_best["cluster"] = scoreTitles.labels_
        df_best["cluster_x"] = embeddings[:,0]
        df_best["cluster_y"] = embeddings[:,1]


        fig = go.Figure(data=go.Scatter(x = embeddings[:,0],
                                        y = embeddings[:,1],

                                        mode='markers',
                                        text = df_best.index,
                                        marker=dict(
                                            size=16,
                                            color=df_best.cluster, #set color equal to a variable
                                            colorscale= "deep",#'Viridis', # one of plotly colorscales
                                            showscale=True,
                                            colorbar=dict(title="Clusters")
                                            )
                                        ))
        fig.update_layout(title='Clustering of {} best models - Number of clusters found: {} - Unclustered models: {}'.format(df_best.shape[0],df_best.cluster.max()+1,abs(df_best.cluster[df_best.cluster == -1].sum())))
        fig.show()

    else:

        # Create UMAP reducer
        reducer    = umap.UMAP(n_neighbors=n_neighbors,min_dist = 0)
        embeddings = reducer.fit_transform(df_best_for_clustering)

        # Create HDBSCAN clusters
        hdb = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size)
        scoreTitles = hdb.fit(df_best_for_clustering)

        df_best["cluster_prob"] = scoreTitles.probabilities_
        df_best["cluster"] = scoreTitles.labels_
        df_best["cluster_x"] = embeddings[:,0]
        df_best["cluster_y"] = embeddings[:,1]

        fig = go.Figure(data=go.Scatter(x = embeddings[:,0],
                                        y = embeddings[:,1],

                                        mode='markers',
                                        text = df_best.index,
                                        marker=dict(
                                            size=16,
                                            color=df_best.cluster, #set color equal to a variable
                                            colorscale= "deep",#'Viridis', # one of plotly colorscales
                                            showscale=True,
                                            colorbar=dict(title="Clusters")
                                            )
                                        ))
        fig.update_layout(title='Clustering of {} best models - Number of clusters found: {} - Unclustered models: {}'.format(df_best.shape[0],df_best.cluster.max()+1,abs(df_best.cluster[df_best.cluster == -1].sum())))
        fig.show()
    
    return df_best

def plot_hist(df,setup_all,dataset, misfit_tolerance = None):
    
    columns = setup_all[dataset[0]]["columns"]

    cols_range = [1,2,3]

    n_cols = int(len(cols_range))
    n_rows = int(np.ceil(len(columns)/n_cols))

    cols = cols_range* n_rows 
    len_row  = list(np.arange(1,n_rows+1,1))
    rows = sorted(n_cols*len_row)

    for i in range(0,len(rows)):
        rows[i]=rows[i].item()
        
    n_subplots = len(columns)/n_rows/n_cols

    fig = make_subplots(rows=n_rows, cols=n_cols, subplot_titles=(columns))

    if misfit_tolerance is not None:

        df_best =df[(df.misfit <= misfit_tolerance)]
        df_best = df_best[columns]

        for i in range(0,len(columns)):
            # fig.append_trace(go.Histogram(x=df[columns[i]]),row = rows[i],col = cols[i])

            fig.append_trace(go.Histogram(x=df_best[columns[i]]),row = rows[i],col = cols[i])

            fig.update_layout(
                    showlegend=False,
                    barmode='overlay'        # Overlay both histograms
                    )
            fig.update_traces(opacity = 0.75) # Reduce opacity to see both histograms


        fig.update_layout(autosize=False,
            title= "Histogram Parameters",
            width=1000,
            height=750*(n_subplots)
        )
        fig.show()

    else:

        for i in range(0,len(columns)):

            fig.append_trace(go.Histogram(x=df[columns[i]]),row = rows[i],col = cols[i])
            fig.update_layout(
                    showlegend=False
                    )
        fig.update_layout(autosize=False,
            title= "Histogram Parameters",
            width=1000,
            height=750*(n_subplots)
        )
        fig.show()

def plot_box(df,setup_all,dataset):

    max_iters = df.iteration.max() + 1
    columns = setup_all[dataset[0]]["columns"]
    misfit = ["misfit"]
    misfit.extend(columns)
    columns = misfit.copy()

    for j in range(0,len(columns)):
        fig = go.Figure()
        for i in range (0,max_iters):
            fig.add_trace(go.Box(y=df[df.iteration ==i][columns[j]],name="Iteration {}".format(i)))

        fig.update_layout(
            title= columns[j],
            showlegend=False
        )
        fig.show()
    #dropdown with the parameter taht I would like to change.

def save_best_clustered_models(df_best,datasets):
    

    n_clusters = df_best.cluster.max()
    
    best_models_to_save = pd.DataFrame()

    base_path = pathlib.Path(__file__).parent

    # randomly sample 1 model from each lcusters that has a perfect match to the cluster (prob =1)
    for i in range(0,n_clusters+1):
        best_per_cluster = df_best[(df_best.cluster == i) & (df_best.cluster_prob ==1)].sample()
        best_models_to_save = best_models_to_save.append(best_per_cluster)

    # get models that arent clustered and append them
    if -1 is not df_best.cluster:
        best_models_to_save = best_models_to_save.append(df_best[df_best.cluster == -1])

    # save best models
    n_datasets = len(datasets)

    for i in range(0,n_datasets):

        # open df_position to figure out which models performed best
        path = str(base_path / "../Output/") + "/" + datasets[i] + "/"

        n_best_models = len(best_models_to_save)
        best_models_index = best_models_to_save.index.tolist()

        #path to all models 
        all_path = path + "all_models/"
        data_all_path = all_path + "DATA/"
        include_all_path = all_path + "INCLUDE/"
        permx_all_path = include_all_path + "PERMX/"
        permy_all_path = include_all_path + "PERMY/"
        permz_all_path = include_all_path + "PERMZ/"
        poro_all_path = include_all_path + "PORO/"

        #path to best models 
        destination_best_path = path + "best_models/"
        data_best_path = destination_best_path + "DATA/"
        include_best_path = destination_best_path + "INCLUDE/"
        permx_best_path = include_best_path + "PERMX/"
        permy_best_path = include_best_path + "PERMY/"
        permz_best_path = include_best_path + "PERMZ/"
        poro_best_path = include_best_path + "PORO/"

        if not os.path.exists(destination_best_path):
            # make folders and subfolders
            os.makedirs(destination_best_path)
            os.makedirs(data_best_path)
            os.makedirs(include_best_path)
            os.makedirs(permx_best_path)
            os.makedirs(permy_best_path)
            os.makedirs(permz_best_path)
            os.makedirs(poro_best_path)

            # save best position df as csv. the selected ones and all best models
            best_position_path = destination_best_path + "best_models_selected.csv"
            best_position_path_2 = destination_best_path + "best_models.csv"
            best_models_to_save.to_csv(best_position_path,index=False)
            df_best.to_csv(best_position_path_2,index = False)

            #cop and paste generic files into Data
            DP_pvt_all_path = include_all_path + "DP_pvt.INC"
            GRID_all_path = include_all_path + "GRID.GRDECL"
            ROCK_RELPERMS_all_path = include_all_path + "ROCK_RELPERMS.INC"
            SCHEDULE_all_path = include_all_path + "5_spot.INC"
            SOLUTION_all_path = include_all_path + "SOLUTION.INC"
            SUMMARY_all_path = include_all_path + "SUMMARY.INC"

            DP_pvt_best_path = include_best_path + "DP_pvt.INC"
            GRID_best_path = include_best_path + "GRID.GRDECL"
            ROCK_RELPERMS_best_path = include_best_path + "ROCK_RELPERMS.INC"
            SCHEDULE_best_path = include_best_path + "5_spot.INC"
            SOLUTION_best_path = include_best_path + "SOLUTION.INC"
            SUMMARY_best_path = include_best_path + "SUMMARY.INC"

            shutil.copy(DP_pvt_all_path,DP_pvt_best_path)
            shutil.copy(GRID_all_path,GRID_best_path)
            shutil.copy(ROCK_RELPERMS_all_path,ROCK_RELPERMS_best_path)
            shutil.copy(SCHEDULE_all_path,SCHEDULE_best_path)
            shutil.copy(SOLUTION_all_path,SOLUTION_best_path)
            shutil.copy(SUMMARY_all_path,SUMMARY_best_path)

        if os.path.exists(destination_best_path):
            #copy and paste best models 
            for i in range(0,n_best_models):
                
                data_file_all_path = data_all_path + "M{}.DATA".format(best_models_index[i])  
                permx_file_all_path = permx_all_path + "M{}.GRDECL".format(best_models_index[i])  
                permy_file_all_path = permy_all_path + "M{}.GRDECL".format(best_models_index[i])  
                permz_file_all_path = permz_all_path + "M{}.GRDECL".format(best_models_index[i])  
                poro_file_all_path = poro_all_path + "M{}.GRDECL".format(best_models_index[i])

                data_file_best_path = data_best_path + "M{}.DATA".format(best_models_index[i])  
                permx_file_best_path = permx_best_path + "M{}.GRDECL".format(best_models_index[i])  
                permy_file_best_path = permy_best_path + "M{}.GRDECL".format(best_models_index[i])  
                permz_file_best_path = permz_best_path + "M{}.GRDECL".format(best_models_index[i])  
                poro_file_best_path = poro_best_path + "M{}.GRDECL".format(best_models_index[i]) 

                shutil.copy(data_file_all_path,data_file_best_path)
                shutil.copy(permx_file_all_path,permx_file_best_path)
                shutil.copy(permy_file_all_path,permy_file_best_path)
                shutil.copy(permz_file_all_path,permz_file_best_path)
                shutil.copy(poro_file_all_path,poro_file_best_path)

                # save best position df as csv. the selected ones and all best models
                best_position_path = destination_best_path + "best_models_selected.csv"
                best_position_path_2 = destination_best_path + "best_models.csv"
                best_models_to_save.to_csv(best_position_path,index=False)
                df_best.to_csv(best_position_path_2,index = False)
        
    return best_models_to_save

def plot_performance(df_performance,df_position,FD_targets,setup_all,dataset,misfit_tolerance):
    # Create traces

    fig = make_subplots(rows = 2, cols = 2,
                       subplot_titles = ("Misfit","LC plot","F - Phi Graph","Sweep Efficieny Graph"))

    ### Misfit ###
    
    fig.add_trace(go.Scatter(x = df_position.index[(df_position.dataset == dataset[0])], y=df_position.misfit[(df_position.dataset == dataset[0])],
                            mode='markers',
                            line = dict(color = "black"),
                            name='misfit'),row =1, col =1)
    
    fig.add_trace(go.Scatter( x= df_position.index[(df_position.misfit <= misfit_tolerance) & (df_position.dataset == dataset[0])],y=df_position.loc[(df_position.misfit <= misfit_tolerance) & (df_position.dataset == dataset[0]),"misfit"],
                            mode = "markers",
                            line = dict(color = "magenta")))

    fig.update_xaxes(range = [0,df_position.index.max()],row =1, col =1)
    fig.update_yaxes(range = [0,1], row =1, col = 1)

    ### LC plot ###
    
    fig.add_trace(go.Scatter(x = df_position.index, y=df_position.LC,
                            mode='markers',
                            line = dict(color = "lightgray"),
                            name='Simulated'),row =1, col =2)
    fig.add_trace(go.Scatter( x= df_position.index[(df_position.misfit <= misfit_tolerance)],y=df_position.loc[(df_position.misfit <= misfit_tolerance,"LC")],
                                mode = "markers",
                            line = dict(color = "magenta")),row =1, col =2)
    
    fig.add_shape(
            # Line Horizontal
                type="line",
                x0=0,
                y0=FD_targets[dataset[0]]["LC_interpolated"], # make date a criterion taht can be changed
                x1=df_position.index.max(),
                y1=FD_targets[dataset[0]]["LC_interpolated"],
                line=dict(
                    color="red",
                    width=2),row =1, col = 2)

    fig.update_xaxes(title_text = "particles",row = 1, col = 1)
    fig.update_yaxes(title_text = "RMSE",row = 1, col = 1)
    fig.update_xaxes(title_text = "particles",range = [0,df_position.index.max()],row =1, col = 2)
    fig.update_yaxes(title_text = "LC",range = [0,1], row =1, col = 2)

    ### F - Phi plot ###
    
    fig.add_trace(go.Scatter(x=df_performance.Phi, y=df_performance.F,
                            mode='lines',
                            line = dict(color = "lightgray"),
                            name='Simulated'),row =2, col =1)

    fig.add_trace(go.Scatter(x = df_performance.loc[(df_performance.misfit <= misfit_tolerance,"Phi")], # make misfit value a criterion that can be changed
                             y = df_performance.loc[(df_performance.misfit <= misfit_tolerance,"F")],
                            mode = "lines",
                            line = dict(color = "magenta"),
                            text = "nothing yet",
                            name = "best simulations"),row =2, col =1)
    
    fig.add_trace(go.Scatter(x = FD_targets[dataset[0]]["Phi_interpolated"], y = FD_targets[dataset[0]]["F_interpolated"],
                            mode = "lines",
                            line = dict(color = "red", width = 3),
                            name = "target"),row =2, col =1)
    
    fig.add_trace(go.Scatter(x = [0,1], y = [0,1],
                            mode = "lines",
                            line = dict(color = "black", width = 3),
                            name = "homogeneous"),row =2, col =1)

    fig.update_xaxes(title_text = "Phi", range = [0,1],row =2, col =1)
    fig.update_yaxes(title_text = "F",range = [0,1], row =2, col = 1)

    ### Sweep efficiency plot ###
    
    for i in range (0,df_performance.iteration.max()):
        iteration = i 
        for j in range(0,df_performance.particle_no.max()):
            particle_no = j
            EV = df_performance[(df_performance.iteration == iteration) & (df_performance.particle_no == particle_no)].EV
            tD = df_performance[(df_performance.iteration == iteration) & (df_performance.particle_no == particle_no)].tD

            fig.add_trace(go.Scatter(x=tD, y=EV,
                                mode='lines',
                                line = dict(color = "lightgray"),
                                text = "nothing yet",
                                name = "Simulated"),row =2, col =2)

    for i in range (0,df_performance.iteration.max()):
        iteration = i 
        for j in range(0,df_performance.particle_no.max()):
            particle_no = j
            EV = df_performance[(df_performance.iteration == iteration) & (df_performance.particle_no == particle_no) & (df_performance.misfit <= misfit_tolerance)].EV
            tD = df_performance[(df_performance.iteration == iteration) & (df_performance.particle_no == particle_no) & (df_performance.misfit <= misfit_tolerance)].tD

            fig.add_trace(go.Scatter(x=tD, y=EV,
                                mode='lines',
                                line = dict(color = "magenta"),
                                text = "nothing yet",
                                name = "best simulations"),row =2, col =2)

    fig.update_xaxes(title_text = "tD", range = [0,1],row =2, col =2)
    fig.update_yaxes(title_text = "Ev",range = [0,1], row =2, col = 2)

    fig.update_layout(title='Performance Evaluation - Simulation run {}'.format(dataset),
                       autosize = False,
                     width = 1000,
                     height = 1000,
                     showlegend = False)

    fig.show()

def plot_tof_hist(df, misfit_tolerance = None):
   
    window_shape = (10,10,7)
    step_size = 10
    iterations = df["iteration"].unique().tolist()
    particle_no = df["particle_no"].unique().tolist()
    df_best_for_clustering = pd.DataFrame(columns = np.arange(20*10*1))


    for i in range(0,len(df.iteration)):
        iteration = iterations[i]
        for j in range(0,len(particle_no)):
            particle = particle_no[j]
            tof_single_particle = np.array(df[(df.iteration == iteration) & (df.particle_no == particle)].tof)
            tof_single_particle_3d = tof_single_particle.reshape((200,100,7))
            tof_single_particle_moving_window = view_as_windows(tof_single_particle_3d, window_shape, step= step_size)
            tof_single_particle_upscaled = []
            for k in range(0,20):
                for l in range(0,10):
                    for m in range(0,1):
                        single_cell_temp = np.round(np.mean(tof_single_particle_moving_window[k,l,m]),2)
                        tof_single_particle_upscaled.append(single_cell_temp)

            df_tof_single_particle_upscaled = pd.DataFrame(np.log10(np.array(tof_single_particle_upscaled)))
            df_tof_single_particle_upscaled_transposed = df_tof_single_particle_upscaled.T
            df_tof_single_particle_upscaled_transposed["particle_no"] = particle
            df_tof_single_particle_upscaled_transposed["iteration"] = iteration
            df_best_for_clustering = df_best_for_clustering.append(df_tof_single_particle_upscaled_transposed)

    columns = list(np.arange(0,df_best_for_clustering.shape[1]))

    cols_range = [1,2,3]

    n_cols = int(len(cols_range))
    n_rows = int(np.ceil(len(columns)/n_cols))

    cols = cols_range* n_rows 
    len_row  = list(np.arange(1,n_rows+1,1))
    rows = sorted(n_cols*len_row)

    for i in range(0,len(rows)):
        rows[i]=rows[i].item()
        
    n_subplots = len(columns)/n_rows/n_cols

    fig = make_subplots(rows=n_rows, cols=n_cols, subplot_titles=(columns))

    if misfit_tolerance is not None:

        df_best =df[(df.misfit <= misfit_tolerance)]
        df_best = df_best[columns]

        for i in range(0,len(columns)):
            # fig.append_trace(go.Histogram(x=df[columns[i]]),row = rows[i],col = cols[i])

            fig.append_trace(go.Histogram(x=df_best[columns[i]]),row = rows[i],col = cols[i])

            fig.update_layout(
                    showlegend=False,
                    barmode='overlay'        # Overlay both histograms
                    )
            fig.update_traces(opacity = 0.75) # Reduce opacity to see both histograms


        fig.update_layout(autosize=False,
            title= "Histogram Parameters",
            width=1000,
            height=750*(n_subplots)
        )
        fig.show()

    else:

        for i in range(0,len(columns)):

            fig.append_trace(go.Histogram(x=df[columns[i]]),row = rows[i],col = cols[i])
            fig.update_layout(
                    showlegend=False
                    )
        fig.update_layout(autosize=False,
            title= "Histogram Parameters",
            width=1000,
            height=750*(n_subplots)
        )
        fig.show()