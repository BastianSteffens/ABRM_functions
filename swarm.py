
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
import matplotlib.pyplot as plt
import shutil
import umap
import hdbscan
import re
import pathlib
import glob
from pyentrp import entropy as ent
from skimage.util.shape import view_as_windows
from GRDECL_file_reader.GRDECL2VTK import *

from geovoronoi import voronoi_regions_from_coords

# import pyvista as pv
########################

class swarm():
    
    def __init__(self,x_swarm,setup,iteration):

        # set base path where data is stored
        self.setup = setup

        # data storage
        self.x_swarm = x_swarm.astype("float32")
        self.iteration = iteration
        self.n_particles = x_swarm.shape[0]
        self.misfit_swarm = np.zeros(self.n_particles)
        self.entropy_swarm = np.zeros(self.n_particles)
        self.tof_upscaled_entropy_swarm = np.zeros(self.n_particles)
        self.swarm_performance = pd.DataFrame()
        self.LC_swarm = np.zeros(self.n_particles)

        print("########################################## starting model evaluation  iteration {}/{} ##########################################".format(self.iteration,self.setup["n_iters"]-1))

        # convert particle values to values suitable for model building
        self.convert_particle_values()
        
        # Built new geomodels (with batch files) in Petrel based upon converted particles.
        self.built_batch_file_for_petrel_models_uniform()

        # built multibat files to run petrel licences in parallel
        self.built_multibat_files()
        
        # run these batch files to built new geomodels 
        self.run_batch_file_for_petrel_models()

        # if working with voronoi tesselation for zonation. now its time to patch the previously built models together
        if self.setup["n_voronoi"] > 0:
            print ("Start Voronoi-Patching         ",end = "\r")
            self.patch_voronoi_models()
            print ("Voronoi-Patching done         ",end = "\r")


        # built FD_Data files required for model evaluation
        self.built_FD_Data_files()

    def swarm_iterator(self):
        for i in range(self.n_particles):
            particle_misfit, particle_performance = self.particle(i)

            self.LC_swarm[i] = particle_performance.LC[0]
            self.misfit_swarm[i] = particle_misfit 

            self.swarm_performance = self.swarm_performance.append(particle_performance) # store for data saving
            
        print('swarm misfit {}                  '.format(np.round(self.misfit_swarm,2)))


        #built df with all the information desired for postprocessing
        self.get_output_dfs()
        
        return np.array(self.misfit_swarm),self.swarm_performance_short,self.tof,self.particle_values,self.particle_values_converted,self.setup

    def get_output_dfs(self):
        ##prepare dfs of whole swarm with output that is ready for postprocessing
        
        # raw data from FD
        self.tof = self.swarm_performance[["tof","misfit","iteration","particle_no"]].copy()
        self.swarm_performance_short = self.swarm_performance.iloc[::100,:].copy()
        
        # converted particles and raw particles tother with simulation outputs
        columns = self.setup["columns"]
        folder_path = self.setup["folder_path"]
        self.particle_values_converted = pd.DataFrame(data = self.x_swarm_converted,columns = columns)
        self.particle_values = pd.DataFrame(data = self.x_swarm,columns = columns)
        # add misfit to df
        self.particle_values_converted["misfit"]= self.misfit_swarm
        self.particle_values["misfit"]= self.misfit_swarm
        # add LC to df
        self.particle_values_converted["LC"] = self.LC_swarm.astype("float32")
        self.particle_values["LC"] = self.LC_swarm.astype("float32")
        # add iteration to df
        self.particle_values_converted["iteration"] = self.iteration
        self.particle_values["iteration"] = self.iteration
        # add particle no to df
        particle_no = np.arange(self.x_swarm_converted.shape[0], dtype = int)
        self.particle_values_converted["particle_no"] = particle_no
        self.particle_values["particle_no"] = particle_no

    def convert_particle_values(self):

        varminmax = self.setup["varminmax"]
        n_particles = self.setup["n_particles"]
        parameter_name = self.setup["columns"]
        continuous_discrete = self.setup["continuous_discrete"]
        n_parameters = len(parameter_name)

        # turn discrete parameters from continuous back into discrete parameters. Therefore have to set up the boundaries for each parameter
        converted_vals_range = varminmax
        # transpose particle values to make value conversion easier
        x_t = self.x_swarm.T.copy()

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
        self.x_swarm_converted = np.array(converted_vals.T).astype("float32")

        # # swap around parameters that work together and where min requires to be bigger than max. ohterwise wont do anzthing in petrel workflow.
        # for i in range(0,n_particles):

        #     for j in range(0,n_parameters):
        #         match_min = re.search("min",parameter_name[j],re.IGNORECASE)

        #         if match_min:
        #             match_max = re.search("max",parameter_name[j+1],re.IGNORECASE)

        #             if match_max:
        #                 if converted_particle_values[i,j] > converted_particle_values[i,j+1]:
        #                     converted_particle_values[i,j],converted_particle_values[i,j+1] = converted_particle_values[i,j+1],converted_particle_values[i,j] 
    
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
        particle = self.x_swarm_converted    # all particles together    
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

    def patch_voronoi_models(self):

        n_particles = self.setup["n_particles"]
        n_voronoi = self.setup["n_voronoi"]
        n_voronoi_zones = self.setup["n_voronoi_zones"]
        parameter_type = self.setup["parameter_type"]
        n_parameters = self.setup["n_parameters"]
        parameter_name = self.setup["columns"]
        nx = self.setup["nx"]
        ny = self.setup["ny"]
        nz = self.setup["nz"]
        iter_ticker = self.iteration 

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
                        voronoi_x_temp = self.x_swarm_converted[i,j]
                        voronoi_x.append(voronoi_x_temp)
                    elif "y" in parameter_name[j]:
                        voronoi_y_temp = self.x_swarm_converted[i,j]
                        voronoi_y.append(voronoi_y_temp)
                    # elif "z" in parameter_name[j]:
                    #     voronoi_z_temp = x_swarm_converted[i,j]
                    #     voronoi_z.append(voronoi_z_temp)

            # use these points to built a voronoi tesselation
            voronoi_x = np.array(voronoi_x)
            voronoi_y = np.array(voronoi_y)
            voronoi_points = np.vstack((voronoi_x,voronoi_y)).T
            # voronoi_z = np.array(voronoi_z)

            #crosscheck if any points lie on top of each other. if so --> move one
            for j in range(len(voronoi_points)):
                boolean = voronoi_points == voronoi_points[j]
                dublicate_tracker = 0
                for k in range(len(boolean)):
                    if np.sum(boolean[k]) == 2:
                        dublicate_tracker +=1
                    if dublicate_tracker ==2:
                        print("dublicate voronoi points --> reassignemnt")
                        voronoi_points[k] = voronoi_points[k]+ np.random.randint(0,10)

                        if voronoi_points[k,0] >= nx:
                            voronoi_points[k,0] = voronoi_points[k,0] - np.random.randint(0,20)
                        if voronoi_points[k,1] >= ny:
                            voronoi_points[k,1] = voronoi_points[k,1] - np.random.randint(0,20)

                        dublicate_tracker = 0

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


                self.setup["assign_voronoi_zone_" +str(i)] = assign_voronoi_zone
                # also need a fix for this. might not need this anymore.
                # with open(pickle_file,'wb') as f:
                #     pickle.dump(setup,f)

            else:
            # load voronoi zone assignemnt
            # and for this.
                # with open(pickle_file, "rb") as f:
                #     setup = pickle.load(f)
                # assign_voronoi_zone = setup["assign_voronoi_zone_" +str(i)]
                assign_voronoi_zone = self.setup["assign_voronoi_zone_" + str(i)]
            # in what voronoi zone and vornoi polygon do cell centers plot

            for j in range(len(all_cell_center)):
                for voronoi_polygon_id in range(n_voronoi):
                    
                    polygon = poly_shapes[voronoi_polygon_id]
                    cell_id = Point(all_cell_center[j,0],all_cell_center[j,1])
                    
                    if polygon.intersects(cell_id):
                        all_cell_center[j,2] = assign_voronoi_zone[voronoi_polygon_id]
            
            # load and assign correct grdecl files to each polygon zone and patch togetehr to new model
            #output from reservoir modelling
            cell_vornoi_combination = np.tile(all_cell_center[:,2],nz).reshape((nx,ny,nz))
            cell_vornoi_combination_flatten = cell_vornoi_combination.flatten()

            all_model_values_permx = np.zeros((n_voronoi_zones,len(cell_vornoi_combination_flatten)))
            all_model_values_permy = np.zeros((n_voronoi_zones,len(cell_vornoi_combination_flatten)))
            all_model_values_permz = np.zeros((n_voronoi_zones,len(cell_vornoi_combination_flatten)))
            all_model_values_poro = np.zeros((n_voronoi_zones,len(cell_vornoi_combination_flatten)))

            geomodel_path = str(self.setup["base_path"] / "../FD_Models/INCLUDE/GRID.grdecl")
            Model = GeologyModel(filename = geomodel_path)
            data_file_path = self.setup["base_path"] / "../FD_Models/DATA/M_FD_{}.DATA".format(i)

            for j in range(n_voronoi_zones):
                temp_model_path_permx = self.setup["base_path"] / '../FD_Models/INCLUDE/Voronoi/Patch_{}/PERMX/M{}.GRDECL'.format(j,i)
                temp_model_path_permy = self.setup["base_path"] / '../FD_Models/INCLUDE/Voronoi/Patch_{}/PERMY/M{}.GRDECL'.format(j,i)
                temp_model_path_permz = self.setup["base_path"] / '../FD_Models/INCLUDE/Voronoi/Patch_{}/PERMZ/M{}.GRDECL'.format(j,i)
                temp_model_path_poro = self.setup["base_path"] / '../FD_Models/INCLUDE/Voronoi/Patch_{}/PORO/M{}.GRDECL'.format(j,i)
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


            file_permx_beginning = "FILEUNIT\nMETRIC /\n\nPERMX\n"
            permx_file_path = self.setup["base_path"] / "../FD_Models/INCLUDE/PERMX/M{}.GRDECL".format(i)
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
            permy_file_path = self.setup["base_path"] / "../FD_Models/INCLUDE/PERMY/M{}.GRDECL".format(i)
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
            permz_file_path = self.setup["base_path"] / "../FD_Models/INCLUDE/PERMZ/M{}.GRDECL".format(i)
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
            poro_file_path = self.setup["base_path"] / "../FD_Models/INCLUDE/PORO/M{}.GRDECL".format(i)
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

        # # also here.
        # iter_ticker+=1
        # setup["iter_ticker"] = iter_ticker
        # with open(pickle_file,'wb') as f:
        #     pickle.dump(setup,f)

        # print ("Voronoi-Patching done")

    def built_FD_Data_files(self):
        # loading in settings that I set up on init_ABRM.py for this run

        n_particles = self.setup["n_particles"]
        schedule = self.setup["schedule"]

        for i in range (n_particles):

            data_file = "RUNSPEC\n\nTITLE\nModel_{}\n\nDIMENS\n--NX NY NZ\n200 100 7 /\n\n--Phases\nOIL\nWATER\n\n--DUALPORO\n--NODPPM\n\n--Units\nMETRIC\n\n--Number of Saturation Tables\nTABDIMS\n1 /\n\n--Maximum number of Wells\nWELLDIMS\n10 100 5 10 /\n\n--First Oil\nSTART\n1 OCT 2017 /\n\n--Memory Allocation\nNSTACK\n100 /\n\n--How many warnings allowed, but terminate after first error\nMESSAGES\n11*5000 1 /\n\n--Unified Output Files\nUNIFOUT\n\n--======================================================================\n\nGRID\n--Include corner point geometry model\nINCLUDE\n'..\INCLUDE\GRID.GRDECL'\n/\n\nACTNUM\n140000*1 /\n\n--Porosity\nINCLUDE\n'..\INCLUDE\PORO\M{}.GRDECL'\n/\n\n--Permeability\nINCLUDE\n'..\INCLUDE\PERMX\M{}.GRDECL'\n/\nINCLUDE\n'..\INCLUDE\PERMY\M{}.GRDECL'\n/\nINCLUDE\n'..\INCLUDE\PERMZ\M{}.GRDECL'\n/\n\n--Net to Gross\nNTG\n140000*1\n/\n\n--Output .INIT file to allow viewing of grid data in post proessor\nINIT\n\n--======================================================================\n\nPROPS\n\nINCLUDE\n'..\INCLUDE\DP_pvt.inc' /\n\nINCLUDE\n'..\INCLUDE\ROCK_RELPERMS.INC' /\n\n--======================================================================\n\nREGIONS\n\nEQLNUM\n140000*1\n/\nSATNUM\n140000*1\n/\nPVTNUM\n140000*1\n/\n\n--======================================================================\n\nSOLUTION\n\nINCLUDE\n'..\INCLUDE\SOLUTION.INC' /\n\n--======================================================================\n\nSUMMARY\n\nINCLUDE\n'..\INCLUDE\SUMMARY.INC' /\n\n--======================================================================\n\nSCHEDULE\n\nINCLUDE\n'..\INCLUDE\{}.INC' /\n\nEND".format(i,i,i,i,i,schedule)  
            data_file_path = self.setup["base_path"] / "../FD_Models/DATA/M_FD_{}.DATA".format(i)

            file = open(data_file_path, "w+")
            # write petrelfilepath and licence part into file and seed
            file.write(data_file)

            # close file
            file.close()

    def particle(self,i):
        # Objective Function run flow diagnostics
        particle_performance = self.obj_fkt_FD(i)

        # Compute Performance
        particle_misfit = self.misfit_fkt_F_Phi_curve(particle_performance)
        print('particle {}/{} - misfit {}'.format(i,self.setup["n_particles"],np.round(particle_misfit,3)),end = "\r")

        ### 10 ###
        # Compute particle parameter entropy
        # particle_entropy = compute_particle_paramter_entropy(x)
        #self.particle_entropy = self.compute_particle_paramter_entropy(x)

        ### 11 ###
        # compute upscaled model entropy
        #particle_tof_upscaled_entropy = compute_upscaled_model_entropy(particle_performance["tof"])

        ### 12 ###
        # Compute entropy and misfit combined
        # particle_combined_misfit_entropy = compute_combined_misfit_entropy(particle_misfit,particle_entropy,particle_tof_upscaled_entropy )

        # store misfit and particle no and iteration in dataframe
        particle_performance["iteration"] =self.iteration
        particle_performance["particle_no"] = i
        particle_performance["misfit"] = particle_misfit

        return particle_misfit,particle_performance

    def obj_fkt_FD(self,i):

        eng = matlab.engine.start_matlab()

        # run matlab and mrst
        eng.matlab_starter(nargout = 0)
        # print('sucessfully launched MRST.')

        # run FD and output dictionary
        FD_data = eng.FD_BS(i)

        # split into Ev tD F Phi and LC and tof column
        FD_data = np.array(FD_data._data).reshape((6,len(FD_data)//6))
        particle_performance = pd.DataFrame()
        
        particle_performance["EV"] = FD_data[0]
        particle_performance["tD"] = FD_data[1]
        particle_performance["F"] = FD_data[2]
        particle_performance["Phi"] = FD_data[3]
        particle_performance["LC"] = FD_data[4]
        particle_performance["tof"] = FD_data[5]
        particle_performance = particle_performance.astype("float32")

        return(particle_performance)

    def misfit_fkt_F_Phi_curve(self,particle_performance):

        F_points_target = self.setup["F_points_target"]
        Phi_points_target = self.setup["Phi_points_target"]
        # interpolate F-Phi curve from imput points with spline
        tck = interpolate.splrep(Phi_points_target,F_points_target, s = 0)
        Phi_interpolated = np.linspace(0,1,num = len(particle_performance["Phi"]),endpoint = True)
        F_interpolated = interpolate.splev(Phi_interpolated,tck,der = 0) # here can easily get first and second order derr.

        # calculate first order derivate of interpolated F-Phi curve and modelled F-Phi curve
        F_interpolated_first_derr = np.gradient(F_interpolated)
        F_first_derr = np.gradient(particle_performance["F"])
        F_interpolated_second_derr = np.gradient(F_interpolated_first_derr)
        F_second_derr = np.gradient(F_first_derr)

        # calculate LC for interpolatd F-Phi curve and modelled F-Phi curve
        LC_interpolated = self.compute_LC(F_interpolated,Phi_interpolated)

        LC = self.compute_LC(particle_performance["F"],particle_performance["Phi"])
        # calculate rmse for each curve and LC
        rmse_0 = mean_squared_error(F_interpolated,particle_performance["F"],squared=False)
        rmse_1 = mean_squared_error(F_interpolated_first_derr,F_first_derr,squared=False)
        rmse_2 = mean_squared_error(F_interpolated_second_derr,F_second_derr,squared=False)
        LC_error = abs(LC-LC_interpolated)

        # calculate misfit - RMSE if i turn squared to True it will calculate MSE
        misfit = rmse_0 + rmse_1 + rmse_2 + LC_error
        misfit = misfit.astype("float32")

        return misfit

    def compute_LC(self,F,Phi):

        v = np.diff(Phi,1)
        
        LC = 2*(np.sum(((np.array(F[0:-1]) + np.array(F[1:]))/2*v))-0.5)
        LC = LC.astype("float32")
        return LC
    