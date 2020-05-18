
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
from scipy import interpolate
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import shutil
import umap
import hdbscan
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import re
from pathlib import Path
import glob

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
    # built FD_Data files required for model evaluation
    built_FD_Data_files()

    ### 7 ###
    # evaluate model performance
    n_particles = x_swarm.shape[0]
    misfit_swarm = np.zeros(n_particles)
    swarm_performance = pd.DataFrame()
    LC_swarm = np.zeros(n_particles)

    for i in range(n_particles):
        misfit_particle, particle_performance = particle(x_swarm[i],i) # evaluate
        LC_swarm[i] = particle_performance.LC[0]
        misfit_swarm[i] = misfit_particle # store for return of function
        swarm_performance = swarm_performance.append(particle_performance) # store for data saving
    
    print('swarm {}'.format(misfit_swarm))

    ### 8 ###
    # save swarm_particle values and swarm_performance in df also save all models
    save_swarm_performance(swarm_performance)

    save_particle_values_converted(x_swarm_converted,misfit_swarm,LC_swarm)
    
    save_all_models()
    
    return np.array(misfit_swarm)	

def built_batch_file_for_petrel_models_uniform(x):

    # loading in settings that I set up on init_ABRM.py for this run
    base_path = Path(__file__).parent
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
    base_path = Path(__file__).parent
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

        # this bit here will overwrite the last run_petrel comman and replace the start with a call. this will allow me to wait unitl all processes are finished before the program moves on.
        # file_wait = open(built_multibat, "rt")
        # run_petrel_ticker_wait = run_petrel_ticker -1
        # run_petrel_bat_wait = '\nCall {}/batch_files/run_petrel_{}.bat'.format(base_path,run_petrel_ticker_wait)
        # insert_wait = file_wait.read()
        # insert_wait = insert_wait.replace(run_petrel_bat,run_petrel_bat_wait)
        # file_wait.close()

        # file_wait = open(built_multibat,"w+")
        # file_wait.write(insert_wait)
        # file_wait.close()

def run_batch_file_for_petrel_models(x):

    # loading in settings that I set up on init_ABRM.py for this run
    base_path = Path(__file__).parent
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
            while len(glob.glob(lock_files)) >= 1 or kill_timer < 7200:
                kill_timer += 1
                time.sleep(5)
            time.sleep(30)
            subprocess.call([kill_petrel]) # might need to add something that removes lock file here.


        print('Building models complete')

    else:
        print(" dry run - no model building")

def particle(x,i):

    ### 5 ###
    # Objective Function run flow diagnostics
    particle_performance = obj_fkt_FD(i)
    print("FD_performance worked")

    ### 6 ###
    # Compute Performance
    misfit = misfit_fkt_F_Phi_curve(particle_performance["F"],particle_performance["Phi"])
    print('misfit {}'.format(misfit))

    # # store misfit and particle no in dataframe
    particle_performance["particle_no"] = i
    particle_performance["misfit"] = misfit

    return misfit,particle_performance

def save_all_models():
    
    # loading in settings that I set up on init_ABRM.py for this run
    # loading in settings that I set up on init_ABRM.py for this run
    base_path = Path(__file__).parent
    pickle_file = base_path / "../Output/variable_settings.pickle"    # pickle_file = "C:/AgentBased_RM/Output/2020_04_23_12_49/variable_settings_saved.pickel"
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

            #cop and paste generic files into Data
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

def built_FD_Data_files():

    # loading in settings that I set up on init_ABRM.py for this run
    base_path = Path(__file__).parent
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
    base_path = Path(__file__).parent
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

    # loading in settings that I set up on init_ABRM.py for this run
    base_path = Path(__file__).parent
    pickle_file = base_path / "../Output/variable_settings.pickle"    
    with open(pickle_file, "rb") as f:
        setup = pickle.load(f)

    folder_path = setup["folder_path"]
    output_file_performance = "/swarm_performance_all_iter.csv"
    file_path = folder_path + output_file_performance

    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    if os.path.exists(file_path):

        swarm_performance_all_iter = pd.read_csv(file_path)
        
        #update iterations
        iteration = swarm_performance_all_iter.iteration.max() + 1
        swarm_performance["iteration"] = iteration

        # appends the performance of the swarm at a single iteration to the swarm performance of all previous iterations
        swarm_performance_all_iter = swarm_performance_all_iter.append(swarm_performance)
        
        # save again
        swarm_performance_all_iter.to_csv(file_path,index=False)

    else:
        # this should only happen in first loop.

        # number of iterations
        swarm_performance["iteration"] = 0

        swarm_performance.to_csv(file_path,index=False)        

def save_particle_values_converted(x_swarm_converted,misfit_swarm,LC_swarm):

    # loading in settings that I set up on init_ABRM.py for this run
        # loading in settings that I set up on init_ABRM.py for this run
    base_path = Path(__file__).parent
    pickle_file = base_path / "../Output/variable_settings.pickle"
    with open(pickle_file, "rb") as f:
        setup = pickle.load(f)

    columns = setup["columns"]
    folder_path = setup["folder_path"]

    particle_values_converted = pd.DataFrame(data = x_swarm_converted, columns= columns)

    # add misfit to df
    particle_values_converted["misfit"]= misfit_swarm

    # add particle no to df
    particle_no = np.arange(x_swarm_converted.shape[0], dtype = int)
    particle_values_converted["particle_no"] = particle_no

    # add LC to df
    particle_values_converted["LC"] = LC_swarm

    # filepath setup
    output_file_partilce_values_converted = "/swarm_particle_values_converted_all_iter.csv"

    file_path = folder_path + output_file_partilce_values_converted

    # check if folder exists
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    if os.path.exists(file_path):

        swarm_particle_values_converted_all_iter = pd.read_csv(file_path)

        #update iterations
        iteration = swarm_particle_values_converted_all_iter.iteration.max() + 1
        particle_values_converted["iteration"] = iteration

        # appends the performance of the swarm at a single iteration to the swarm performance of all previous iterations
        swarm_particle_values_converted_all_iter = swarm_particle_values_converted_all_iter.append(particle_values_converted)

        # save again
        swarm_particle_values_converted_all_iter.to_csv(file_path,index=False)

    else:
        # this should only happen in first loop.

        # number of iterations
        particle_values_converted["iteration"] = 0

        particle_values_converted.to_csv(file_path,index=False) 

def obj_fkt_FD(x):

    eng = matlab.engine.start_matlab()

    # run matlab and mrst
    eng.matlab_starter(nargout = 0)
    # print('sucessfully launched MRST.')

    # run FD and output dictionary
    FD_performance = eng.FD_BS(x)
    print('sucessfully ran FD.')
    # use this to only get data from matlab array instead of rest that matlab gives me
    FD_performance = np.array(FD_performance._data).T.tolist()

    # split into Ev tD F Phi and LC column
    FD_performance = np.reshape(FD_performance,(5,len(FD_performance)//5)).T
    # FD_performance = np.reshape(FD_performance,(6,len(FD_performance)//6)).T

    # convert to df
    columns = ["EV","tD","F","Phi","LC"]
    # columns = ["EV","tD","F","Phi","LC","tof"]

    FD_performance = pd.DataFrame(data = FD_performance, columns= columns)

    return FD_performance

def convert_particle_values(x):

    # loading in settings that I set up on init_ABRM.py for this run
    base_path = Path(__file__).parent
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
    base_path = Path(__file__).parent
    pickle_file = base_path / "../Output/variable_settings.pickle"
    with open(pickle_file, "rb") as f:
        setup = pickle.load(f)

    LC_target = setup["LC_target"]

    misfit = abs(LC_target - x)

    return misfit

def misfit_fkt_F_Phi_curve(F,Phi):
    
    # loading in settings that I set up on init_ABRM.py for this run
    base_path = Path(__file__).parent
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
    base_path = Path(__file__).parent
    output_path = str(base_path / "../Output/")

    #folder name will be current date and time
    output_folder = str(datetime.datetime.today().replace(microsecond= 0, second = 0).strftime("%Y_%m_%d_%H_%M"))
    output_file_variables = "/variable_settings_saved.pickel"
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
    file_path_constant = base_path / "../Output/variable_settings.pickle"
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

### Postprocessing functions ### 

def read_data(data_to_process):
    
    base_path = Path(__file__).parent

    # how many different datasets:
    n_datasets = len(data_to_process)

    # df to store all datasets in
    df_position = pd.DataFrame()
    df_performance = pd.DataFrame()
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
        setup = "variable_settings_saved.pickel"

        performance_path = path + performance
        position_path = path + position
        setup_path = path + setup

        df_performance_single = pd.read_csv(performance_path)
        df_position_single = pd.read_csv(position_path)
        with open(setup_path,'rb') as f:
            setup_single = pickle.load(f)        
        
        df_position_single["dataset"] = data_to_process[i]
        df_performance_single["dataset"] = data_to_process[i]
        
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
        setup_all[data_to_process[i]] = setup_single
        FD_targets[data_to_process[i]] = dict(Phi_interpolated = Phi_interpolated,F_interpolated =F_interpolated,LC_interpolated = LC_interpolated)

        # uniform index
        df_position.reset_index(drop = True,inplace = True)
        df_performance.reset_index(drop = True,inplace = True)
    
    return df_performance,df_position,setup_all,FD_targets

def best_model_selection_UMAP_HDBSCAN(df,setup_all,dataset,n_neighbors,min_cluster_size,misfit_tolerance,use_UMAP = True):
    # at some point make n_eighbours and min cluster size nad misfit tolernaze sliding scales
    # model parameters that generate lowest misfit
    if use_UMAP == True:
        columns = setup_all[dataset[0]]["columns"]
        df_best =df[(df.misfit <= misfit_tolerance)].copy()
        df_best_for_clustering = df_best[columns].copy()
        df_best_for_clustering["LC"] = df_best.LC.copy()
        # Create UMAP reducer
        reducer    = umap.UMAP(n_neighbors=n_neighbors)
        embeddings = reducer.fit_transform(df_best_for_clustering)

        # Create HDBSCAN clusters
        hdb = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size)
        scoreTitles = hdb.fit(embeddings)
        
        df_best["cluster_prob"] = scoreTitles.probabilities_
        df_best["cluster"] = scoreTitles.labels_
        
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
        columns = setup_all[dataset[0]]["columns"]
        df_best =df[(df.misfit <= misfit_tolerance)].copy()
        df_best_for_clustering = df_best[columns].copy()
        
        # Create UMAP reducer
        reducer    = umap.UMAP(n_neighbors=n_neighbors)
        embeddings = reducer.fit_transform(df_best_for_clustering)

        # Create HDBSCAN clusters
        hdb = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size)
        scoreTitles = hdb.fit(df_best_for_clustering)
        
        df_best["cluster_prob"] = scoreTitles.probabilities_
        df_best["cluster"] = scoreTitles.labels_
        
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
            height=650*(n_subplots)
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
            height=650*(n_subplots)
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

    base_path = Path(__file__).parent

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


# def built_batch_file_for_petrel_models(x):

#     # loading in settings that I set up on init_ABRM.py for this run
#     base_path = Path(__file__).parent
#     pickle_file = base_path / "../Output/variable_settings.pickle"
#     print(pickle_file)
#     with open(pickle_file, "rb") as f:
#         setup = pickle.load(f)

#     seed = setup["set_seed"]
#     n_modelsperbatch = setup["n_modelsperbatch"]
#     runworkflow = setup["runworkflow"]
#     n_particles = setup["n_particles"]
#     n_parallel_petrel_licenses = setup["n_parallel_petrel_licenses"]
#     petrel_path = setup["petrel_path"]

#     if runworkflow == "WF_2020_04_16":
#         #  Petrel has problems with batch files that get too long --> if I run
#         #  20+ models at once. Therefore split it up in to sets of 3 particles / models
#         #  per Petrel license and then run them in parallel. hard on CPU but
#         #  shouldnt be too time expensive. Parameter iterations = number of
#         #  models.
#         particle = x    # all particles together    
#         particle_1d_array =  x.reshape((x.shape[0]*x.shape[1]))    # all particles together                
#         particlesperwf = np.linspace(25,27,n_modelsperbatch, endpoint = True,dtype = int) # use 25,26,27 because of petrel wf. there the variables are named like that and cant bothered to change that.
#         n_particles = x.shape[0]    # how many particles
#         single_wf = [str(i) for i in np.tile(particlesperwf,n_particles)]
#         single_particle_in_wf = [str(i) for i in np.arange(0,n_particles+1)]
#         particle_str = np.asarray([str(i) for i in particle_1d_array]).reshape(x.shape[0],x.shape[1])
#         slicer_length = int(np.ceil(n_particles/n_modelsperbatch)) # always rounds up.
#         slicer = np.arange(0,slicer_length,dtype = int)     # slicer = np.arange(0,(n_particles/n_modelsperbatch),dtype = int)

#         # might need to use these variables if I also want to generate multibat files (see bottom of file)
#         # n_parallel_petrel_licenses = 3  # how many petrel licenses can I call in parallel before I crash the CPU of my computer 3 for my machine
#         # parallel_petrel_licenses = np.arange(0,n_parallel_petrel_licenses)
#         # n_multibatfiles = int(np.ceil(n_particles/n_modelsperbatch/n_parallel_petrel_licenses)) # how many multibat files will I need to create
#         # multibatfiles = np.arange(0,n_multibatfiles,dtype = int)

#         #  Which TI to use (required for Petrel), still testing this out
#         #  Idea is to swap one of the "not in use" sings with the TI that were
#         #  using and renaming that string to TI1 and then call that with petrel.
#         #  The other TIs (not in use) will also appear as string variables in
#         #  the petrel workflow, but will remain unused.
#         TI1 = np.array([ ['None'] * 4 ]*n_particles) 
#         TI2 = np.array([ ['None'] * 4 ]*n_particles)  
#         TI3 = np.array([ ['None'] * 4 ]*n_particles)

#         for index in range (0, n_particles):
#         # which Training image for zone 1
#             if particle[index,0] == 1:
#                 particle_str[index,0] = "TI1_1"
#                 TI1[index,0] = "TI1"
#             elif particle[index,0] == 2:
#                 particle_str[index,0] = "TI1_2"
#                 TI1[index,1] = "TI1"
#             elif particle[index,0] == 3:
#                 particle_str[index,0] = "TI1_3"
#                 TI1[index,2] = "TI1"
#             else:
#                 particle_str[index,0] = "TI1_4"
#                 TI1[index,3] = "TI1"

#         # which Training image for zone 2
#             if particle[index,7] == 1:
#                 particle_str[index,7] = "TI2_1"
#                 TI2[index,0] = "TI2" # figure out what this is for again
#             elif particle[index,7] == 2:
#                 particle_str[index,7] = "TI2_2"
#                 TI2[index,1] = "TI2"
#             elif particle[index,7] == 3:
#                 particle_str[index,7] = "TI2_3"
#                 TI2[index,2] = "TI2"
#             else:
#                 particle_str[index,7] = "TI2_4"
#                 TI2[index,3] = "TI2"

#         # which Training image for zone 3
#             if particle[index,14] == 1:
#                 particle_str[index,14] = "TI3_1"
#                 TI3[index,0] = "TI3" # figure out what this is for again
#             elif particle[index,14] == 2:
#                 particle_str[index,14] = "TI3_2"
#                 TI3[index,1] = "TI3"
#             elif particle[index,14] == 3:
#                 particle_str[index,14] = "TI3_3"
#                 TI3[index,2] = "TI3"
#             else:
#                 particle_str[index,14] = "TI3_4"
#                 TI3[index,3] = "TI3"

#         # set up file path to petrel, petrel license and petrel projects and seed
#         callpetrel = 'call "{}" ^'.format(petrel_path)
#         license = '\n/licensePackage Standard ^'
#         runworkflow = '\n/runWorkflow "{}" ^\n'.format(runworkflow)
#         seed_petrel = '/nParm seed={} ^\n'.format(seed) 


#         projectpath = []
#         parallel_petrel_licenses = np.arange(0,n_parallel_petrel_licenses,1)
#         for i in range(0,len(parallel_petrel_licenses)):
#             # path_petrel_projects = Path(__file__).parent / "../"
#             # base_path = Path(__file__).parent
#             path_petrel_projects = base_path / "../Petrel_Projects/ABRM_"
#             # path = '\n"C:/AgentBased_RM/Petrel_Projects/ABRM_{}.pet"'.format(parallel_petrel_licenses[i])
#             path = '\n"{}{}.pet"'.format(path_petrel_projects,parallel_petrel_licenses[i])
#             print(path)
#             projectpath.append(path)
#         projectpath_repeat = projectpath * (len(slicer))    

#         quiet = '/quiet ^' #wf wont pop up
#         noshowpetrel = '\n/nosplashscreen ^' # petrel wont pop up
#         exit = '\n/exit ^'  # exit petrel
#         exit_2 = '\nexit' 	# exit bash file
    
#         # set path for batch file to open and start writing into it
#         for _i, index_2 in enumerate(slicer):
            
#             # path to petrel project
#             path = projectpath_repeat[index_2]

#             # path to batch file
#             run_petrel_batch = base_path / "../ABRM_functions/batch_files/run_petrel_{}.bat".format(index_2)
#             # run_petrel_batch = 'C:/AgentBased_RM/ABRM_functions/batch_files/run_petrel_{}.bat'.format(index_2)

#             # open batch file to start writing into it / updating it
#             file = open(run_petrel_batch, "w+")

#             # write petrelfilepath and licence part into file and seed
#             file.write(callpetrel)
#             file.write(license)
#             file.write(runworkflow)
#             file.write(seed_petrel)

#             # generate n models per batch file / petrel license
#             variables_per_model = np.arange((n_modelsperbatch*slicer[index_2]),(n_modelsperbatch*(index_2+1)))
#             for _index_3, j in enumerate(variables_per_model):

#                 # parameter setup so that particles can be inserted into petrel workflow {} are place holders that will be fileld in with variable values,changing with each workflow
#                 Modelname = '/sparm ModelName_{}=M{} ^\n'.format(single_wf[j],single_particle_in_wf[j])

#                 # TI1 zonation, TI seleciton, Curvature prob
#                 TI1_1 = '/sparm TI1_1_{}={} ^\n'.format(single_wf[j],TI1[j,0])
#                 TI1_2 = '/sparm TI1_2_{}={} ^\n'.format(single_wf[j],TI1[j,1])
#                 TI1_3 = '/sparm TI1_3_{}={} ^\n'.format(single_wf[j],TI1[j,2])
#                 TI1_4 = '/sparm TI1_4_{}={} ^\n'.format(single_wf[j],TI1[j,3])
#                 F1_I_MIN = '/nParm F1_I_MIN_{}={} ^\n'.format(single_wf[j],particle_str[j,1])
#                 F1_I_MAX = '/nParm F1_I_MAX_{}={} ^\n'.format(single_wf[j],particle_str[j,2])
#                 F1_J_MIN = '/nParm F1_J_MIN_{}={} ^\n'.format(single_wf[j],particle_str[j,3])
#                 F1_J_MAX = '/nParm F1_J_MAX_{}={} ^\n'.format(single_wf[j],particle_str[j,4])
#                 F1_K_MIN = '/nParm F1_K_MIN_{}={} ^\n'.format(single_wf[j],particle_str[j,5])
#                 F1_K_MAX = '/nParm F1_K_MAX_{}={} ^\n'.format(single_wf[j],particle_str[j,6])
#                 F1_Curve_Prob = '/nParm F1_Curve_Prob_{}={} ^\n'.format(single_wf[j],particle_str[j,21])

#                 # TI2 zonation, TI seleciton, Curvature prob
#                 TI2_1 = '/sparm TI2_1_{}={} ^\n'.format(single_wf[j],TI2[j,0])
#                 TI2_2 = '/sparm TI2_2_{}={} ^\n'.format(single_wf[j],TI2[j,1])
#                 TI2_3 = '/sparm TI2_3_{}={} ^\n'.format(single_wf[j],TI2[j,2])
#                 TI2_4 = '/sparm TI2_4_{}={} ^\n'.format(single_wf[j],TI2[j,3])
#                 F2_I_MIN = '/nParm F2_I_MIN_{}={} ^\n'.format(single_wf[j],particle_str[j,8])
#                 F2_I_MAX = '/nParm F2_I_MAX_{}={} ^\n'.format(single_wf[j],particle_str[j,9])
#                 F2_J_MIN = '/nParm F2_J_MIN_{}={} ^\n'.format(single_wf[j],particle_str[j,10])
#                 F2_J_MAX = '/nParm F2_J_MAX_{}={} ^\n'.format(single_wf[j],particle_str[j,11])
#                 F2_K_MIN = '/nParm F2_K_MIN_{}={} ^\n'.format(single_wf[j],particle_str[j,12])
#                 F2_K_MAX = '/nParm F2_K_MAX_{}={} ^\n'.format(single_wf[j],particle_str[j,13])
#                 F2_Curve_Prob = '/nParm F2_Curve_Prob_{}={} ^\n'.format(single_wf[j],particle_str[j,22])

#                 # TI2 zonation, TI seleciton, Curvature prob
#                 TI3_1 = '/sparm TI3_1_{}={} ^\n'.format(single_wf[j],TI3[j,0])
#                 TI3_2 = '/sparm TI3_2_{}={} ^\n'.format(single_wf[j],TI3[j,1])
#                 TI3_3 = '/sparm TI3_3_{}={} ^\n'.format(single_wf[j],TI3[j,2])
#                 TI3_4 = '/sparm TI3_4_{}={} ^\n'.format(single_wf[j],TI3[j,3])
#                 F3_I_MIN = '/nParm F3_I_MIN_{}={} ^\n'.format(single_wf[j],particle_str[j,15])
#                 F3_I_MAX = '/nParm F3_I_MAX_{}={} ^\n'.format(single_wf[j],particle_str[j,16])
#                 F3_J_MIN = '/nParm F3_J_MIN_{}={} ^\n'.format(single_wf[j],particle_str[j,17])
#                 F3_J_MAX = '/nParm F3_J_MAX_{}={} ^\n'.format(single_wf[j],particle_str[j,18])
#                 F3_K_MIN = '/nParm F3_K_MIN_{}={} ^\n'.format(single_wf[j],particle_str[j,19])
#                 F3_K_MAX = '/nParm F3_K_MAX_{}={} ^\n'.format(single_wf[j],particle_str[j,20])
#                 F3_Curve_Prob = '/nParm F3_Curve_Prob_{}={} ^\n'.format(single_wf[j],particle_str[j,23])

#                 # Permeabilities
#                 FracpermX = '/nParm FracPermX_{}={} ^\n'.format(single_wf[j],particle_str[j,24])
#                 MatrixpermX = '/nParm MatrixPermX_{}={} ^\n'.format(single_wf[j],particle_str[j,25])
#                 FracpermY = '/nParm FracPermY_{}={} ^\n'.format(single_wf[j],particle_str[j,26])
#                 MatrixpermY = '/nParm MatrixPermY_{}={} ^\n'.format(single_wf[j],particle_str[j,27])
#                 FracpermZ = '/nParm FracPermZ_{}={} ^\n'.format(single_wf[j],particle_str[j,28])
#                 MatrixpermZ = '/nParm MatrixPermZ_{}={} ^\n'.format(single_wf[j],particle_str[j,29])
                
#                 # write into file
#                 file.write(Modelname)
#                 file.write(TI1_1)
#                 file.write(TI1_2)
#                 file.write(TI1_3)
#                 file.write(TI1_4)
#                 file.write(TI2_1)
#                 file.write(TI2_2)
#                 file.write(TI2_3)
#                 file.write(TI2_4)
#                 file.write(TI3_1)
#                 file.write(TI3_2)
#                 file.write(TI3_3)
#                 file.write(TI3_4)
#                 file.write(F1_I_MIN)
#                 file.write(F1_I_MAX)
#                 file.write(F1_J_MIN)
#                 file.write(F1_J_MAX)
#                 file.write(F1_K_MIN)
#                 file.write(F1_K_MAX)
#                 file.write(F2_I_MIN)
#                 file.write(F2_I_MAX)
#                 file.write(F2_J_MIN)
#                 file.write(F2_J_MAX)
#                 file.write(F2_K_MIN)
#                 file.write(F2_K_MAX)
#                 file.write(F3_I_MIN)
#                 file.write(F3_I_MAX)
#                 file.write(F3_J_MIN)
#                 file.write(F3_J_MAX)
#                 file.write(F3_K_MIN)
#                 file.write(F3_K_MAX)
#                 file.write(F1_Curve_Prob)
#                 file.write(F2_Curve_Prob)
#                 file.write(F3_Curve_Prob)
#                 file.write(FracpermX)
#                 file.write(MatrixpermX) 
#                 file.write(FracpermY)
#                 file.write(MatrixpermY)
#                 file.write(FracpermZ)
#                 file.write(MatrixpermZ)
#             # write into file
#             file.write(quiet)
#             file.write(noshowpetrel)
#             file.write(exit)
#             file.write(path)
#             file.write(exit_2)


#             # close file
#             file.close()

#     elif runworkflow == "WF_test":
    
#         particle = x    # all particles together    
#         particle_1d_array =  x.reshape((x.shape[0]*x.shape[1]))    # all particles together                
#         # n_modelsperbatch = 3    # limitation for how logn a single petrel workflow batch file can be ==> 3 models
#         particlesperwf = np.linspace(25,30,n_modelsperbatch, endpoint = True,dtype = int) # use 25,26,27 because of petrel wf. there the variables are named like that and cant bothered to change that.
#         n_particles = x.shape[0]    # how many particles
#         single_wf = [str(i) for i in np.tile(particlesperwf,n_particles)]
#         single_particle_in_wf = [str(i) for i in np.arange(0,n_particles+1)]
#         particle_str = np.asarray([str(i) for i in particle_1d_array]).reshape(x.shape[0],x.shape[1])
#         slicer_length = int(np.ceil(n_particles/n_modelsperbatch)) # always rounds up.
#         slicer = np.arange(0,slicer_length,dtype = int)     # slicer = np.arange(0,(n_particles/n_modelsperbatch),dtype = int)

#         # set up file path to petrel, petrel license and petrel projects and seed
#         callpetrel = 'call "{}" ^'.format(petrel_path)  
#         license = '\n/licensePackage Standard ^'
#         runworkflow = '\n/runWorkflow "{}" ^\n'.format(runworkflow)
#         seed_petrel = '/nParm seed={} ^\n'.format(seed) 

#         projectpath = []
#         parallel_petrel_licenses = np.arange(0,n_parallel_petrel_licenses,1)
#         for i in range(0,len(parallel_petrel_licenses)):
#             path_petrel_projects = base_path / "../Petrel_Projects/ABRM_"
#             # path = '\n"C:/AgentBased_RM/Petrel_Projects/ABRM_{}.pet"'.format(parallel_petrel_licenses[i])
#             path = '\n"{}{}.pet"'.format(path_petrel_projects,parallel_petrel_licenses[i])
#             projectpath.append(path)
#         projectpath_repeat = projectpath * (len(slicer))    

#         quiet = '/quiet ^' #wf wont pop up
#         noshowpetrel = '\n/nosplashscreen ^' # petrel wont pop up
#         exit = '\n/exit ^'  # exit petrel
#         exit_2 = '\nexit' 	# exit bash file
    
#         # set path for batch file to open and start writing into it
#         for _i, index_2 in enumerate(slicer):
            
#             # path to petrel project
#             path = projectpath_repeat[index_2]

#             # path to batch file
#             run_petrel_batch = base_path / "../ABRM_functions/batch_files/run_petrel_{}.bat".format(index_2)

#             # open batch file to start writing into it / updating it
#             file = open(run_petrel_batch, "w+")

#             # write petrelfilepath and licence part into file and seed
#             file.write(callpetrel)
#             file.write(license)
#             file.write(runworkflow)
#             file.write(seed_petrel)

#             # generate n models per batch file / petrel license
#             variables_per_model = np.arange((n_modelsperbatch*slicer[index_2]),(n_modelsperbatch*(index_2+1)))
#             for _index_3, j in enumerate(variables_per_model):

#                 # parameter setup so that particles can be inserted into petrel workflow {} are place holders that will be fileld in with variable values,changing with each workflow
#                 Modelname = '/sparm ModelName_{}=M{} ^\n'.format(single_wf[j],single_particle_in_wf[j])
#                 # TI1 zonation, TI seleciton, Curvature prob
#                 Region = '/nParm Region_{}={} ^\n'.format(single_wf[j],particle_str[j,0])
#                 Perm = '/nParm Perm_{}={} ^\n'.format(single_wf[j],particle_str[j,1])
#                 # example variable = "/{}Parm {}_{} = {}".fromat(n or s, column name, model name, value)

#                 # write into file
#                 file.write(Modelname)
#                 file.write(Region)
#                 file.write(Perm)
                
#             # write into file
#             file.write(quiet)
#             file.write(noshowpetrel)
#             file.write(exit)
#             file.write(path)
#             file.write(exit_2)

#             # close file
#             file.close()