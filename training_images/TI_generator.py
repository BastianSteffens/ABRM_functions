import numpy as np
import pandas as pd
import random 
import os
import shutil
import bz2
import _pickle as cPickle
import subprocess
import time
import glob 

class TI_generator():
    """ class that allows to the automatic generation of n new training images in petrel and output there poro/perm files. """

    def __init__(self, seed,setup):
        """ set seed and initialize TI_generator """

        self.set_seed = seed
        random.seed(self.set_seed)
        self.setup = setup
        self.n_TI = self.setup["n_TI"]
        self.df_all_TI_data = pd.DataFrame()


    def run_TI_generator(self):
        """ make training images and run TI_generator"""
        
        n_fracsets_sampling_style = self.setup["n_fracsets_sampling_style"]
        n_fracsets_random_range = self.setup["n_fracsets_random_range"]
        n_fracsets_area_specific_range = self.setup["n_fracsets_area_specific_range"]
        property_stats_random = self.setup["property_stats_random"]
        property_stats_area_specific = self.setup["property_stats_area_specific"]
        
        for i in range(self.n_TI):

            # property_stats = self.setup["property_stats"][i]

            self.df_current_TI_data = pd.DataFrame()
            self.df_current_TI_data["TI_no"] = [i]
            
            #set P32, and fraction of random fracutes. P32 is then distributed between the two.
            P32_random, P32_area_specific = self.set_fracture_P32_distribution()

            # set number of fracsets
            n_fracsets_random = np.round(self.sample_from_distribution(sampling_style= n_fracsets_sampling_style,property_stats = n_fracsets_random_range))
            # if n_fracsets_random == 0:
            #     n_fracsets_random == 0.01
            n_fracsets_area_specific = np.round(self.sample_from_distribution(sampling_style= n_fracsets_sampling_style,property_stats = n_fracsets_area_specific_range))
            # if n_fracsets_area_specific == 0:
            #     n_fracsets_area_specific == 0.01
            self.df_current_TI_data["n_fracsets_random"] = [n_fracsets_random]
            self.df_current_TI_data["n_fracsets_area_specific"] = [n_fracsets_area_specific]

            # set up random fracture sets
            for j in range(n_fracsets_random_range[1]):
                self.generate_variables(P32_fractype = P32_random,n_fracsets = n_fracsets_random, fracset_no = j,frac_type ="random")

            # set up area specific fracture sets
            # for j in range(n_fracsets_area_specific_range[1]):
            for j in range(4):

                self.generate_variables(P32_fractype = P32_area_specific,n_fracsets = n_fracsets_area_specific, fracset_no = j,frac_type ="area_specific")
            
            # add data to overarching df
            self.df_all_TI_data = self.df_all_TI_data.append(self.df_current_TI_data, ignore_index = True)
        
        # save TI_settings data
        self.save_TI_settings()
        
        # built batch files
        self.built_batch_file_petrel()

        # built mutlibat files
        self.built_multibat_files()

        # run bat files
        self.run_batch_file_for_petrel_models()

        # built DATA files
        self.built_data_files()

        # extract mean_permxzy and frac_cell_fraction
        self.get_TI_frac_data()

        # save everything
        self.save_data()

    def set_fracture_P32_distribution(self):
        """ define P32, and how it is distributed over random and zone specific fractures """
            
        P32_total_sampling_style = self.setup["P32_total_sampling_style"]
        P32_total_range = self.setup["P32_total_range"]
        random_fracs_sampling_style = self.setup["random_fracs_sampling_style"]
        random_fracs_fraction_range = self.setup["random_fracs_fraction_range"]

        # set P32
        P32_all = np.round(self.sample_from_distribution(sampling_style =P32_total_sampling_style,  property_stats = P32_total_range),5)

        # set % random fractures
        random_fracs_share = np.round(self.sample_from_distribution(sampling_style = random_fracs_sampling_style,property_stats = random_fracs_fraction_range),2)
        while random_fracs_share > 1 or random_fracs_share <0:
            random_fracs_share = np.round(self.sample_from_distribution(sampling_style = random_fracs_sampling_style,property_stats = random_fracs_fraction_range),2)

        # set % zone specific fractures
        area_fracs_share = np.round(1-random_fracs_share,2)

        # P32 random fracs
        P32_random = np.round(P32_all * random_fracs_share,5)

        # P32 zone specific fracs
        P32_area_specific = np.round(P32_all * area_fracs_share,5)

        self.df_current_TI_data["P32"] = [P32_all]
        self.df_current_TI_data["P32_random"] = [P32_random]
        self.df_current_TI_data["P32_area_specific"] = [P32_area_specific]

        return P32_random, P32_area_specific
    
    def sample_from_distribution(self,sampling_style,property_stats):
        """ random sample from uniform/gaussian/logarithmic/bimodal distribution """

        if sampling_style == "uniform":
            low = property_stats[0]
            high = property_stats[1]
            random_draw = np.random.uniform(low,high)
        
        elif sampling_style == "normal":
            mean = property_stats[0]
            std = property_stats[1]
            random_draw = np.random.normal(mean,std)
            while random_draw <0:
                random_draw = np.random.normal(mean,std)

            # if random_draw <= 0:
            #     random_draw = 0.0001

        return random_draw

    def generate_variables(self,P32_fractype,n_fracsets,fracset_no,frac_type):
        """ generate variables that are required to built fractures in Petrel for 1 fracture set """ 
        
        # get data for fracture property ranges
        if frac_type == "random":
            property_stats = self.setup["property_stats_random"][fracset_no]
        elif frac_type == "area_specific":
            property_stats = self.setup["property_stats_area_specific"][fracset_no]
        
        # how to sample variables
        sampling_styles = self.setup["property_sampling_style"]

        # what variables to sample
        property_name = self.setup["property_name"]

        # set seed for each individual fracture set
        fracset_seed = "seed_" + frac_type + "_" + str(fracset_no)
        seed = np.round(np.random.uniform(0,100000),1)
        # seed = 123
        # seed = 111222.2
        self.df_current_TI_data[fracset_seed] = [seed]

        # set P32 for fracture set
        fracset_P32 = "P32_" + frac_type + "_" + str(fracset_no)
        if n_fracsets == 0:
            P32 = 0.0001
        else:
            P32 = P32_fractype/n_fracsets
        if n_fracsets <= fracset_no:
            P32 = 0.0001
    
        self.df_current_TI_data[fracset_P32] = [P32]

        # set all other paramters
        for i in range(len(property_name)):
            fracset_property_name = property_name[i] + "_" + frac_type + "_" + str(fracset_no)
            fracset_property_value = np.round(self.sample_from_distribution(sampling_style= sampling_styles[i],property_stats = property_stats[i]),2)
            self.df_current_TI_data[fracset_property_name] = [fracset_property_value]
    
    def built_batch_file_petrel(self):
        """ built a batch file with the sampled values to generate Training images in Petrel """
        
        seed = self.set_seed
        n_modelsperbatch = self.setup["n_modelsperbatch"]
        runworkflow = self.setup["runworkflow"]
        petrel_path = self.setup["petrel_path"]
        n_parallel_petrel_licenses = self.setup["n_parallel_petrel_licenses"]
        parameter_name = self.df_all_TI_data.columns.values.tolist()
        unwanted = {"TI_no","P32","P32_random","P32_area_specific","n_fracsets_random","n_fracsets_area_specific"}
        parameter_name = [e for e in parameter_name if e not in unwanted]
        n_parameters = len(parameter_name)
        parameter_type = self.setup["property_type"]
        parameter_type = np.zeros(n_parameters).tolist()
        n_TI = self.n_TI
        all_TI_data = self.df_all_TI_data.copy()
        all_TI_data = all_TI_data.drop(columns =["P32","P32_random","P32_area_specific","TI_no","n_fracsets_random","n_fracsets_area_specific"])
        all_TI_data = all_TI_data.to_numpy()
        all_TI_data_1D = all_TI_data.reshape((all_TI_data.shape[0]*all_TI_data.shape[1]))
        TIsperwf = np.linspace(0,n_modelsperbatch,n_modelsperbatch, endpoint = False,dtype = int) # this is how it should be. This is the name that each variable has per model in the petrel wf
        single_wf = [str(i) for i in np.tile(TIsperwf,n_TI)]
        single_TI_in_wf = [str(i) for i in np.arange(0,n_TI+1)]
        TI_str = np.asarray([str(i) for i in all_TI_data_1D]).reshape(all_TI_data.shape[0],all_TI_data.shape[1])
        parameter_name_str = np.asarray([parameter_name * n_TI]).reshape(all_TI_data.shape[0],all_TI_data.shape[1]) 
        parameter_type_str = np.asarray([parameter_type * n_TI]).reshape(all_TI_data.shape[0],all_TI_data.shape[1])
        slicer_length = int(np.ceil(n_TI/n_modelsperbatch)) # always rounds up.
        slicer = np.arange(0,slicer_length,dtype = int) 


        # set up file path to petrel, petrel license and petrel projects and seed etc
        callpetrel = 'call "{}" ^'.format(petrel_path)
        license = '\n/licensePackage Petrel_112614311_MACKPHiBP8aUA ^'

        runworkflow = '\n/runWorkflow "{}" ^\n'.format(runworkflow)
        # seed_petrel = '/nParm seed={} ^\n'.format(seed) 
        projectpath = []
        parallel_petrel_licenses = np.arange(0,n_parallel_petrel_licenses,1)
        for i in range(0,len(parallel_petrel_licenses)):
            path_petrel_projects = self.setup["base_path"] / "../../Petrel_Projects/TI_generator_"
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
            run_petrel_batch = self.setup["base_path"] / "../../ABRM_functions/batch_files/run_petrel_{}.bat".format(i)

            # open batch file to start writing into it / updating it
            file = open(run_petrel_batch, "w+")

            # write petrelfilepath and licence part into file and seed
            file.write(callpetrel)
            file.write(license)
            file.write(runworkflow)
            # file.write(seed_petrel)

            # generate n models per batch file / petrel license
            variables_per_model = np.arange((n_modelsperbatch*slicer[i]),(n_modelsperbatch*(i+1)))
            for _index_3, j in enumerate(variables_per_model):
                
                # parameter setup so that particles can be inserted into petrel workflow {} are place holders that will be fileld in with variable values,changing with each workflow
                Modelname = '/sparm ModelName_{}=TI{} ^\n'.format(single_wf[j],single_TI_in_wf[j])
                file.write(Modelname)
                # for parameter name feature create something similar to singlewf or particle str feature as done above.

                for k in range(0,n_parameters):

                    # string parameters
                    if parameter_type_str[j,k] == 0:
                        parameter = '/sParm {}_{}={} ^\n'.format(parameter_name_str[j,k],single_wf[j],TI_str[j,k])
                        file.write(parameter)

                    # numeric parameters 
                    elif parameter_type_str[j,k] == 1:
                        parameter = '/nParm {}_{}={} ^\n'.format(parameter_name_str[j,k],single_wf[j],TI_str[j,k])
                        file.write(parameter)

            # write into file
            file.write(quiet)
            file.write(noshowpetrel)
            file.write(exit)
            file.write(path)
            file.write(exit_2)

            # close file
            file.close()

    def built_multibat_files(self):
        """ built multi_batch file bat files that can launch several petrel licenses (run_petrel) at once """

        n_TI = self.n_TI
        n_modelsperbatch = self.setup["n_modelsperbatch"]
        n_parallel_petrel_licenses = self.setup["n_parallel_petrel_licenses"] 

        exit_bat = "\nexit"

        n_multibats = int(np.ceil(n_TI / n_modelsperbatch/n_parallel_petrel_licenses))
        run_petrel_ticker = 0 # naming of petrelfiles to run. problem: will alwazs atm write 3 files into multibatfile.

        for i in range(0,n_multibats):
            built_multibat = self.setup["base_path"] / "../../ABRM_functions/batch_files/multi_bat_{}.bat".format(i)
            file = open(built_multibat, "w+")

            for _j in range(0,n_parallel_petrel_licenses):

                run_petrel_bat = '\nStart ' + str(self.setup["base_path"] / "../../ABRM_functions/batch_files/run_petrel_{}.bat".format(run_petrel_ticker))
                file.write(run_petrel_bat)
                run_petrel_ticker+=1

            file.write(exit_bat)
            file.close()

    def run_batch_file_for_petrel_models(self):
        
        petrel_on = self.setup["petrel_on"]
        n_TI = self.n_TI
        n_modelsperbatch = self.setup["n_modelsperbatch"]
        n_parallel_petrel_licenses = self.setup["n_parallel_petrel_licenses"]    
        lock_files = str(self.setup["base_path"] / "../../Petrel_Projects/*.lock")

        kill_petrel = str(self.setup["base_path"] / "../../ABRM_functions/batch_files/kill_petrel.bat")

        if petrel_on == True:
            # initiate model by running batch file make sure that petrel has sufficient time to built the models and shut down again. 
            print(' Start building Training Images',end = "\r")

            #how many multibat files to run
            n_multibats = int(np.ceil(n_TI / n_modelsperbatch/n_parallel_petrel_licenses))
            for i in range(0,n_multibats):
                run_multibat = str(self.setup["base_path"] / "../../ABRM_functions/batch_files/multi_bat_{}.bat".format(i))
                subprocess.call([run_multibat])
                # not continue until lock files are gone and petrel is finished.
                time.sleep(60)
                kill_timer = 1 # waits 1h before petrel project is shut down if it has a bug 
                n_locked_files = len(glob.glob(lock_files)) 
                while n_locked_files >= 1:
                    kill_timer += 1
                    time.sleep(5)
                    print("waiting for petrel: {}/15min".format((np.round(5*kill_timer/60,2))),end = "\r")
                    if kill_timer >= 180:
                        n_locked_files = 0
                    else:
                        n_locked_files = len(glob.glob(lock_files)) 

                time.sleep(30)
                subprocess.call([kill_petrel]) # might need to add something that removes lock file here.

            print('Building Training Images complete',end = "\r")

        else:
            print("dry run - no Training Image building",end = "\r")

    def save_TI_settings(self):
        """ save df to csv files with all data used for Training Image generation """

        # filepath setup
        folder_path = self.setup["folder_path"]
        schedule = self.setup["shedule"]
        base_path = self.setup["base_path"]
        
        output_Training_image_settings = "TI_properties.csv"
        setup_file = "setup.pickle"

        file_path_output_Training_image_settings = folder_path / output_Training_image_settings
        file_path_setup = folder_path / setup_file

        # make folder
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        
        # save all
        self.df_all_TI_data.to_csv(file_path_output_Training_image_settings,index=False)
        with bz2.BZ2File(file_path_setup,"w") as f:
            cPickle.dump(self.setup,f, protocol= 4)

    def save_data(self):
        """ save df to csv files with all data used for Training Image generation also data from petrel get saved in extra folder """

        # filepath setup
        folder_path = self.setup["folder_path"]
        schedule = self.setup["shedule"]
        base_path = self.setup["base_path"]
        
        output_Training_image_settings = "TI_properties.csv"
        setup_file = "setup.pickle"

        file_path_output_Training_image_settings = folder_path / output_Training_image_settings
        file_path_setup = folder_path / setup_file

        # make folder
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        
        # save all
        self.df_all_TI_data.to_csv(file_path_output_Training_image_settings,index=False)
        with bz2.BZ2File(file_path_setup,"w") as f:
            cPickle.dump(self.setup,f, protocol= 4)


        destination_path = self.setup["folder_path"] / 'all_models'
        data_path = destination_path / "DATA"
        include_path = destination_path / "INCLUDE"
        permx_path = include_path / "PERMX"
        permy_path = include_path / "PERMY"
        permz_path = include_path / "PERMZ"
        poro_path = include_path / "PORO"

        n_TI = self.n_TI
        source_path = base_path / "../../Output/training_images/current_run"

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

            for TI_id in range(0,n_TI):

                # set path for Datafile
                data_file_path = data_path / "TI{}.DATA".format(model_id)
                # getting higher model numbers for saving
                while os.path.exists(data_file_path):
                    model_id += 1
                    data_file_path = data_path / "TI{}.DATA".format(model_id)

                # open datafile file to start writing into it / updating it
                self.built_data_file(data_file_path,model_id)

                #copy and paste permxyz and poro files to new location
                permx_file_src_path = PERMX_src_path / "TI{}.GRDECL".format(TI_id)  
                permy_file_src_path = PERMY_src_path / "TI{}.GRDECL".format(TI_id)  
                permz_file_src_path = PERMZ_src_path / "TI{}.GRDECL".format(TI_id)  
                poro_file_src_path = PORO_src_path / "TI{}.GRDECL".format(TI_id)

                permx_file_dest_path = PERMX_dest_path / "TI{}.GRDECL".format(TI_id)  
                permy_file_dest_path = PERMY_dest_path / "TI{}.GRDECL".format(TI_id)  
                permz_file_dest_path = PERMZ_dest_path / "TI{}.GRDECL".format(TI_id)  
                poro_file_dest_path = PORO_dest_path / "TI{}.GRDECL".format(TI_id) 

                shutil.copy(permx_file_src_path,permx_file_dest_path)
                shutil.copy(permy_file_src_path,permy_file_dest_path)
                shutil.copy(permz_file_src_path,permz_file_dest_path)
                shutil.copy(poro_file_src_path,poro_file_dest_path)

    def built_data_files(self):
        """ built data files that can be used for flow simulations or flow diagnostics """
        shedule = self.setup["shedule"]
        n_TI = self.n_TI
        base_path = self.setup["base_path"]

        destination_path = base_path / '../../Output/training_images/current_run'
        data_path = destination_path / "DATA"

        for model_id in range(n_TI):
            data_file_path = data_path / "TI{}.DATA".format(model_id)

        
            data_file = "RUNSPEC\n\nTITLE\nModel_{}\n\nDIMENS\n--NX NY NZ\n60 60 1 /\n\n--Phases\nOIL\nWATER\n\n--DUALPORO\n--NODPPM\n\n--Units\nMETRIC\n\n--Number of Saturation Tables\nTABDIMS\n1 /\n\n--Maximum number of Wells\nWELLDIMS\n10 100 5 10 /\n\n--First Oil\nSTART\n1 OCT 2017 /\n\n--Memory Allocation\nNSTACK\n100 /\n\n--How many warnings allowed, but terminate after first error\nMESSAGES\n11*5000 1 /\n\n--Unified Output Files\nUNIFOUT\n\n--======================================================================\n\nGRID\n--Include corner point geometry model\nINCLUDE\n'..\INCLUDE\GRID.GRDECL'\n/\n\nACTNUM\n3600*1 /\n\n--Porosity\nINCLUDE\n'..\INCLUDE\PORO\TI{}.GRDECL'\n/\n\n--Permeability\nINCLUDE\n'..\INCLUDE\PERMX\TI{}.GRDECL'\n/\nINCLUDE\n'..\INCLUDE\PERMY\TI{}.GRDECL'\n/\nINCLUDE\n'..\INCLUDE\PERMZ\TI{}.GRDECL'\n/\n\n--Net to Gross\nNTG\n3600*1\n/\n\n--Output .INIT file to allow viewing of grid data in post proessor\nINIT\n\n--======================================================================\n\nPROPS\n\nINCLUDE\n'..\INCLUDE\DP_pvt.inc' /\n\nINCLUDE\n'..\INCLUDE\ROCK_RELPERMS.INC' /\n\n--======================================================================\n\nREGIONS\n\nEQLNUM\n3600*1\n/\nSATNUM\n3600*1\n/\nPVTNUM\n3600*1\n/\n\n--======================================================================\n\nSOLUTION\n\nINCLUDE\n'..\INCLUDE\SOLUTION.INC' /\n\n--======================================================================\n\nSUMMARY\n\nINCLUDE\n'..\INCLUDE\SUMMARY.INC' /\n\n--======================================================================\n\nSCHEDULE\n\nINCLUDE\n'..\INCLUDE\{}.INC' /\n\nEND".format(model_id,model_id,model_id,model_id,model_id,shedule)  
            
            file = open(data_file_path, "w+")
            # write petrelfilepath and licence part into file and seed
            file.write(data_file)

            # close file
            file.close()

    def built_data_file(self,data_file_path,model_id):
        """ built data files that can be used for flow simulations or flow diagnostics """
        shedule = self.setup["shedule"]
        data_file = "RUNSPEC\n\nTITLE\nModel_{}\n\nDIMENS\n--NX NY NZ\n60 60 1 /\n\n--Phases\nOIL\nWATER\n\n--DUALPORO\n--NODPPM\n\n--Units\nMETRIC\n\n--Number of Saturation Tables\nTABDIMS\n1 /\n\n--Maximum number of Wells\nWELLDIMS\n10 100 5 10 /\n\n--First Oil\nSTART\n1 OCT 2017 /\n\n--Memory Allocation\nNSTACK\n100 /\n\n--How many warnings allowed, but terminate after first error\nMESSAGES\n11*5000 1 /\n\n--Unified Output Files\nUNIFOUT\n\n--======================================================================\n\nGRID\n--Include corner point geometry model\nINCLUDE\n'..\INCLUDE\GRID.GRDECL'\n/\n\nACTNUM\n3600*1 /\n\n--Porosity\nINCLUDE\n'..\INCLUDE\PORO\TI{}.GRDECL'\n/\n\n--Permeability\nINCLUDE\n'..\INCLUDE\PERMX\TI{}.GRDECL'\n/\nINCLUDE\n'..\INCLUDE\PERMY\TI{}.GRDECL'\n/\nINCLUDE\n'..\INCLUDE\PERMZ\TI{}.GRDECL'\n/\n\n--Net to Gross\nNTG\n3600*1\n/\n\n--Output .INIT file to allow viewing of grid data in post proessor\nINIT\n\n--======================================================================\n\nPROPS\n\nINCLUDE\n'..\INCLUDE\DP_pvt.inc' /\n\nINCLUDE\n'..\INCLUDE\ROCK_RELPERMS.INC' /\n\n--======================================================================\n\nREGIONS\n\nEQLNUM\n3600*1\n/\nSATNUM\n3600*1\n/\nPVTNUM\n3600*1\n/\n\n--======================================================================\n\nSOLUTION\n\nINCLUDE\n'..\INCLUDE\SOLUTION.INC' /\n\n--======================================================================\n\nSUMMARY\n\nINCLUDE\n'..\INCLUDE\SUMMARY.INC' /\n\n--======================================================================\n\nSCHEDULE\n\nINCLUDE\n'..\INCLUDE\{}.INC' /\n\nEND".format(model_id,model_id,model_id,model_id,model_id,shedule)  
        
        file = open(data_file_path, "w+")
        # write petrelfilepath and licence part into file and seed
        file.write(data_file)

        # close file
        file.close()
    
    def Remove_Comment_Lines(self,data,commenter='--'):

        #Remove comment and empty lines as well as -- Generated : Petrel
        data_lines=data.strip().split('\n')
        newdata=[]
        for line in data_lines:
            if line.startswith(commenter) or not line.strip():
                # skip comments and blank lines
                continue
            
            newdata.append(line)
        return '\n'.join(newdata)
    
    def get_TI_frac_data(self):
        """ extract mean permxzy and fraction of fracutred cells from output file from petrel - a bit tedious """

        base_path = self.setup["base_path"]
        source_path = base_path / "../../Output/training_images/current_run"
        mean_permx_path = source_path / "mean_permx"
        mean_permy_path = source_path / "mean_permy"
        mean_permz_path = source_path / "mean_permz"
        frac_cell_fraction_path = source_path / "frac_cell_fraction"
        print(base_path)
        print("source_path>")
        print(source_path)
        print(mean_permx_path)
        self.mean_permx_all = []
        self.mean_permy_all = []
        self.mean_permz_all = []
        self.frac_cell_fraction_all = []

        for TI_id in range(self.n_TI):

            mean_permx_file_path = mean_permx_path / "TI{}.GRDECL".format(TI_id)  
            mean_permx=open(mean_permx_file_path)
            mean_permx_contents=mean_permx.read()
            mean_permx_contents=self.Remove_Comment_Lines(mean_permx_contents,commenter='--')
            mean_permx_block_dataset=mean_permx_contents.strip().split() #Sepeart input file by slash /
            mean_permx_block_dataset=np.array(mean_permx_block_dataset)
            mean_permx_block_dataset = mean_permx_block_dataset[12]
            mean_permx_value = np.array(mean_permx_block_dataset[5:],dtype=float)
            self.mean_permx_all.append(mean_permx_value)
            mean_permx.close() 

            mean_permy_file_path = mean_permy_path / "TI{}.GRDECL".format(TI_id)  
            mean_permy=open(mean_permy_file_path)
            mean_permy_contents=mean_permy.read()
            mean_permy_contents=self.Remove_Comment_Lines(mean_permy_contents,commenter='--')
            mean_permy_block_dataset=mean_permy_contents.strip().split() #Sepeart input file by slash /
            mean_permy_block_dataset=np.array(mean_permy_block_dataset)
            mean_permy_block_dataset = mean_permy_block_dataset[12]
            mean_permy_value = np.array(mean_permy_block_dataset[5:],dtype=float)
            self.mean_permy_all.append(mean_permy_value)
            mean_permy.close() 

            mean_permz_file_path = mean_permz_path / "TI{}.GRDECL".format(TI_id)  
            mean_permz=open(mean_permz_file_path)
            mean_permz_contents=mean_permz.read()
            mean_permz_contents=self.Remove_Comment_Lines(mean_permz_contents,commenter='--')
            mean_permz_block_dataset=mean_permz_contents.strip().split() #Sepeart input file by slash /
            mean_permz_block_dataset=np.array(mean_permz_block_dataset)
            mean_permz_block_dataset = mean_permz_block_dataset[12]
            mean_permz_value = np.array(mean_permz_block_dataset[5:],dtype=float)
            self.mean_permz_all.append(mean_permz_value)
            mean_permz.close() 

            frac_cell_fraction_file_path = frac_cell_fraction_path / "TI{}.GRDECL".format(TI_id)  
            frac_cell_fraction=open(frac_cell_fraction_file_path)
            frac_cell_fraction_contents=frac_cell_fraction.read()
            frac_cell_fraction_contents=self.Remove_Comment_Lines(frac_cell_fraction_contents,commenter='--')
            frac_cell_fraction_dataset=frac_cell_fraction_contents.strip().split() #Sepeart input file by slash /
            frac_cell_fraction_dataset=np.array(frac_cell_fraction_dataset)
            frac_cell_fraction_dataset = frac_cell_fraction_dataset[12]
            frac_cell_fraction_value = np.array(frac_cell_fraction_dataset[5:],dtype=float)
            self.frac_cell_fraction_all.append(frac_cell_fraction_value)
            frac_cell_fraction.close()

        self.df_all_TI_data["mean_permx"] = self.mean_permx_all
        self.df_all_TI_data["mean_permy"] = self.mean_permy_all     
        self.df_all_TI_data["mean_permz"] = self.mean_permz_all
        self.df_all_TI_data["frac_cell_fraction"] = self.frac_cell_fraction_all
