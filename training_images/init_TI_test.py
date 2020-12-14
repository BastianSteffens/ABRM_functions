############################################################################

import TI_generator 
import TI_run_FD
import numpy as np
import matplotlib.pyplot as plt
import random 
import datetime
import pathlib


############################################################################

def init():

    set_seed = 234343124
    random.seed(set_seed)

    ###### Set variables for Training Image generation ######
    n_TI = 600
    

    n_fracsets_random_range = [4,4]
    n_fracsets_area_specific_range = [4,4]
    n_fracsets_sampling_style = "uniform"
    P32_total_range = [0.01,0.01]
    P32_total_sampling_style = "uniform"
    random_fracs_fraction_range = [0.1,0.1]
    random_fracs_sampling_style = "uniform"
    # ranges P32=[0.001,3] frac_length_scale = [5,25], frac_length_shape = [2.1,2.5], frac_length_max = [100,1000], frac_orient_dip = [0,90], 
    # frac_orient_azimuth = [0,360] if want 2 sets of fracs, both showing 360 but one dipping 15 degrees left, other 15degrees right, i need to give one 180 degree, frac_orient_concentration = [0,100] 100 = all fractures very much algined
    property_name = ["frac_length_shape","frac_length_scale","frac_length_max","frac_orient_dip","frac_orient_azimuth","frac_orient_concentration"]
    property_sampling_style = ["uniform","uniform","uniform","uniform","uniform","uniform","uniform"]
    # next to properties listed above this also has to include individual seed for fracutre networks and also P32 for each fracture network
    property_continuous_discrete = ["continuous","continuous","continuous","continuous","continuous","continuous","continuous","continuous",]
    property_type = [0,0,0,0,0,0,0,0]
    #this has to be done for ever fracture set 
    # first are random fracs.
    # uniform [min,max]
    # normal  [mean,std]
    # gamma etc.
    property_stats_random = [[[2.1,2.1],[5,5],[500,500],[80,80],[300,300],[50,50]],
                             [[2.1,2.1],[5,5],[500,500],[80,80],[300,300],[50,50]],
                             [[2.1,2.1],[5,5],[500,500],[80,80],[300,300],[50,50]],
                             [[2.1,2.1],[5,5],[500,500],[80,80],[300,300],[50,50]],
                     ]
    property_stats_area_specific = [[[2.1,2.1],[5,5],[500,500],[80,80],[300,300],[50,50]],
                             [[2.1,2.1],[5,5],[500,500],[80,80],[300,300],[50,50]],
                             [[2.1,2.1],[5,5],[500,500],[80,80],[300,300],[50,50]],
                             [[2.1,2.1],[5,5],[500,500],[80,80],[300,300],[50,50]],
                     ]


    # seed
    set_seed = random.randint(0,10000000)
    random.seed(set_seed)
    # models per petrel workflow (limited to 3 atm)
    n_modelsperbatch = 2
    # how many potrel licenses to run at onces
    n_parallel_petrel_licenses = 3
    # which workflow to run in petrel (atm onlz 1 wf)
    runworkflow = "WF_TI_crest_generator"   
    petrel_path = "C:/Program Files/Schlumberger/Petrel 2017/Petrel.exe"
    base_path = pathlib.Path(__file__).parent
    output_path = base_path / "../../Output/training_images/TI_crest/"
    output_folder = str(datetime.datetime.today().replace(microsecond= 0, second = 0).strftime("%Y_%m_%d_%H_%M"))
    output_file_variables = "variable_settings_saved.pickle"
    folder_path = output_path / output_folder
    file_path = folder_path / output_file_variables
    pool = 6


    shedule = "5_spot"
    n_shedules = 2
    petrel_on = True

    setup = dict(n_TI = n_TI, n_fracsets_random_range = n_fracsets_random_range, n_fracsets_area_specific_range = n_fracsets_area_specific_range,
                 n_fracsets_sampling_style = n_fracsets_sampling_style,P32_total_sampling_style = P32_total_sampling_style,P32_total_range = P32_total_range,
                 random_fracs_sampling_style = random_fracs_sampling_style,
                 set_seed = set_seed, random_fracs_fraction_range = random_fracs_fraction_range, property_name = property_name,
                 property_sampling_style = property_sampling_style, property_stats_random =property_stats_random,
                 property_stats_area_specific = property_stats_area_specific,property_continuous_discrete = property_continuous_discrete,property_type = property_type,
                 n_modelsperbatch = n_modelsperbatch, runworkflow = runworkflow,n_parallel_petrel_licenses = n_parallel_petrel_licenses,
                 petrel_path = petrel_path,base_path = base_path,folder_path = folder_path,file_path = file_path,
                 shedule = shedule, petrel_on = petrel_on, pool = pool, n_shedules = n_shedules
                )

    ###### Initialize Training image generator ######

    # Call instance of TI generator
    TI_gen = TI_generator.TI_generator(seed = set_seed,setup = setup)

    TI_gen.run_TI_generator()

    print("{} Training Images successfully generated".format(n_TI))

    TI_run = TI_run_FD.TI_run_FD(setup = setup)
    TI_run.TI_run_FD_runner()

    print("Finished running flow diagnostics on training images")
    
    # setup = 1
    # dataset = "2020_11_20_16_03"

    # TI_data = TI_selection(setup = setup, dataset = dataset)
    # TI_data.clustering_tof_or_TI_props(n_neighbors = 2,cluster_parameter = "tof")
    # TI_data.cluster_TI_selection()
    # TI_data.save_best_clustered_TIs()
def main():
    init()

if __name__ == "__main__":
    main()


