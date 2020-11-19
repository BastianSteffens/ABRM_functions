############################################################################

import TI_generator 
import numpy as np
import matplotlib.pyplot as plt
import random 
import datetime
import pathlib

############################################################################

def init():

    set_seed = 234
    random.seed(set_seed)

    ###### Set variables for Training Image generation ######
    n_TI = 6
    

    n_fracsets_random_range = [0,4]
    n_fracsets_area_specific_range = [0,4]
    n_fracsets_sampling_style = "uniform"
    P32_total_range = [0.5,0.2]
    P32_total_sampling_style = "normal"
    random_fracs_fraction_range = [0.4,0.1]
    random_fracs_sampling_style = "normal"
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
    property_stats_random = [[[2.1,3.0],[15,25],[50,500],[45,85],[25,50],[0,50]],
                      [[2.1,3.0],[15,25],[50,500],[45,85],[25,50],[0,50]],
                      [[2.1,3.0],[15,25],[50,500],[45,85],[25,50],[0,50]],
                      [[2.1,3.0],[15,25],[50,500],[45,85],[25,50],[0,50]],
                     ]
    property_stats_area_specific = [[[2.1,3.0],[15,25],[50,500],[45,85],[25,50],[0,50]],
                      [[2.1,3.0],[15,25],[50,500],[45,85],[25,50],[0,50]],
                      [[2.1,3.0],[15,25],[50,500],[45,85],[25,50],[0,50]],
                      [[2.1,3.0],[15,25],[50,500],[45,85],[25,50],[0,50]],
                     ]

    # seed
    set_seed = random.randint(0,10000000)
    random.seed(set_seed)
    # models per petrel workflow (limited to 3 atm)
    n_modelsperbatch = 2
    # how many potrel licenses to run at onces
    n_parallel_petrel_licenses = 3
    # which workflow to run in petrel (atm onlz 1 wf)
    runworkflow = "WF_TI_1_generator"   
    petrel_path = "C:/Program Files/Schlumberger/Petrel 2017/Petrel.exe"
    base_path = pathlib.Path(__file__).parent
    output_path = base_path / "../../Output/training_images/TI_1/"
    output_folder = str(datetime.datetime.today().replace(microsecond= 0, second = 0).strftime("%Y_%m_%d_%H_%M"))
    output_file_variables = "variable_settings_saved.pickle"
    folder_path = output_path / output_folder
    file_path = folder_path / output_file_variables

    shedule = "5_spot"
    petrel_on = False

    setup = dict(n_TI = n_TI, n_fracsets_random_range = n_fracsets_random_range, n_fracsets_area_specific_range = n_fracsets_area_specific_range,
                 n_fracsets_sampling_style = n_fracsets_sampling_style,P32_total_sampling_style = P32_total_sampling_style,P32_total_range = P32_total_range,
                 random_fracs_sampling_style = random_fracs_sampling_style,
                 set_seed = set_seed, random_fracs_fraction_range = random_fracs_fraction_range, property_name = property_name,
                 property_sampling_style = property_sampling_style, property_stats_random =property_stats_random,
                 property_stats_area_specific = property_stats_area_specific,property_continuous_discrete = property_continuous_discrete,property_type = property_type,
                 n_modelsperbatch = n_modelsperbatch, runworkflow = runworkflow,n_parallel_petrel_licenses = n_parallel_petrel_licenses,
                 petrel_path = petrel_path,base_path = base_path,folder_path = folder_path,file_path = file_path,
                 shedule = shedule, petrel_on = petrel_on
                )

    ###### Initialize Training image generator ######

    # Call instance of TI generator
    TI_gen = TI_generator.TI_generator(seed = set_seed,setup = setup)

    TI_gen.run_TI_generator()

    print("{} Training Images successfully generated".format(n_TI))




def main():
    init()

if __name__ == "__main__":
    main()


