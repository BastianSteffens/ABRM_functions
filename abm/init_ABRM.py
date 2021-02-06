############################################################################

# import pyswarms_modified as ps       # PSO package in Python
# import matplotlib.pyplot as plt
# from pyswarms.utils.plotters import (plot_cost_history, plot_contour, plot_surface)
# import random 
# import pickle
# import datetime
# import lhsmdu
# import pathlib
# from pyswarms_modified.particles.particle import particle
import random as rn
import numpy as np
import os 
from typing import NoReturn, Text
import time

#import local scripts
from Agent import Agent
from Model import Model
from Grid import Grid
############################################################################
    ## Set seed for random generation
def seed_everything(seed : int) -> NoReturn :
    """To set the see for all potential random number usages. If you are using 
    any other package which might use seed for random generation add it here"""
    rn.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    #tf.random.set_seed(seed)  ##uncomment if you are using tensorflow

def init():
    
    
    ### set parameters 
    TURNS = 20
    AGENTSPERTURN = 5
    RANDOMAGENTSGENRATIONBOTTOM = 0.75
    RATIOTRACKEDAGENTS = 1.
    SEED = 3721286
    all_vals_len = 200*100
    TI_zone = np.zeros((all_vals_len,4))
    TI_zone[:10000,2] = 0
    TI_zone[10000:,2] = 1
    TI_zone[15000:,2] = 2
    TI_zone[:3000,2] = 3
    TI_zones = TI_zone[:,2].reshape((200,100,1))
    # preference_L = [[[2., 1., 0.],[2., 1., 0.], [2., 1., 0.]],
    #             [[2., 1., 0.], [3., 0., 0.], [2., 1., 0.]],
    #             [[2., 1., 0.], [2., 1., 0.], [2., 1., 0.]]]
    # PREFERENCE_MATRIX = np.array([np.array(np.array([Lii for Lii in Li])) for Li in preference_L])

    # stochastic_L = [[[0.075, 0.0375, 0.], [0.075, 0.0375, 0.], [0.075, 0.0375, 0.]],
    #             [[0.075, 0.0375, 0.], [0.1, 0., 0.], [0.075, 0.0375, 0.]],
    #             [[0.075, 0.0375, 0.], [0.075, 0.0375, 0.], [0.075, 0.0375, 0.]]]
    # STOCHASTIC_MATRIX = np.array([np.array(np.array([Lii for Lii in Li])) for Li in stochastic_L])
    
    
    seed_everything(SEED)



    t1 = time.time()

    model = Model(env = TI_zones, 
                  number_of_turns=TURNS, number_of_starting_agents = 5,new_agent_every_n_turns = 1,max_number_agents = 10,
                  ratio_of_tracked_agents = 1.,number_training_image_zones = 2, number_training_images_per_zone = 20
                 )
    model.run()
    print("Simulation took {0:2.2f} seconds".format(time.time()-t1))
    test_df =model.get_final_results()
    print(test_df.head(20))
def main():
    init()

if __name__ == "__main__":
    main()


