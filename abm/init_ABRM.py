############################################################################
import pickle
import random as rn
import numpy as np
import os 
from typing import NoReturn, Text
import time
import datetime

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
    TURNS = 6
    AGENTSPERTURN = 1
    RANDOMAGENTSGENRATIONBOTTOM = 0.75
    RATIOTRACKEDAGENTS = 1.
    SEED = 3721286
    all_vals_len = 200*100

    # set up TI_zone env
    # generate 2D mesh
    x = np.arange(0,200+1,1,)
    y = np.arange(0,100+1,1)
    x_grid, y_grid = np.meshgrid(x,y)
    #get cell centers of mesh
    x_cell_center = x_grid[:-1,:-1]+0.5
    y_cell_center = y_grid[:-1,:-1]+0.5
    TI_zones = []
    for y in range(x_cell_center.shape[0]):
        for x in range(len(x_cell_center[1])):
            if 70 <= x_cell_center[y,x] <=130:
                TI_zones.append(1)
            else:
                TI_zones.append(0)
    TI_zones = np.array(TI_zones)
    TI_zones = TI_zones.reshape((200,100,1))

    # create curve and save resultign desired LC
    Phi_points_target = np.linspace(0, 1, num=11, endpoint=True)
    # F_points_target = np.array([0, 0.2, 0.35, 0.6, 0.65, 0.7, 0.8, 0.85, 0.9, 0.95, 1])
    # F_points_target = np.array([0, 0.7, 0.85, 0.9, 0.95, 0.96, 0.97, 0.98, 0.99, 0.999, 1])
    F_points_target = np.array([0, 0.3, 0.45, 0.6, 0.68, 0.75, 0.8, 0.85, 0.9, 0.95, 1])

    output_folder = str(datetime.datetime.today().replace(microsecond= 0, second = 0).strftime("%Y_%m_%d_%H_%M"))

    
    seed_everything(SEED)



    t1 = time.time()

    model = Model(env = TI_zones, 
                number_of_turns=TURNS, number_of_starting_agents = 6,new_agent_every_n_turns = 100,max_number_agents = 10,
                ratio_of_tracked_agents = 1.,number_training_image_zones = 2, number_training_images_per_zone = 4,output_folder = output_folder,
                Phi_points_target=Phi_points_target,F_points_target=F_points_target,max_number_of_position_tests = 3 ,n_processes = None,neighbourhood_radius = 2, neighbourhood_search_step_size = 1,
                )
    model.run()
    print("Simulation took {0:2.2f} seconds".format(time.time()-t1))
    test_df =model.get_final_results()
    print(test_df.head(20))
def main():
    init()

if __name__ == "__main__":
    main()


