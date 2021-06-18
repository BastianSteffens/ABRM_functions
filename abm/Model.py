from numpy.random import randint
from numpy.random import choice
import numpy as np
import pandas as pd
import random as rn
import os
import time
from datetime import datetime
import copy
import numpy as np
from shapely.strtree import STRtree
from shapely.geometry import Point
from shapely.geometry import Polygon
from shapely.ops import nearest_points
from shapely.geometry import MultiPoint

from geovoronoi import voronoi_regions_from_coords
from geovoronoi import polygon_lines_from_voronoi
from collections import Counter
from GRDECL_file_reader.GRDECL2VTK import*
import pathlib
import matlab.engine
from scipy import interpolate
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
import multiprocessing as mp

from Agent import Agent
from Grid import Grid

class Model():
    """Manage an ensemble of the agents during the simmulation"""

    def __init__(self, 
                env,F_points_target, Phi_points_target,output_folder, 
                number_of_turns=2000,number_of_starting_agents = 5,new_agent_every_n_turns = 1,new_agents_per_n_turns = 1,max_number_agents = 10,max_number_of_position_tests = 30,
                ratio_of_tracked_agents = 1.,number_training_image_zones = 2, number_training_images_per_zone = 20,n_processes = 6,neighbourhood_radius = 2, neighbourhood_search_step_size = 2,
                ):

        self.base_path = pathlib.Path(__file__).parent
        self.output_folder = output_folder
        self.make_directories()
        self.n_processes = n_processes 

        self.env = env
        (self.X, self.Y, self.Z) = self.env.shape
        print('Shape of your environment: ({}, {}, {})'.format(self.X, self.Y, self.Z))

        self.number_training_image_zones = number_training_image_zones
        self.number_training_images_per_zone = number_training_images_per_zone
        self.training_images = np.linspace((1,1),(number_training_images_per_zone,number_training_images_per_zone),number_training_images_per_zone)
        self.load_training_images()

        self.Phi_points_target = Phi_points_target
        self.F_points_target = F_points_target

        self.initiate_grid(neighbourhood_radius = neighbourhood_radius, neighbourhood_search_step_size = neighbourhood_search_step_size)
        self.init_TI_zone_grid()

        self.istep = 0
        self.current_id = -1

        self.number_of_turns = number_of_turns
        self.new_agent_every_n_turns = new_agent_every_n_turns
        self.new_agents_per_n_turns = new_agents_per_n_turns
        self.number_of_starting_agents = number_of_starting_agents
        self.max_number_agents = max_number_agents
        self.max_number_of_position_tests = max_number_of_position_tests
        self.when_new_agent = 0
        self.ratio_of_tracked_agents = ratio_of_tracked_agents

        self.active_agents = []
        self.dead_agents = []
        
        self.resultsDF = pd.DataFrame(columns=['Start_x', 'Start_y', 'Start_z','End_x',
                                            'End_y','End_z','End_turn']).astype(dtype={'Start_x': np.int16, 'Start_y': np.int16, 
                                                                                        'Start_z': np.int16,'End_x': np.int16,
                                                                                        'End_y': np.int16,'End_z': np.int16,
                                                                                        'End_turn': np.int16})
        self.stop_simulation = False

        # start mrst
        self.matlab_runner = matlab.engine.start_matlab()
        # run matlab and mrst
        self.matlab_runner.matlab_starter(nargout = 0)

    def initiate_grid(self,neighbourhood_radius,neighbourhood_search_step_size):
        t_igrid = time.time()
        print('=======Initiating grid=========')
        self.grid = Grid(self.X, self.Y, self.Z,neighbourhood_radius = neighbourhood_radius, neighbourhood_search_step_size = neighbourhood_search_step_size)

        print("Grid initiated! -  took {0:2.2f} seconds".format(time.time()-t_igrid))

    def generate_new_agent(self):

        agentID = self.next_id()

        ###initiate location
        StartPos = self.grid.find_empty_location()
        if StartPos == 'No Empty Cells':
            self.stop_simulation = True
            # break
        if self.ratio_of_tracked_agents == 1.:
            #create tracked agent
            agent = Agent(agentID, StartPos, self.istep)
        elif self.ratio_of_tracked_agents == 0.:
            #create not tracked agent
            agent = Agent(agentID, StartPos, self.istep, isTracked=False)
        else:
            if rn.random()<self.ratio_of_tracked_agents:
                #create tracked agent
                agent = Agent(agentID, StartPos, self.istep)
            else:
                #create not tracked agent
                agent = Agent(agentID, StartPos, self.istep, isTracked=False)
        self.grid.initiate_agent_on_grid(agent)
        self.active_agents.append(agent)
        print("generated new agent")

    def kill_agent(self, agent, next_turn=False):
        if next_turn:
            agent.kill(self.istep+1)
        else:
            agent.kill(self.istep)
        for agent_pos_in_list,agent_to_remove in enumerate(self.active_agents):
            if agent.id == agent_to_remove.id:
                self.active_agents.pop(agent_pos_in_list)
        self.dead_agents.append(agent)
    
    def move_agents(self):
        for iagent, agent in enumerate (self.active_agents):
            copy_active_agents = copy.deepcopy(self.active_agents)
            agent_to_move = copy.deepcopy(agent)

            print("moving agent {}".format(agent.id), end="\r")
            new_pos,training_image = self.get_agent_movement(agent_to_move,copy_active_agents)
            if new_pos[0] ==None:
                print("Killed agent {}        ".format(agent.id))
                self.kill_agent(agent)

            self.grid.move_agent(agent, new_pos)
            self.generate_voronoi_regions()
            self.assign_training_image_to_agent(training_image = training_image,agent_new_TI = agent)
            self.generate_reservoir_model(model_type="final_model_per_iteration")
            FD_performance =  self.run_FD(model_type = "final_model_per_iteration")
            misfit = self.calculate_misfit(FD_performance)
            for agent in self.active_agents:
                agent.update_agent_misfit(misfit)
   
    def get_agent_movement(self, agent_to_move,copy_active_agents):
        free_neighborhood_coord = self.grid.get_empty_neighborhood(agent_to_move.pos,len(copy_active_agents))

        if rn.random() >= 0.95:
            print("random move          ")

            # random movement and TI selection
            i_new_pos = randint(0,len(free_neighborhood_coord))
            new_pos = free_neighborhood_coord[int(i_new_pos)]
            new_TI_zone = randint(0,self.number_training_image_zones)
            new_TI_no = randint(0,self.number_training_images_per_zone)
            training_image = []
            training_image.append(new_TI_zone)
            training_image.append(new_TI_no)

        else:
            # Setup Pool of processes for parallel evaluation
            if self.n_processes == None:
                agent_evaluation,misfit_all = self.evaluate_agent_position(free_neighborhood_coord = free_neighborhood_coord,agent_to_move = agent_to_move,copy_active_agents = copy_active_agents)
            else:
                agent_evaluation,misfit_all = self.evaluate_agent_position_parallel(free_neighborhood_coord = free_neighborhood_coord,agent_to_move = agent_to_move,copy_active_agents = copy_active_agents)
            
            # evaluate best position for agent based on misfit and goodness of TI fit
            best_new_position = self.get_best_quality_fit_agent(agent_evaluation,misfit_all)

            new_pos = agent_evaluation[int(best_new_position)][0:3]
            new_TI_zone = agent_evaluation[int(best_new_position)][3]
            new_TI_no = agent_evaluation[int(best_new_position)][4]
            training_image = []
            training_image.append(new_TI_zone)
            training_image.append(new_TI_no)

        return(new_pos,training_image)

    def get_best_quality_fit_agent(self,agent_evaluation,misfit_all):
        """ combine misfit with how good the selected training image fits in with its surroundings"""

        #get rank of each misfit
        misfit_quality = []
        for i in range(len(misfit_all)):
            if misfit_all[i] <= 0.1:
                misfit_quality.append(0)
            elif 0.1 < misfit_all[i] <= 0.2:
                misfit_quality.append(1)
            elif 0.2 < misfit_all[i] <= 0.3:
                misfit_quality.append(2)
            elif 0.3 < misfit_all[i] <= 0.4:
                misfit_quality.append(3)
            elif 0.4 < misfit_all[i] <= 0.5:
                misfit_quality.append(4)
            elif 0.5 < misfit_all[i] <= 0.6:
                misfit_quality.append(5)
            elif 0.6 < misfit_all[i] <= 0.7:
                misfit_quality.append(6)
            elif 0.8 < misfit_all[i] <= 0.9:
                misfit_quality.append(7)
            elif 0.9 < misfit_all[i] <= 1.0:
                misfit_quality.append(8)
            else:
                misfit_quality.append(9)
        misfit_quality = np.array(misfit_quality).reshape(-1,1)
        scaler = MinMaxScaler()
        scaler.fit(misfit_quality)
        misfit_quality_scaled = scaler.transform(misfit_quality)
        misfit_quality_scaled = misfit_quality_scaled * 0.2

        #get quality level of each TI
        TI_quality = []
        for i in range(len(agent_evaluation)):
            TI_quality.append(agent_evaluation[i][6])
        TI_quality = np.array(TI_quality).reshape(-1,1)
        scaler = MinMaxScaler()
        scaler.fit(TI_quality)
        TI_quality_scaled = scaler.transform(TI_quality)
        TI_quality_scaled = TI_quality_scaled * 0.6

        # preference to move towards a TI zone boundary
        move_quality = []
        for i in range(len(agent_evaluation)):
            move_quality.append(agent_evaluation[i][8])
        move_quality = np.array(move_quality).reshape(-1,1)
        scaler = MinMaxScaler()
        scaler.fit(move_quality)
        move_quality_scaled = scaler.transform(move_quality)
        move_quality_scaled = move_quality_scaled * 0.2

        # add msifit and TI quality and move quality. lowest value is best. can also think about making this into probability to sample from
        best_quality_fit = np.add(misfit_quality_scaled, TI_quality_scaled)
        best_quality_fit = np.add(best_quality_fit,move_quality_scaled)

        return np.argmin(best_quality_fit)  


    def evaluate_agent_position(self,free_neighborhood_coord,agent_to_move,copy_active_agents):
        """single core evaluation of every possible position that particle could take""" 
        all_possible_positions_training_images = []
        # generate all possible training images and coords for each position
        model_index = 0

        model_index_list = []
        while len(all_possible_positions_training_images) < self.max_number_of_position_tests:
            coord = rn.sample(free_neighborhood_coord,1)[0]
            copy_active_agents = self.generate_voronoi_regions(coord = coord,agent_to_move = agent_to_move,copy_active_agents = copy_active_agents)
            possible_training_images = self.get_possible_training_images(copy_active_agents = copy_active_agents,agent_to_move = agent_to_move)
            training_image = rn.sample(possible_training_images,1)[0]
            copy_active_agents = self.assign_training_image_to_agent(copy_active_agents = copy_active_agents,agent_to_move = agent_to_move, training_image=training_image)
            self.generate_reservoir_model(copy_active_agents = copy_active_agents,agent_to_move = agent_to_move,model_type = "training_image_testing", model_index = model_index, training_image = training_image)
            all_possible_positions_training_images.append([coord[0],coord[1],coord[2],training_image[0],training_image[1],model_index,training_image[2]])
            model_index_list.append(model_index)
            model_index += 1

        # add current position to evaluation.
        coord = agent_to_move.pos
        copy_active_agents = self.generate_voronoi_regions(coord = coord,agent_to_move = agent_to_move,copy_active_agents = copy_active_agents)
        possible_training_images = self.get_possible_training_images(copy_active_agents = copy_active_agents,agent_to_move = agent_to_move,get_current_TI = True)
        training_image = rn.sample(possible_training_images,1)[0]
        copy_active_agents = self.assign_training_image_to_agent(copy_active_agents = copy_active_agents,agent_to_move = agent_to_move, training_image=training_image)
        self.generate_reservoir_model(copy_active_agents = copy_active_agents,agent_to_move = agent_to_move,model_type = "training_image_testing", model_index = model_index, training_image = training_image)
        all_possible_positions_training_images.append([coord[0],coord[1],coord[2],training_image[0],training_image[1],model_index,training_image[2]])
        model_index_list.append(model_index)
        model_index += 1

        # filter dublicates
        all_possible_positions_training_images_filtered = []
        for test_position in all_possible_positions_training_images:
            if test_position not in all_possible_positions_training_images_filtered:
                all_possible_positions_training_images_filtered.append(test_position)
        all_possible_positions_training_images = all_possible_positions_training_images_filtered

        # random selection of n szenarios to test out.
        all_possible_positions_training_images_random = rn.sample(all_possible_positions_training_images, len(all_possible_positions_training_images))     
        model_index_list_random = rn.sample(model_index_list,len(model_index_list))
        
        # check if available positions is larger than max to run
        if self.max_number_of_position_tests+1>len(model_index_list_random):
            number_test_runs = len(model_index_list_random)
        else:
            number_test_runs = self.max_number_of_position_tests
        number_test_runs = len(all_possible_positions_training_images_random)
        model_index_list_random = []

        all_possible_positions_training_images_random = all_possible_positions_training_images[0:number_test_runs]
        for i in range(number_test_runs):
            model_index_list_random.append(all_possible_positions_training_images_random[i][5])
        
        agent_evaluation = []
        misfit_all = []
        # run flow diagnostics 
        FD_performance_all = self.run_FD(model_type = "training_image_testing",index = None,training_image = training_image,number_test_runs = number_test_runs,models_to_run  = model_index_list_random)
        for i, model_id in enumerate(model_index_list_random):
            FD_performance = FD_performance_all[FD_performance_all.model_id==model_id]
            misfit = self.calculate_misfit(FD_performance)
            all_possible_positions_training_images_random[i].append(misfit)
            misfit_all.append(misfit)
        
        # check distance to closest TI zone boundary
        for i in range(len(model_index_list_random)):
            if abs(all_possible_positions_training_images_random[i][1]-70)<abs(all_possible_positions_training_images_random[i][1]-130):
                distance_to_a_boundary = abs(all_possible_positions_training_images_random[i][1]-70)
            else:
                distance_to_a_boundary = abs(all_possible_positions_training_images_random[i][1]-130)
            all_possible_positions_training_images_random[i].append(distance_to_a_boundary)

        
        return all_possible_positions_training_images_random,misfit_all

    def evaluate_agent_position_parallel_calculation(self,ti_id):
        """ calculation of FD for parallel processing """
        test_run = self.agent_model_configuration[ti_id]
        coord = [test_run[0],test_run[1],test_run[2]]
        training_image = [test_run[3],test_run[4]]
        index = test_run[5]
        FD_performance = self.run_FD(model_type = "training_image_testing",index = index,training_image = training_image)
        misfit = self.calculate_misfit(FD_performance)
        agent_evaluation = [coord[0],coord[1],coord[2],training_image[0],training_image[1],misfit]
        return agent_evaluation

    def evaluate_agent_position_parallel(self,free_neighborhood_coord,agent_to_move,copy_active_agents):
        """multi core evaluation of every possible position that particle could take""" 
        

        all_possible_positions_training_images = []
        
        # generate all possible training images and coords for each position
        for index,coord in enumerate(free_neighborhood_coord):
            copy_active_agents = self.generate_voronoi_regions(coord = coord,agent_to_move = agent_to_move,copy_active_agents = copy_active_agents)
            possible_training_images = self.get_possible_training_images(copy_active_agents = copy_active_agents,agent_to_move = agent_to_move)
            for i, training_image in enumerate(possible_training_images):
                all_possible_positions_training_images.append([coord[0],coord[1],coord[2],training_image[0],training_image[1],index,agent_to_move,copy_active_agents])

        # random selection of n szenarios to test out.
        all_possible_positions_training_images_random = rn.sample(all_possible_positions_training_images, len(all_possible_positions_training_images))     
        
        # check if available positions is larger than max to run
        if self.max_number_of_position_tests>len(all_possible_positions_training_images_random):
            number_test_runs = len(all_possible_positions_training_images_random)
        else:
            number_test_runs = self.max_number_of_position_tests
        
        self.agent_model_configuration = []
        # built all models for FD simulation
        for i in range(number_test_runs):
            test_run = all_possible_positions_training_images_random[i]
            coord = [test_run[0],test_run[1],test_run[2]]
            index = test_run[5]
            training_image = [test_run[3],test_run[4]]
            copy_active_agents = self.generate_voronoi_regions(coord = coord,agent_to_move = agent_to_move,copy_active_agents = copy_active_agents)
            copy_active_agents = self.assign_training_image_to_agent(copy_active_agents = copy_active_agents,agent_to_move = agent_to_move, training_image=training_image)
            self.generate_reservoir_model(copy_active_agents = copy_active_agents,agent_to_move = agent_to_move,model_type = "training_image_testing", model_index = index, training_image = training_image)
            self.agent_model_configuration.append([coord[0],coord[1],coord[2],training_image[0],training_image[1],index])
        
        # run FD in multiprocessing
        pool = None if self.n_processes is None else mp.Pool(self.n_processes)
        agent_evaluation = pool.map(self.evaluate_agent_position_parallel_calculation,range(number_test_runs))
        pool.close()
        pool.join()

        misfit_all = []
        for misfit in agent_evaluation:
            misfit_all.append(misfit[5])
    
        return agent_evaluation,misfit_all


    def step(self, ):
        print('=======Turn {}========'.format(self.istep))
        self.istep += 1

        if self.new_agent_every_n_turns == self.when_new_agent:
            if len(self.active_agents)< self.max_number_agents:

                for i in range(self.new_agents_per_n_turns):
                    self.generate_new_agent()
                self.when_new_agent = 0
            else:
                print("max number of agents generated - wait till agent dies")

        self.move_agents()
        print('Moving agents done                  ')
        print('Generating agents done              ')

        self.generate_voronoi_regions()
        self.assign_training_image_to_agent()
        self.generate_reservoir_model(model_type="final_model_per_iteration")
        FD_performance =  self.run_FD(model_type = "final_model_per_iteration")
        misfit = self.calculate_misfit(FD_performance)
        print("Misfit: {}".format(misfit))
        for agent in self.active_agents:
            agent.update_agent_misfit(misfit)
        self.track_agents()

        self.when_new_agent +=1
        self.grid.layer = self.Z-1 # not sure what this does.

        # shuffle agent list for random picking of new agents
        self.active_agents = rn.sample(self.active_agents,len(self.active_agents))
    
    def run(self):

        #init model
        self.init_reservoir_model()

        for _ in range(self.number_of_turns):
            self.step()
            if self.stop_simulation:
                print('Simulation stoped because environment is full')
                break
    
    def track_agents(self):
        "track position etc of each agent at end of each iteration"
        for agent in self.active_agents:
            agent.track_agent()

    def get_all_tracks(self):
        """ Extract the tracking path of all the agents during the simulation in numpy.array of shape (number of agent, number of turns, 3 coordinates) """
        track_file = np.empty((self.current_id+1, self.number_of_turns+1  , 7))
        track_file[:] = np.nan
        for agent in self.active_agents:
            if agent.isTracked:
                track_file[agent.id, agent.start_iteration:, :] = agent.track
        for agent in self.dead_agents:
            if agent.isTracked:
                track_file[agent.id, agent.start_iteration:agent.dead_iteration, :] = agent.track
        return track_file

    def get_final_results(self):
        """ Extract ['AgentID', 'Start_x', 'Start_y', 'Start_z','End_x', 'End_y','End_z'] of all the agents in a pandas.DataFrame """
        #iterate on agent starting and final position
        Agent_postions = []
        for agent in self.active_agents:
            Agent_postions.append([agent.id, agent.start_x, agent.start_y, agent.start_z, agent.pos[0], agent.pos[1], agent.pos[2],agent.misfit])
        for agent in self.dead_agents:
            Agent_postions.append([agent.id, agent.start_x, agent.start_y, agent.start_z, agent.pos[0], agent.pos[1], agent.pos[2],agent.misfit])
        self.resultsDF = pd.DataFrame(Agent_postions, columns=['AgentID', 'Start_x', 'Start_y', 'Start_z','End_x', 'End_y','End_z','misfit'])
        return self.resultsDF

    def get_agent_status(self):
        """ Extract ['AgentID', 'Final_status', 'Start_iteration', 'Dead_iteration'] of all the agents in a pandas.DataFrame """
        Agent_status = []
        for agent in self.active_agents:
            Agent_status.append([agent.id, agent.status, agent.start_iteration, None])
        for agent in self.dead_agents:
            Agent_status.append([agent.id, agent.status, agent.start_iteration, agent.dead_iteration])
        self.statusDF = pd.DataFrame(Agent_status, columns=['AgentID', 'Final_status', 'Start_iteration', 'Dead_iteration'])
        return self.statusDF
    
    def next_id(self) -> int:
        """ Return the next unique ID for agents, increment current_id"""
        self.current_id += 1
        return self.current_id

    def init_reservoir_model(self):
        """ initialize reservoir model with first set of agents and according voronoi teselation"""
        for agent in range(self.number_of_starting_agents):
            self.generate_new_agent()
        
        self.generate_voronoi_regions()
        self.assign_training_image_to_agent()
        self.generate_reservoir_model(model_type="final_model_per_iteration")
        FD_performance =  self.run_FD(model_type = "final_model_per_iteration")
        misfit = self.calculate_misfit(FD_performance)
        print("Misfit: {}".format(misfit))
        for agent in self.active_agents:
            agent.update_agent_misfit(misfit)
        self.track_agents()
    
    def init_TI_zone_grid(self):
        """ Grid that is later used to assign each point of model to a polygon from vornoi tesselation"""
        # #define grid and  position initianinon points of n polygons

        self.TI_grid = Polygon([(0, 0), (0, self.Y), (self.X, self.Y), (self.X, 0)])

        # generate 2D mesh
        x = np.arange(0,self.X+1,1,)
        y = np.arange(0,self.Y+1,1)
        x_grid, y_grid = np.meshgrid(x,y)

        #get cell centers of mesh
        x_cell_center = x_grid[:-1,:-1]+0.5
        y_cell_center = y_grid[:-1,:-1]+0.5

        cell_center_which_polygon = np.zeros(len(x_cell_center.flatten()))

        # array to assign polygon id [ last column] to cell id [first 2 columns]
        self.TI_grid_all_cell_center = np.column_stack((x_cell_center.flatten(),y_cell_center.flatten(),cell_center_which_polygon))

        #shapely points of each point of the TI_grid
        self.TI_grid_points = [Point(np.array([self.TI_grid_all_cell_center[i,0],self.TI_grid_all_cell_center[i,1]])) for i in range(len(self.TI_grid_all_cell_center))]
        #index of each shapely point
        self.index_by_id = dict((id(points), i) for i, points in enumerate(self.TI_grid_points))
        #list of TI_grid_points shapely
        self.list_TI_grid_points = []     
        for i in range(len(self.TI_grid_points)):
            self.list_TI_grid_points.append(self.index_by_id[id(self.TI_grid_points[i])])
        #speed up iteration through points in get_voronoi_regions function
        self.tree_TI_grid_points = STRtree(self.TI_grid_points)

    def generate_voronoi_regions(self,coord= None,agent_to_move = None,copy_active_agents = None):
        """ split up model into voronoi regoins, with each agent marking the center of a voronoi region"""
        voronoi_x = []
        voronoi_y = []
        voronoi_z = []
        
        #use current agent positioning to generate voronoi regions
        if agent_to_move == None:
            # get all active agent locations 
            for agent in self.active_agents:
                voronoi_x.append(agent.pos[0])
                voronoi_y.append(agent.pos[1])
                voronoi_z.append(agent.pos[2])

        # use coord position for agent that is moved around to test for best position
        else:
            for agent in copy_active_agents:
                if agent_to_move.id == agent.id:
                    if coord[0] == None: # agent got removed in test szenario
                        agent.move_to(coord)
                        agent_to_move.move_to(coord)

                    else:
                        voronoi_x.append(coord[0])
                        voronoi_y.append(coord[1])
                        voronoi_z.append(coord[2])
                        #update agents position
                        agent.move_to(coord)
                        agent_to_move.move_to(coord)
                else:
                    voronoi_x.append(agent.pos[0])
                    voronoi_y.append(agent.pos[1])
                    voronoi_z.append(agent.pos[2])


        # use these points to built a voronoi tesselation
        voronoi_x = np.array(voronoi_x)
        voronoi_y = np.array(voronoi_y)
        voronoi_points = np.vstack((voronoi_x,voronoi_y)).T
        # voronoi_z = np.array(voronoi_z)

        # get voronoi regions
        poly_shapes, pts, poly_to_pt_assignments = voronoi_regions_from_coords(voronoi_points, self.TI_grid,farpoints_max_extend_factor = 30)
        # get polygons back into correct order with zip and sorted
        poly_to_point = []
        for i in range(len(poly_to_pt_assignments)):
            poly_to_point.append(poly_to_pt_assignments[i][0])
        zipped_lists = zip(poly_to_point,poly_shapes)
        sorted_zipped_lists = sorted(zipped_lists)
        poly_shapes = [element for _, element in sorted_zipped_lists]

        # figure out which points of grid lie in which polygon
        points_in_polygon = []

        for voronoi_polygon_id in range(len(poly_shapes)):
            points_in_polygon.append([[point.x, point.y,self.index_by_id[id(point)]] for point in self.tree_TI_grid_points.query(poly_shapes[voronoi_polygon_id]) if point.intersects(poly_shapes[voronoi_polygon_id])])
       # if points not within polygon due to vertices problem of shapely (point lies on vertices --> point undetected)
        points_in_polygon_temp = [item for sublist in points_in_polygon for item in sublist]
        detected_points = [row[2] for row in points_in_polygon_temp]

        missed_points =list(set(self.list_TI_grid_points) - set(detected_points))

        shift = 1e-9
        while len(missed_points)!=0:
            for points in missed_points:
                point = self.TI_grid_points[points]
                point_shifted = Point(point.x+shift,point.y+shift)
                for voronoi_polygon_id in range(len(poly_shapes)):
                    if point_shifted.within(poly_shapes[voronoi_polygon_id]):
                        point_stats = [point.x,point.y,points]
                        points_in_polygon[voronoi_polygon_id].append(point_stats)
            points_in_polygon_temp = [item for sublist in points_in_polygon for item in sublist]
            detected_points = [row[2] for row in points_in_polygon_temp]

            missed_points =list(set(self.list_TI_grid_points) - set(detected_points))
            shift += 1e-5 

        #id that every polygon has that points belong to. position 4
        polygon_id = []
        for i in range(len(poly_shapes)):
            polygon_id.extend([i]*len(points_in_polygon[i]))
        points_in_polygon = [item for sublist in points_in_polygon for item in sublist]
        for i in range(len(points_in_polygon)):
            points_in_polygon[i].append(polygon_id[i])

        # assign index to list. need this later. position 5
        for i in range(len(points_in_polygon)):
            points_in_polygon[i].append(i)

        # some points are assigned to multiple polygons. dropping dublicate point ids
        points_in_polygon_no_dublicates = []
        points_in_polygon_no_dublicates_set  = set()
        for point in points_in_polygon:
            point_id = tuple(point)[2]
            if point_id not in points_in_polygon_no_dublicates_set:
                points_in_polygon_no_dublicates.append(point)
                points_in_polygon_no_dublicates_set.add(point_id)

        #sort list by point_id
        points_in_polygon_no_dublicates.sort(key=lambda x: x[2])

        #if still not all points found, somethings is wrong. current fix: append last point to list again
        while len(points_in_polygon_no_dublicates)!= len(self.env.flatten()):
            print("problem with polygon to point assignemnt. temporary fix")
            points_in_polygon_no_dublicates.append(points_in_polygon_no_dublicates[-1])
            print(len(points_in_polygon_no_dublicates))

        #append TI_zone that each point belongs to to list. position 6
        TI_zones = self.env.flatten()
        for i in range(len(points_in_polygon_no_dublicates)):
            points_in_polygon_no_dublicates[i].append(TI_zones[i])

        #sort list by index
        points_in_polygon_no_dublicates.sort(key=lambda x: x[3])

        # what TI_zone dominates in which polygon
        TI_zone_most_common_all =  []
        for voronoi_polygon_id in range(len(poly_shapes)):
            TI_counter_per_polygon = []
            for point in points_in_polygon_no_dublicates:
                polygon_id = point[3]
                if polygon_id == voronoi_polygon_id:
                    TI_counter_per_polygon.append(point[5])
            if not TI_counter_per_polygon:
                TI_most_common = np.random.randint(len(poly_shapes)+1)
            else:
                TI_most_common = Counter(TI_counter_per_polygon).most_common(1)[0][0]
      
            TI_zone_most_common_all.extend([TI_most_common]*len(TI_counter_per_polygon))

        #append TI_zone taht is most common to polygon that point is in. position 7
        for i in range(len(points_in_polygon_no_dublicates)):
            points_in_polygon_no_dublicates[i].append(TI_zone_most_common_all[i])

        # what agent what TI_zone assinged
        TI_zone_per_agent = []
        TI_zone_per_agent_set  = set()
        for point in points_in_polygon_no_dublicates:

            polygon_id = tuple(point)[3]

            if polygon_id not in TI_zone_per_agent_set:
                TI_zone_per_agent.append(point[6])
                TI_zone_per_agent_set.add(polygon_id)
        
        # assign polygons to individual agents
        if agent_to_move == None:
            for index,agent in enumerate(self.active_agents):
                agent.update_agent_properties(polygon_id = index,polygon = poly_shapes[index],polygon_area = poly_shapes[index].area,TI_zone_assigned = TI_zone_per_agent[index])
        else:
            index = -1
            for i, agent in enumerate(copy_active_agents):
                index += 1
                if agent.pos[0] == None:
                    index -= 1
                else:
                    agent.update_agent_properties(polygon_id = index,polygon = poly_shapes[index],polygon_area = poly_shapes[index].area,TI_zone_assigned = TI_zone_per_agent[index])

            return copy_active_agents

    def assign_training_image_to_agent(self,copy_active_agents = None,training_image = None,agent_to_move = None,agent_new_TI = None):
        """ decide which training image agents and their polygon will get"""
        if copy_active_agents == None:
            #populate new agents with TI
            for agent in self.active_agents:
                if agent.TI_type == None:
                    agent.update_agent_TI(TI_type = agent.TI_zone_assigned,TI_no =randint(0,self.number_training_images_per_zone)) 
        
        # after testing out all positings --> moving agent to its best position, update TI and TI zone
        if agent_new_TI != None:
            for agent in self.active_agents:
                if agent.id == agent_new_TI.id:
                    agent.update_agent_TI(TI_type = training_image[0],TI_no = training_image[1])
        
        #test out differnt ti_zones and training images on agent that is to be moved.
        if copy_active_agents != None:

            for agent in copy_active_agents:
                if agent.TI_type == None:
                    agent.update_agent_TI(TI_type = agent.TI_zone_assigned,TI_no =randint(0,self.number_training_images_per_zone))
                if agent.id == agent_to_move.id:
                    agent.update_agent_TI(TI_type = training_image[0],TI_no =training_image[1] )
            return copy_active_agents
        
    def load_training_images(self):
        """ upload all training images that are to be used for reservoir model building 1 training image = entire reservoir model. """
        t_igrid = time.time()
        print('=======loading training images=========')

        # arrays for storage of TI: N_TI_zones, N_TI_per_zone, N_values_per_TI
        self.TI_values_permx = np.empty((self.number_training_image_zones,self.number_training_images_per_zone,(self.X*self.Y*7)))
        self.TI_values_permx[:] = np.nan
        self.TI_values_permy = np.empty((self.number_training_image_zones,self.number_training_images_per_zone,(self.X*self.Y*7)))
        self.TI_values_permy[:] = np.nan
        self.TI_values_permz = np.empty((self.number_training_image_zones,self.number_training_images_per_zone,(self.X*self.Y*7)))
        self.TI_values_permz[:] = np.nan
        self.TI_values_poro = np.empty((self.number_training_image_zones,self.number_training_images_per_zone,(self.X*self.Y*7)))
        self.TI_values_poro[:] = np.nan

        # set up path etc. to allow to extract grdcel file models into np arrays.
        geomodel_path = str(self.base_path / "training_images/TI_0/INCLUDE/GRID.grdecl")
        GRID = GeologyModel(filename = geomodel_path)

        # load each model into storage array
        for TI_zone in range(self.number_training_image_zones):
            for TI_no in range(self.number_training_images_per_zone):
                permx_path = str(self.base_path / 'training_images/TI_{} - Copy/INCLUDE/PERMX/M{}.GRDECL'.format(TI_zone,TI_no))
                permy_path = str(self.base_path / 'training_images/TI_{} - Copy/INCLUDE/PERMY/M{}.GRDECL'.format(TI_zone,TI_no))
                permz_path = str(self.base_path / 'training_images/TI_{} - Copy/INCLUDE/PERMZ/M{}.GRDECL'.format(TI_zone,TI_no))
                poro_path = str(self.base_path /'training_images/TI_{} - Copy/INCLUDE/PORO/M{}.GRDECL'.format(TI_zone,TI_no))

                permx = GRID.LoadCellData(varname="PERMX",filename=permx_path)
                permy = GRID.LoadCellData(varname="PERMY",filename=permy_path)
                permz = GRID.LoadCellData(varname="PERMZ",filename=permz_path)
                poro = GRID.LoadCellData(varname="PORO",filename=poro_path)

                self.TI_values_permx[TI_zone,TI_no,:] = permx
                self.TI_values_permy[TI_zone,TI_no,:] = permy
                self.TI_values_permz[TI_zone,TI_no,:] = permz
                self.TI_values_poro[TI_zone,TI_no,:] = poro
        
        print("Training images loaded! -  took {0:2.2f} seconds".format(time.time()-t_igrid))

    def get_possible_training_images(self,copy_active_agents,agent_to_move,get_current_TI = False):
        """ check what training image confiugratinos can be loaded into agent voronoi polygon"""
        TI = []

        # # training image of TI_zone that agent currently has. but plus one and minus one too
        TI.append([agent_to_move.TI_type,agent_to_move.TI_no])
        if  0 < agent_to_move.TI_no:
            TI.append([agent_to_move.TI_type,agent_to_move.TI_no-1])        
        if  agent_to_move.TI_no < self.number_training_images_per_zone-1:
            TI.append([agent_to_move.TI_type,agent_to_move.TI_no+1])
        
        # random training image from current TI zone assigend
        TI.append([agent_to_move.TI_zone_assigned,randint(0,self.number_training_images_per_zone)])

        # training image of current TI_zone assigend . but plus one and minus one too. might be similar to the one 
        TI.append([agent_to_move.TI_zone_assigned,agent_to_move.TI_no])
        if  0 < agent_to_move.TI_no:
            TI.append([agent_to_move.TI_zone_assigned,agent_to_move.TI_no-1])        
        if  agent_to_move.TI_no < self.number_training_images_per_zone-1:
            TI.append([agent_to_move.TI_zone_assigned,agent_to_move.TI_no+1])
        
        # most common TI in neighbourhood
        TI_zone_neighbours = []
        TI_zone_most_common = []
        TI_no_most_common = []
        buffer_value = 0
        while len(TI_zone_most_common) == 0:
            buffer_value += 0.5
            for agent in copy_active_agents:
                if agent.id == agent_to_move.id:
                    continue
                if agent_to_move.polygon.buffer(buffer_value).intersects(agent.polygon):
                    TI_zone_neighbours.append(agent.TI_type)
            TI_zone_most_common = Counter(np.array(TI_zone_neighbours)).most_common(1)

        TI_zone_most_common = TI_zone_most_common[0][0]
        TI_no_neighbours = []
        for agent in copy_active_agents:
            if agent.id == agent_to_move.id:
                continue
            if agent_to_move.polygon.buffer(buffer_value).intersects(agent.polygon):
                TI_no_neighbours.append(agent.TI_no)
        TI_no_most_common = Counter(np.array(TI_no_neighbours)).most_common(1)[0][0]
        TI.append([TI_zone_most_common,TI_no_most_common])

        # random TI
        TI.append([randint(0,self.number_training_image_zones),randint(0,self.number_training_images_per_zone)])

        # get curren_TI stats, ignore rest
        if get_current_TI == True:
            TI=[[agent_to_move.TI_type,agent_to_move.TI_no]]

        # get "goodness"/quality rank for each training image
        for index,training_image in enumerate(TI):
            # most common training image and correct TI_zone
            if training_image[0] == TI_zone_most_common and training_image[1] == TI_no_most_common and TI_zone_most_common == agent_to_move.TI_zone_assigned:
                TI[index].append(0)
            # correct TI_zone
            elif training_image[0] == agent_to_move.TI_zone_assigned:
                TI[index].append(1)
            # most common but incorrec TI_zone
            elif training_image[0] == TI_zone_most_common and training_image[1] == TI_no_most_common and TI_zone_most_common != agent_to_move.TI_zone_assigned:
                TI[index].append(2)
            # incorrect TI_zone
            elif training_image[0] != agent_to_move.TI_zone_assigned:
                TI[index].append(3)

        return TI       

    def generate_reservoir_model(self,copy_active_agents = None,agent_to_move = None,model_type = None, model_index = None, training_image = None):
        """ generate reservoir model from training images, patched together by voronoi polygons"""

        # assign each point in grid to TI that belongs to polygon that point sits in. either the final model or one of the tset models to figure out which one is best suited
        TI_assigned_to_grid_point_2D = []
        if model_type == "final_model_per_iteration":
            for agent in self.active_agents:
                TI_assigned_to_grid_point_2D.append([[agent.TI_type, agent.TI_no,self.index_by_id[id(point)]] for point in self.tree_TI_grid_points.query(agent.polygon) if point.intersects(agent.polygon)])

        elif model_type == "training_image_testing":
            for agent in copy_active_agents:
                if agent.pos[0] == None:
                    pass
                else:
                    TI_assigned_to_grid_point_2D.append([[agent.TI_type, agent.TI_no,self.index_by_id[id(point)]] for point in self.tree_TI_grid_points.query(agent.polygon) if point.intersects(agent.polygon)])

        # if points not within polygon due to vertices problem of shapely (point lies on vertices --> point undetected)
        TI_assigned_to_grid_point_2D_temp = [item for sublist in TI_assigned_to_grid_point_2D for item in sublist]
        detected_points = [row[2] for row in TI_assigned_to_grid_point_2D_temp]

        missed_points =list(set(self.list_TI_grid_points) - set(detected_points))

        while len(missed_points)!=0:
            polygon_centroids = []
            if copy_active_agents == None:
                for agent in self.active_agents:
                    polygon_centroids.append(agent.polygon.centroid)
                polygon_centroids = MultiPoint(polygon_centroids)
            else:
                for agent in copy_active_agents:
                    if agent.pos[0] != None:
                        polygon_centroids.append(agent.polygon.centroid)
                polygon_centroids = MultiPoint(polygon_centroids)
           
            for points in missed_points:
                point = self.TI_grid_points[points]
                nearest_polygon_centroid = nearest_points(point,polygon_centroids)
                nearest_polygon_centroid = list(nearest_polygon_centroid[1].coords)[0]

                if copy_active_agents == None:
                    for index, agent in enumerate(self.active_agents):
                        agent_polygon_centroid = list(agent.polygon.centroid.coords)[0]
                        if agent_polygon_centroid == nearest_polygon_centroid:
                            point_stats = [agent.TI_type,agent.TI_no,self.index_by_id[id(point)]]
                            TI_assigned_to_grid_point_2D[index].append(point_stats)
                             
                else:
                    index = -1
                    for i, agent in enumerate(copy_active_agents):
                        index += 1
                        if agent.pos[0] == None:
                            index -= 1
                        else:
                            agent_polygon_centroid = list(agent.polygon.centroid.coords)[0]
                            if agent_polygon_centroid == nearest_polygon_centroid:
                                point_stats = [agent.TI_type,agent.TI_no,self.index_by_id[id(point)]]
                                TI_assigned_to_grid_point_2D[index].append(point_stats)
            
            TI_assigned_to_grid_point_2D_temp = [item for sublist in TI_assigned_to_grid_point_2D for item in sublist]
            detected_points = [row[2] for row in TI_assigned_to_grid_point_2D_temp]

            missed_points =list(set(self.list_TI_grid_points) - set(detected_points))

        # unpack list in list
        TI_assigned_to_grid_point_2D = [item for sublist in TI_assigned_to_grid_point_2D for item in sublist]

        # assign index to list. need this later. position 4
        for i in range(len(TI_assigned_to_grid_point_2D)):
            TI_assigned_to_grid_point_2D[i].append(i)

        # some points are assigned to multiple polygons. dropping dublicate point ids
        TI_assigned_to_grid_point_2D_no_dublicates = []
        TI_assigned_to_grid_point_2D_no_dublicates_set  = set()
        for point in TI_assigned_to_grid_point_2D:
            point_id = tuple(point)[2]
            if point_id not in TI_assigned_to_grid_point_2D_no_dublicates_set:
                TI_assigned_to_grid_point_2D_no_dublicates.append(point)
                TI_assigned_to_grid_point_2D_no_dublicates_set.add(point_id)

        #sort list by point_id
        TI_assigned_to_grid_point_2D_no_dublicates.sort(key=lambda x: x[2])

        #if still not all points found, somethings is wrong. current fix: append last point to list again
        while len(TI_assigned_to_grid_point_2D_no_dublicates)!= len(self.env.flatten()):
            print("problem with polygon to point assignemnt. temporary fix")
            TI_assigned_to_grid_point_2D_no_dublicates.append(TI_assigned_to_grid_point_2D_no_dublicates[-1])
            print(len(TI_assigned_to_grid_point_2D_no_dublicates))

        # extent to full 3D reservoir model. so far only have 1 layer
        extend_2D_to_3D = int((self.X*self.Y*7)/(self.X*self.Y*self.Z))
        TI_assigned_to_grid_point = []
        for i in range(extend_2D_to_3D):
            TI_assigned_to_grid_point.extend(TI_assigned_to_grid_point_2D_no_dublicates)


        # assign each point in grid the correct TI image TI zone poro/perm values
        # patch things together
        patch_permx  = []
        patch_permy  = []
        patch_permz  = []
        patch_poro  = []

        for i in range(len(TI_assigned_to_grid_point)):

            TI_zone = int(TI_assigned_to_grid_point[i][0])
            TI_no = int(TI_assigned_to_grid_point[i][1])

            permx = self.TI_values_permx[TI_zone,TI_no,i]
            permy = self.TI_values_permy[TI_zone,TI_no,i]              
            permz = self.TI_values_permz[TI_zone,TI_no,i]              
            poro = self.TI_values_poro[TI_zone,TI_no,i]

            patch_permx.append(permx)
            patch_permy.append(permy)
            patch_permz.append(permz)
            patch_poro.append(poro)

        # save generated patched reservoir model properties
        self.save_reservoir_model_properties(dataset = patch_permx,prop = "PERMX",model_type = model_type,index = model_index, training_image = training_image)
        self.save_reservoir_model_properties(dataset = patch_permy,prop = "PERMY",model_type = model_type,index = model_index, training_image = training_image)
        self.save_reservoir_model_properties(dataset = patch_permz,prop = "PERMZ",model_type = model_type,index = model_index, training_image = training_image)
        self.save_reservoir_model_properties(dataset = patch_poro,prop = "PORO",model_type = model_type,index = model_index, training_image = training_image)

        # generate and save DATA file
        self.built_Data_files(index = model_index, training_image = training_image,model_type= model_type)

    def save_reservoir_model_properties(self,dataset,index = None,training_image = None,prop = None,model_type = None):
        """ save properties of patched reservoir model so that they can be run with FD"""

        if self.n_processes == None:
            if model_type == "training_image_testing":
                file_path = self.base_path / 'training_image_testing/INCLUDE/{}/M{}.GRDECL'.format(prop,index)
        else: 
            if model_type == "training_image_testing":
                file_path = self.base_path / "training_image_testing/INCLUDE/{}/M_{}_{}_{}.GRDECL".format(prop,index,training_image[0],training_image[1]) # saved in test folder to decide which TI to go forward with
        if model_type == "final_model_per_iteration":
            file_path = self.base_path / 'results/INCLUDE/{}/M{}.GRDECL'.format(prop,self.istep)

        file_beginning = "FILEUNIT\nMETRIC /\n\n{}\n".format(prop)
        dataset[-1] = "{} /".format(dataset[-1])
        with open(file_path,"w+") as f:
            f.write(file_beginning)
            newline_ticker = 0
            for item in dataset:
                newline_ticker += 1
                if newline_ticker == 50:
                    f.write("\n")
                    newline_ticker = 0
                f.write("{} ".format(item))
            f.close()

    def built_Data_files(self,index = None,training_image = None,model_type = None):
        """ built Data files to run FD and full flow simulations on """
        
        if self.n_processes == None:
            if model_type == "training_image_testing":
                file_path = self.base_path / 'training_image_testing/DATA/M{}.DATA'.format(index)
                data_file = "RUNSPEC\n\nTITLE\nModel_{}\n\nDIMENS\n--NX NY NZ\n200 100 7 /\n\n--Phases\nOIL\nWATER\n\n--DUALPORO\n--NODPPM\n\n--Units\nMETRIC\n\n--Number of Saturation Tables\nTABDIMS\n1 /\n\n--Maximum number of Wells\nWELLDIMS\n10 100 5 10 /\n\n--First Oil\nSTART\n1 OCT 2017 /\n\n--Memory Allocation\nNSTACK\n100 /\n\n--How many warnings allowed, but terminate after first error\nMESSAGES\n11*5000 1 /\n\n--Unified Output Files\nUNIFOUT\n\n--======================================================================\n\nGRID\n--Include corner point geometry model\nINCLUDE\n'..\INCLUDE\GRID.GRDECL'\n/\n\nACTNUM\n140000*1 /\n\n--Porosity\nINCLUDE\n'..\INCLUDE\PORO\M{}.GRDECL'\n/\n\n--Permeability\nINCLUDE\n'..\INCLUDE\PERMX\M{}.GRDECL'\n/\nINCLUDE\n'..\INCLUDE\PERMY\M{}.GRDECL'\n/\nINCLUDE\n'..\INCLUDE\PERMZ\M{}.GRDECL'\n/\n\n--Net to Gross\nNTG\n140000*1\n/\n\n--Output .INIT file to allow viewing of grid data in post proessor\nINIT\n\n--======================================================================\n\nPROPS\n\nINCLUDE\n'..\INCLUDE\DP_pvt.inc' /\n\nINCLUDE\n'..\INCLUDE\ROCK_RELPERMS.INC' /\n\n--======================================================================\n\nREGIONS\n\nEQLNUM\n140000*1\n/\nSATNUM\n140000*1\n/\nPVTNUM\n140000*1\n/\n\n--======================================================================\n\nSOLUTION\n\nINCLUDE\n'..\INCLUDE\SOLUTION.INC' /\n\n--======================================================================\n\nSUMMARY\n\nINCLUDE\n'..\INCLUDE\SUMMARY.INC' /\n\n--======================================================================\n\nSCHEDULE\n\nINCLUDE\n'..\INCLUDE\SCHEDULE.INC' /\n\nEND".format(index,index,index,index,index)  

        else:
            if model_type == "training_image_testing":
                file_path = self.base_path / "training_image_testing/DATA/M_{}_{}_{}.DATA".format(index,training_image[0],training_image[1])# saved in test folder to decide which TI to go forward with
                data_file = "RUNSPEC\n\nTITLE\nModel_{}\n\nDIMENS\n--NX NY NZ\n200 100 7 /\n\n--Phases\nOIL\nWATER\n\n--DUALPORO\n--NODPPM\n\n--Units\nMETRIC\n\n--Number of Saturation Tables\nTABDIMS\n1 /\n\n--Maximum number of Wells\nWELLDIMS\n10 100 5 10 /\n\n--First Oil\nSTART\n1 OCT 2017 /\n\n--Memory Allocation\nNSTACK\n100 /\n\n--How many warnings allowed, but terminate after first error\nMESSAGES\n11*5000 1 /\n\n--Unified Output Files\nUNIFOUT\n\n--======================================================================\n\nGRID\n--Include corner point geometry model\nINCLUDE\n'..\INCLUDE\GRID.GRDECL'\n/\n\nACTNUM\n140000*1 /\n\n--Porosity\nINCLUDE\n'..\INCLUDE\PORO\M_{}_{}_{}.GRDECL'\n/\n\n--Permeability\nINCLUDE\n'..\INCLUDE\PERMX\M_{}_{}_{}.GRDECL'\n/\nINCLUDE\n'..\INCLUDE\PERMY\M_{}_{}_{}.GRDECL'\n/\nINCLUDE\n'..\INCLUDE\PERMZ\M_{}_{}_{}.GRDECL'\n/\n\n--Net to Gross\nNTG\n140000*1\n/\n\n--Output .INIT file to allow viewing of grid data in post proessor\nINIT\n\n--======================================================================\n\nPROPS\n\nINCLUDE\n'..\INCLUDE\DP_pvt.inc' /\n\nINCLUDE\n'..\INCLUDE\ROCK_RELPERMS.INC' /\n\n--======================================================================\n\nREGIONS\n\nEQLNUM\n140000*1\n/\nSATNUM\n140000*1\n/\nPVTNUM\n140000*1\n/\n\n--======================================================================\n\nSOLUTION\n\nINCLUDE\n'..\INCLUDE\SOLUTION.INC' /\n\n--======================================================================\n\nSUMMARY\n\nINCLUDE\n'..\INCLUDE\SUMMARY.INC' /\n\n--======================================================================\n\nSCHEDULE\n\nINCLUDE\n'..\INCLUDE\SCHEDULE.INC' /\n\nEND".format(self.istep,index,training_image[0],training_image[1],index,training_image[0],training_image[1],index,training_image[0],training_image[1],index,training_image[0],training_image[1])  

        if model_type == "final_model_per_iteration":
            file_path = self.base_path / 'results/DATA/M{}.DATA'.format(self.istep)
            data_file = "RUNSPEC\n\nTITLE\nModel_{}\n\nDIMENS\n--NX NY NZ\n200 100 7 /\n\n--Phases\nOIL\nWATER\n\n--DUALPORO\n--NODPPM\n\n--Units\nMETRIC\n\n--Number of Saturation Tables\nTABDIMS\n1 /\n\n--Maximum number of Wells\nWELLDIMS\n10 100 5 10 /\n\n--First Oil\nSTART\n1 OCT 2017 /\n\n--Memory Allocation\nNSTACK\n100 /\n\n--How many warnings allowed, but terminate after first error\nMESSAGES\n11*5000 1 /\n\n--Unified Output Files\nUNIFOUT\n\n--======================================================================\n\nGRID\n--Include corner point geometry model\nINCLUDE\n'..\INCLUDE\GRID.GRDECL'\n/\n\nACTNUM\n140000*1 /\n\n--Porosity\nINCLUDE\n'..\INCLUDE\PORO\M{}.GRDECL'\n/\n\n--Permeability\nINCLUDE\n'..\INCLUDE\PERMX\M{}.GRDECL'\n/\nINCLUDE\n'..\INCLUDE\PERMY\M{}.GRDECL'\n/\nINCLUDE\n'..\INCLUDE\PERMZ\M{}.GRDECL'\n/\n\n--Net to Gross\nNTG\n140000*1\n/\n\n--Output .INIT file to allow viewing of grid data in post proessor\nINIT\n\n--======================================================================\n\nPROPS\n\nINCLUDE\n'..\INCLUDE\DP_pvt.inc' /\n\nINCLUDE\n'..\INCLUDE\ROCK_RELPERMS.INC' /\n\n--======================================================================\n\nREGIONS\n\nEQLNUM\n140000*1\n/\nSATNUM\n140000*1\n/\nPVTNUM\n140000*1\n/\n\n--======================================================================\n\nSOLUTION\n\nINCLUDE\n'..\INCLUDE\SOLUTION.INC' /\n\n--======================================================================\n\nSUMMARY\n\nINCLUDE\n'..\INCLUDE\SUMMARY.INC' /\n\n--======================================================================\n\nSCHEDULE\n\nINCLUDE\n'..\INCLUDE\SCHEDULE.INC' /\n\nEND".format(self.istep,self.istep,self.istep,self.istep,self.istep)  

        file = open(file_path, "w+")
        # write petrelfilepath and licence part into file and seed
        file.write(data_file)

        # close file
        file.close()
    
    def run_FD(self,model_type = None,index = None,training_image = None,number_test_runs = None,models_to_run = None):
        """ run Flow Diagnostics on selected reservoir model"""

        if self.n_processes == None:
            if model_type == "training_image_testing":
                FD_data = self.matlab_runner.FD_ABM_testrun_multimodel(models_to_run)
                # split into Ev tD F Phi and LC and tof column
                FD_data = np.array(FD_data._data).reshape((10,len(FD_data)//10))
                FD_performance = pd.DataFrame()
                
                FD_performance["EV"] = FD_data[0]
                FD_performance["tD"] = FD_data[1]
                FD_performance["F"] = FD_data[2]
                FD_performance["Phi"] = FD_data[3]
                FD_performance["LC"] = FD_data[4]
                FD_performance["tof_for"] = FD_data[5]
                FD_performance["tof_back"] = FD_data[6]
                FD_performance["tof_combi"] = FD_data[7]
                FD_performance["prod_part"] = FD_data[8]
                FD_performance["inj_part"] = FD_data[9]
                model_id_all = []
                for model_id in models_to_run:
                    for n_cells in range(self.X*self.Y*7):
                        model_id_all.append(model_id)
                FD_performance["model_id"] = model_id_all


        elif model_type == "training_image_testing":
            model_id = "M_{}_{}_{}".format(index,training_image[0],training_image[1])
            # run FD and output dictionary
            FD_data = self.matlab_runner.FD_ABM_testrun(model_id)

        if model_type == "final_model_per_iteration":
            model_id = "M{}".format(self.istep)
            # run FD and output dictionary
            FD_data = self.matlab_runner.FD_ABM(model_id)
            # split into Ev tD F Phi and LC and tof column
            FD_data = np.array(FD_data._data).reshape((10,len(FD_data)//10))
            FD_performance = pd.DataFrame()
            
            FD_performance["EV"] = FD_data[0]
            FD_performance["tD"] = FD_data[1]
            FD_performance["F"] = FD_data[2]
            FD_performance["Phi"] = FD_data[3]
            FD_performance["LC"] = FD_data[4]
            FD_performance["tof_for"] = FD_data[5]
            FD_performance["tof_back"] = FD_data[6]
            FD_performance["tof_combi"] = FD_data[7]
            FD_performance["prod_part"] = FD_data[8]
            FD_performance["inj_part"] = FD_data[9]
  

        FD_performance = FD_performance.astype("float32")
        return(FD_performance)

    def calculate_misfit(self,FD_performance):
        """ caclulate misfit between objective set and model flow diagnostic response"""

        # interpolate F-Phi curve from imput points with spline
        tck = interpolate.splrep(self.Phi_points_target,self.F_points_target, s = 0)
        Phi_interpolated = np.linspace(0,1,num = len(FD_performance["Phi"]),endpoint = True)
        F_interpolated = interpolate.splev(Phi_interpolated,tck,der = 0) # here can easily get first and second order derr.

        # calculate first order derivate of interpolated F-Phi curve and modelled F-Phi curve
        F_interpolated_first_derr = np.gradient(F_interpolated)
        F_first_derr = np.gradient(FD_performance["F"])
        F_interpolated_second_derr = np.gradient(F_interpolated_first_derr)
        F_second_derr = np.gradient(F_first_derr)

        # calculate LC for interpolatd F-Phi curve and modelled F-Phi curve
        LC_interpolated = self.compute_LC(F_interpolated,Phi_interpolated)

        LC = self.compute_LC(FD_performance["F"],FD_performance["Phi"])
        # calculate rmse for each curve and LC
        rmse_0 = mean_squared_error(F_interpolated,FD_performance["F"],squared=False)
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
    
    def make_directories(self):
        """ iniitate directory strucutre to save data"""

        output_path_test = self.base_path / "training_image_testing"# / self.output_folder
        output_path_results = self.base_path / "results"# / self.output_folder

        data_path_test = output_path_test / "DATA"
        include_path_test = output_path_test / "INCLUDE"
        permx_path_test = include_path_test / "PERMX"
        permy_path_test = include_path_test / "PERMY"
        permz_path_test = include_path_test / "PERMZ"
        poro_path_test = include_path_test / "PORO"

        data_path_results = output_path_results / "DATA"
        include_path_results = output_path_results / "INCLUDE"
        permx_path_results = include_path_results / "PERMX"
        permy_path_results = include_path_results / "PERMY"
        permz_path_results = include_path_results / "PERMZ"
        poro_path_results = include_path_results / "PORO"

        if not os.path.exists(data_path_test):
            # make folders and subfolders
            os.makedirs(data_path_test)
            os.makedirs(include_path_test)
            os.makedirs(permx_path_test)
            os.makedirs(permy_path_test)
            os.makedirs(permz_path_test)
            os.makedirs(poro_path_test)
        if not os.path.exists(data_path_results):
            # make folders and subfolders
            os.makedirs(data_path_results)
            os.makedirs(include_path_results)
            os.makedirs(permx_path_results)
            os.makedirs(permy_path_results)
            os.makedirs(permz_path_results)
            os.makedirs(poro_path_results)