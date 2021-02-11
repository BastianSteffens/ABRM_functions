from numpy.random import randint
from numpy.random import choice
import numpy as np
import pandas as pd
import random as rn
import time
from datetime import datetime
import copy
import numpy as np
from shapely.strtree import STRtree
from shapely.geometry import Point
from shapely.geometry import Polygon
from geovoronoi import voronoi_regions_from_coords
from geovoronoi import polygon_lines_from_voronoi
from collections import Counter
from GRDECL_file_reader.GRDECL2VTK import*
import pathlib

from Agent import Agent
from Grid import Grid

preference_L = [[[1., 1., 1.],[1., 1., 1.], [1., 1., 1.]],
                [[1., 1., 1.], [1., 1., 1.], [1., 1., 1.]],
                [[1., 1., 1.], [1., 1., 1.], [1., 1., 1.]]]
preference_mat = np.array([np.array(np.array([Lii for Lii in Li])) for Li in preference_L])

stochastic_L = [[[0.075, 0.0375, 0.], [0.075, 0.0375, 0.], [0.075, 0.0375, 0.]],
                [[0.075, 0.0375, 0.], [0.1, 0., 0.], [0.075, 0.0375, 0.]],
                [[0.075, 0.0375, 0.], [0.075, 0.0375, 0.], [0.075, 0.0375, 0.]]]
stochastic_mat = np.array([np.array(np.array([Lii for Lii in Li])) for Li in stochastic_L])

import pickle
def to_pickle(obj, file):
    with open(file, 'wb') as handle:
        pickle.dump(obj, handle, protocol=pickle.HIGHEST_PROTOCOL)


class Model():
    """Manage an ensemble of the agents during the simmulation"""

    def __init__(self, 
                env, 
                number_of_turns=2000,number_of_starting_agents = 10,new_agent_every_n_turns = 1,new_agents_per_n_turns = 1,max_number_agents = 10,
                ratio_of_tracked_agents = 1.,number_training_image_zones = 2, number_training_images_per_zone = 20,
                move_preference_matrix=preference_mat, 
                stochastic_move_matrix=stochastic_mat, 
                ):

        self.base_path = pathlib.Path(__file__).parent
 

        self.env = env
        (self.X, self.Y, self.Z) = self.env.shape
        print('Shape of your environment: ({}, {}, {})'.format(self.X, self.Y, self.Z))

        self.number_training_image_zones = number_training_image_zones
        self.number_training_images_per_zone = number_training_images_per_zone
        self.training_images = np.linspace((1,1),(number_training_images_per_zone,number_training_images_per_zone),number_training_images_per_zone)
        self.load_training_images()

        self.initiate_grid()
        self.init_TI_zone_grid()


        self.istep = 0
        self.current_id = -1

        self.number_of_turns = number_of_turns
        self.new_agent_every_n_turns = new_agent_every_n_turns
        self.new_agents_per_n_turns = new_agents_per_n_turns
        self.number_of_starting_agents = number_of_starting_agents
        self.max_number_agents = max_number_agents
        self.when_new_agent = 0


        self.ratio_of_tracked_agents = ratio_of_tracked_agents

        self.move_preference_matrix = move_preference_matrix
        self.stochastic_move_matrix = stochastic_move_matrix

        self.active_agents = []
        self.dead_agents = []
        
        self.resultsDF = pd.DataFrame(columns=['Start_x', 'Start_y', 'Start_z','End_x',
                                            'End_y','End_z','End_turn']).astype(dtype={'Start_x': np.int16, 'Start_y': np.int16, 
                                                                                        'Start_z': np.int16,'End_x': np.int16,
                                                                                        'End_y': np.int16,'End_z': np.int16,
                                                                                        'End_turn': np.int16})

        self.stop_simulation = False

    def initiate_grid(self):
        t_igrid = time.time()
        print('=======Initiating grid=========')
        self.grid = Grid(self.X, self.Y, self.Z)
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
        copy_active_agents = copy.copy(self.active_agents)
        for iagent, agent in enumerate (copy_active_agents):
            print("moving agent {}".format(agent.id), end="\r")
            new_pos = self.get_agent_movement(agent,copy_active_agents)
            if new_pos ==(None,None,None):
                print("Killed agent {}        ".format(agent.id))
                self.kill_agent(agent)

            self.grid.move_agent(agent, new_pos)
            self.generate_voronoi_regions()
            self.assign_training_image_to_agent()
            # self.generate_reservoir_model()
            # self.run_FD()
            # self.calculate_misfit()
   

    def get_agent_movement(self, agent,copy_active_agents):
        free_neighborhood_coord = self.grid.get_empty_neighborhood(agent.pos,len(self.active_agents))
        
        if rn.random() <= 0.9:
            print("penis will do this later")
        #     #move to max preference position multiply by env value
        #     max_value = -1
        #     for coord in free_neighborhood_coord:
        #         dcoord = (coord[0] - agent.pos[0] + 1, coord[1] - agent.pos[1] + 1, coord[2] - agent.pos[2] + 1)
        #         move_value = self.get_location_movement_value(self.move_preference_matrix[dcoord[0], 
        #                                                     dcoord[1], dcoord[2]], self.classified_env[coord[0], coord[1], coord[2]])
        #         if move_value > max_value:
        #             max_value = move_value
        #             new_pos = coord
        #         elif move_value == max_value:
        #             new_pos = rn.choice([coord, new_pos])
        #         if move_value < 0:
        #             print('Error move value negtive {}'.format(move_value))
        #             self.stop_simulation = True
        else:
            print("not random move. not ready now.")
            #run through all possible szenarios this is the bit that should be parallelized if pool = None else self.evalute_position_iterator position. self.evaluate_position_parallel
            # for index,coord in enumerate(free_neighborhood_coord):
            #     self.generate_voronoi_regions()
            #     possible_training_images = self.get_possible_training_images()
            #     for training_image in possible_training_images:
            #         self.assign_training_image_to_agent()
            #         self.generate_reservoir_model()
            #         self.run_FD()
            #         self.calculate_misfit()                
                
        #     #stochasticly move according to stochastic_move_matrix
        #     stochastic_prob = []
        #     for coord in free_neighborhood_coord:
        #         dcoord = (coord[0] - agent.pos[0] + 1, coord[1] - agent.pos[1] + 1, coord[2] - agent.pos[2] + 1)
        #         stochastic_prob.append(self.stochastic_move_matrix[dcoord[0], dcoord[1], dcoord[2]])
        #     stochastic_prob = [elm*(1./sum(stochastic_prob)) for elm in stochastic_prob]    #sum probabilities to 1.
        #     i_new_pos = choice(np.arange(len(free_neighborhood_coord)), 1, p=stochastic_prob) #choice weighted by probability distribution: (list_of_candidates, 
        #                                                                                 #number_of_items_to_pick, p=probability_distribution)

        # random movement for now
        i_new_pos = randint(0,len(free_neighborhood_coord))
        new_pos = free_neighborhood_coord[int(i_new_pos)]
        return(new_pos)

    def get_location_movement_value(self, movement_preference_value, env_value):
        """ Takes into account movement preference and environment value """
        return movement_preference_value*(env_value)**2

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
        # track agents
        self.track_agents()
        self.grid.layer = self.Z-1 # not sure what this does.
        self.when_new_agent +=1

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
        track_file = np.empty((self.current_id+1, self.number_of_turns+1  , 4))
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
            Agent_postions.append([agent.id, agent.start_x, agent.start_y, agent.start_z, agent.pos[0], agent.pos[1], agent.pos[2]])
        for agent in self.dead_agents:
            Agent_postions.append([agent.id, agent.start_x, agent.start_y, agent.start_z, agent.pos[0], agent.pos[1], agent.pos[2]])
        self.resultsDF = pd.DataFrame(Agent_postions, columns=['AgentID', 'Start_x', 'Start_y', 'Start_z','End_x', 'End_y','End_z'])
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
        # self.generate_reservoir_model()
        # self.run_FD()
        # self.calculate_misfit()
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

    def generate_voronoi_regions(self):
        """ split up model into voronoi regoins, with each agent marking the center of a voronoi region"""
        voronoi_x = []
        voronoi_y = []
        voronoi_z = []
        
        # get all active agent locations 
        for agent in self.active_agents:
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
        for i,agent in enumerate(self.active_agents):
            agent.update_agent_properties(polygon_id = i,polygon = poly_shapes[i],polygon_area = poly_shapes[i].area,TI_zone_assigned = TI_zone_per_agent[i])

    def assign_training_image_to_agent(self):
        """ decide which training image agents and their polygon will get"""

        #populate init grid
        if self.istep == 0:
            for agent in self.active_agents:
                agent.update_agent_TI(TI_type = agent.TI_zone_assigned,TI_no =randint(1,self.number_training_images_per_zone+1) )

        #for now the same but later more advanced. this is jsut to get a simulation running
        else:
            for agent in self.active_agents:
                agent.update_agent_TI(TI_type = agent.TI_zone_assigned,TI_no =randint(1,self.number_training_images_per_zone+1) )

    def load_training_images(self):
        """ upload all training images that are to be used for reservoir model building 1 training image = entire resevoir model. """
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
                permx_path = str(self.base_path / 'training_images/TI_{}/INCLUDE/PERMX/M{}.GRDECL'.format(TI_zone,TI_no))
                permy_path = str(self.base_path / 'training_images/TI_{}/INCLUDE/PERMY/M{}.GRDECL'.format(TI_zone,TI_no))
                permz_path = str(self.base_path / 'training_images/TI_{}/INCLUDE/PERMZ/M{}.GRDECL'.format(TI_zone,TI_no))
                poro_path = str(self.base_path /'training_images/TI_{}/INCLUDE/PORO/M{}.GRDECL'.format(TI_zone,TI_no))

                permx = GRID.LoadCellData(varname="PERMX",filename=permx_path)
                permy = GRID.LoadCellData(varname="PERMY",filename=permy_path)
                permz = GRID.LoadCellData(varname="PERMZ",filename=permz_path)
                poro = GRID.LoadCellData(varname="PORO",filename=poro_path)

                self.TI_values_permx[TI_zone,TI_no,:] = permx
                self.TI_values_permy[TI_zone,TI_no,:] = permy
                self.TI_values_permz[TI_zone,TI_no,:] = permz
                self.TI_values_poro[TI_zone,TI_no,:] = poro
        
        print("Training images loaded! -  took {0:2.2f} seconds".format(time.time()-t_igrid))



    def get_possible_training_images(self):
        """ check what training image confiugratinos can be loaded into agent voronoi polygon"""

    def generate_reservoir_model(self):
        """ generate reservoir model from training images, patched together by voronoi polygons"""



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
        data_file_path = self.setup["base_path"] / "../FD_Models/DATA/M_FD_{}.DATA".format(particle_no)

        for j in range(n_voronoi_zones):
            temp_model_path_permx = self.setup["base_path"] / '../FD_Models/INCLUDE/Voronoi/Patch_{}/PERMX/M{}.GRDECL'.format(j,particle_no)
            temp_model_path_permy = self.setup["base_path"] / '../FD_Models/INCLUDE/Voronoi/Patch_{}/PERMY/M{}.GRDECL'.format(j,particle_no)
            temp_model_path_permz = self.setup["base_path"] / '../FD_Models/INCLUDE/Voronoi/Patch_{}/PERMZ/M{}.GRDECL'.format(j,particle_no)
            temp_model_path_poro = self.setup["base_path"] / '../FD_Models/INCLUDE/Voronoi/Patch_{}/PORO/M{}.GRDECL'.format(j,particle_no)
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
        permx_file_path = self.setup["base_path"] / "../FD_Models/INCLUDE/PERMX/M{}.GRDECL".format(particle_no)
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
        permy_file_path = self.setup["base_path"] / "../FD_Models/INCLUDE/PERMY/M{}.GRDECL".format(particle_no)
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
        permz_file_path = self.setup["base_path"] / "../FD_Models/INCLUDE/PERMZ/M{}.GRDECL".format(particle_no)
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
        poro_file_path = self.setup["base_path"] / "../FD_Models/INCLUDE/PORO/M{}.GRDECL".format(particle_no)
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