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
                number_of_turns=2000,number_of_starting_agents = 10,new_agent_every_n_turns = 1,max_number_agents = 10,
                ratio_of_tracked_agents = 1.,number_training_image_zones = 2, number_training_images_per_zone = 20,
                move_preference_matrix=preference_mat, 
                stochastic_move_matrix=stochastic_mat, 
                ):
        self.env = env
        (self.X, self.Y, self.Z) = self.env.shape
        print('Shape of your environment: ({}, {}, {})'.format(self.X, self.Y, self.Z))

        self.number_training_image_zones = number_training_image_zones
        self.number_training_images_per_zone = number_training_images_per_zone
        self.training_images = np.linspace((1,1),(number_training_images_per_zone,number_training_images_per_zone),number_training_images_per_zone)

        self.initiate_grid()

        self.istep = 0
        self.current_id = 0

        self.number_of_turns = number_of_turns
        self.new_agent_every_n_turns = new_agent_every_n_turns
        self.number_of_starting_agents = number_of_starting_agents
        # self.new_agents_per_turn = new_agents_per_turn
        # self.bottom_agent_ratio_birth = bottom_agent_ratio_birth
        self.ratio_of_tracked_agents = ratio_of_tracked_agents

        self.move_preference_matrix = move_preference_matrix
        self.stochastic_move_matrix = stochastic_move_matrix

        self.active_agents = set()
        self.dead_agents = set()
        
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
        # if self.current_id == 0:
        #     agentID = 0
        # else:
            # agentID = self.next_id()
        agentID = self.next_id()
        print("agent_ID")
        print(agentID)

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
        self.active_agents.add(agent)
        # print('Generating agent {}/{}'.format(inew_agent+1, self.new_agents_per_turn), end="\r")

    def kill_agent(self, agent, next_turn=False):
        if next_turn:
            agent.kill(self.istep+1)
        else:
            agent.kill(self.istep)
        self.active_agents.discard(agent)
        self.dead_agents.add(agent)
    
    def move_agents(self):
        copy_active_agents = copy.copy(self.active_agents)
        # n_agents = len(copy_active_agents)

        for iagent, agent in enumerate (copy_active_agents):
            new_pos = self.get_agent_movement(agent)
            if type(new_pos) == int:
                if new_pos == 5:
                    #kill agent bacause blocked (all neighbors occupied) for the last 5 turns
                    self.kill_agent(agent)
                else:
                    if agent.isTracked:
                        agent.track.append([agent.pos[0], agent.pos[1], agent.pos[2]])
            else:
                self.grid.move_agent(agent, new_pos)
                self.generate_voronoi_regions()
                self.assign_training_image_to_agent()
   

    def get_agent_movement(self, agent):
        free_neighborhood_coord = self.grid.get_empty_neighborhood(agent.pos)
        # if not free_neighborhood_coord:
        #     agent.blocked_turns += 1
        #     return agent.blocked_turns
        
        # if rn.random() <= 0.9:
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
        # else:
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

        if self.istep > 0:
            if self.istep % self.new_agent_every_n_turns == 0:
                self.generate_new_agent()
                print("generating new agent")

        self.move_agents()
        print('Moving agents done                  ')
        print('Generating agents done              ')
        self.grid.layer = self.Z-1 # not sure what this does.
        self.istep += 1

    def run(self):

        #init model
        self.init_reservoir_model()

        for _ in range(self.number_of_turns):
            self.step()
            if self.stop_simulation:
                print('Simulation stoped because environment is full')
                break

    def get_all_tracks(self):
        """ Extract the tracking path of all the agents during the simulation in numpy.array of shape (number of agent, number of turns, 3 coordinates) """
        track_file = np.empty((self.number_of_turns*self.new_agents_per_turn, self.number_of_turns+self.number_extra_turns, 3))
        track_file[:] = np.nan
        for agent in self.active_agents:
            if agent.isTracked:
                track_file[agent.id, agent.start_iteration:, :] = agent.track
        for agent in self.dead_agents:
            if agent.isTracked:
                track_file[agent.id, agent.start_iteration:agent.dead_iteration, :] = agent.track
                if agent.dead_iteration != self.number_of_turns+self.number_extra_turns:
                    track_file[agent.id, agent.dead_iteration:, :] = [agent.track[-1]]*(self.number_of_turns+self.number_extra_turns-agent.dead_iteration)
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

        # #define grid and  position initianinon points of n polygons
        grid = Polygon([(0, 0), (0, self.Y), (self.X, self.Y), (self.X, 0)])

        # generate 2D mesh
        x = np.arange(0,self.X+1,1,)
        y = np.arange(0,self.Y+1,1)
        x_grid, y_grid = np.meshgrid(x,y)

        #get cell centers of mesh
        x_cell_center = x_grid[:-1,:-1]+0.5
        y_cell_center = y_grid[:-1,:-1]+0.5

        cell_center_which_polygon = np.zeros(len(x_cell_center.flatten()))

        # array to assign polygon id [ last column] to cell id [first 2 columns]
        all_cell_center = np.column_stack((x_cell_center.flatten(),y_cell_center.flatten(),cell_center_which_polygon))

        # get voronoi regions
        poly_shapes, pts, poly_to_pt_assignments = voronoi_regions_from_coords(voronoi_points, grid,farpoints_max_extend_factor = 30)
        
        # assign polygons to TI_zones that they hold the majority of
        grid_points = [Point(np.array([all_cell_center[i,0],all_cell_center[i,1]])) for i in range(len(all_cell_center))]

        tree_grid_points = STRtree(grid_points)
        index_by_id = dict((id(points), i) for i, points in enumerate(grid_points))
        points_in_polygon = []

        for voronoi_polygon_id in range(len(poly_shapes)):
            points_in_polygon.append([[point.x, point.y,index_by_id[id(point)]] for point in tree_grid_points.query(poly_shapes[voronoi_polygon_id]) if point.intersects(poly_shapes[voronoi_polygon_id])])
        # if points not within polygon due to vertices problem of shapely (point lies on vertices --> point undetected)
        points_in_polygon_temp = [item for sublist in points_in_polygon for item in sublist]
        detected_points = [row[2] for row in points_in_polygon_temp]
        all_points = []
        for i in range(len(grid_points)):
            all_points.append(index_by_id[id(grid_points[i])])
        missed_points =list(set(all_points) - set(detected_points))
        for points in missed_points:
            point = grid_points[points]
            point_shifted = Point(point.x+1e-9,point.y+1e-9)
            for voronoi_polygon_id in range(len(poly_shapes)):
                if point_shifted.within(poly_shapes[voronoi_polygon_id]):
                    print("penis")
                    point_stats = [point.x,point.y,points]
                    points_in_polygon[voronoi_polygon_id].append(point_stats)
            
        polygon_id = []

        for i in range(len(poly_shapes)):
            polygon_id.extend([i]*len(points_in_polygon[i]))

        points_in_polygon = [item for sublist in points_in_polygon for item in sublist]

        for i in range(len(points_in_polygon)):
            points_in_polygon[i].append(polygon_id[i])
            
        df_points_in_polygon = pd.DataFrame(points_in_polygon,columns=["point_x","point_y","point_id","polygon_id"])
        # df_points_in_polygon = df_points_in_polygon.reset_index(drop = True)
        df_points_in_polygon.drop_duplicates(subset ="point_id", keep = "last", inplace = True)

        # # if point not assigend to polygon
        # missing_points = self.missing_elements(list(df_points_in_polygon.point_id))

        # if len(missing_points) > 0:
        #     for i in missing_points:
        #         missing_point_x = grid_points[i].x
        #         missing_point_y = grid_points[i].y
        #         point_id = i
        #         point_id_next = i+1
                
        #         polygon_id = df_points_in_polygon[df_points_in_polygon.point_id == point_id_next].polygon_id.to_list()
        #         df_missing_point = pd.DataFrame([[missing_point_x,missing_point_y,point_id,polygon_id[0]]], columns = ["point_x","point_y","point_id","polygon_id"])
        #         df_points_in_polygon = df_points_in_polygon.append(df_missing_point,ignore_index=True)

        df_points_in_polygon = df_points_in_polygon.sort_values(by = ["point_id"])
        print("df_length")
        print(len(df_points_in_polygon))
        print(len(self.env.flatten()))
        df_points_in_polygon["TI_zone"] = self.env.flatten()


        df_points_in_polygon = df_points_in_polygon.sort_index()

        TI_zone_value_all = []
        for voronoi_polygon_id in range(len(poly_shapes)):
            TI_zone_value = df_points_in_polygon[df_points_in_polygon.polygon_id == voronoi_polygon_id]["TI_zone"].value_counts(dropna = False).index
            if len(TI_zone_value) == 0:
                TI_zone_value = TI_zone_value_all[-1]
            else:
                TI_zone_value = TI_zone_value[0].tolist()

            TI_zone_value_all.extend([TI_zone_value]*len(df_points_in_polygon[df_points_in_polygon.polygon_id ==voronoi_polygon_id ]))
        df_points_in_polygon["TI_zone_assigend_to_polygon"] = TI_zone_value_all
        df_polygon_TI_zone = df_points_in_polygon.drop_duplicates(subset ="polygon_id", keep = "last")
        list_polygon_TI_zone = df_polygon_TI_zone.TI_zone_assigend_to_polygon.tolist() 
        # assign polygons to individual agents
        for i,agent in enumerate(self.active_agents):
            agent.update_agent_properties(polygon_id = i,polygon = poly_shapes[i],polygon_area = poly_shapes[i].area,TI_zone_assigned = list_polygon_TI_zone[i])

    def missing_elements(self,L):
        """ check if list of values is throughgoing . return values taht are not """ 
        start, end = L[0], L[-1]
        return sorted(set(range(start, end + 1)).difference(L))

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
