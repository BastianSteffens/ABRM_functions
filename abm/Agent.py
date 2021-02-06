class Agent():
    """Object modeling agent movement in the reservoir model"""

    def __init__( self, agentID, pos, iteration, isTracked=True):
        self.id = agentID
        self.status = 'Alive'
        self.isTracked = isTracked
        #starting position of the agent
        (self.start_x, self.start_y, self.start_z) = pos
        #save the iteration the agent is created
        self.start_iteration = iteration
        #id of voronoi polygon belonging to agent
        self.polygon_id = -1
        #voronoi polygon shape of agent
        self.polygon = -1
        self.polygon_area = -1
        #what area TI zone area is most abundant in polygon
        self.TI_zone_assigned = -1
        #what TI type (crest, flank etc.) is assigned to agent
        self.TI_type = -1
        # which TI from that TI type is assigned to agent
        self.TI_no = -1
        #current postition of the agent
        self.pos = pos
        if self.isTracked:
            self.track = [[pos[0], pos[1], pos[2]],self.polygon_id,self.polygon,self.polygon_area,self.TI_zone_assigned,self.TI_type,self.TI_no]
        self.blocked_turns = 0

    def move_to(self, new_pos):
        """Move the agent to a specified position keeping the track of it if specified"""
        self.pos = new_pos
        self.blocked_turns = 0
        # if self.isTracked:
            # self.track.append([new_pos[0], new_pos[1], new_pos[2],self.polygon_id,self.polygon,self.polygon_area,self.TI_zone_assigned,self.TI_type,self.TI_no])
    
    def update_agent_properties(self,polygon_id,polygon,polygon_area,TI_zone_assigned):
        """ update agent properties"""
        self.polygon_id = polygon_id
        self.polygon = polygon
        self.polygon_area = polygon_area
        self.TI_zone_assigned = TI_zone_assigned
        # self.TI_type = TI_type
        # self.TI_no = TI_no
        # if self.isTracked:
            # self.track.append([self.pos[0], self.pos[1], self.pos[2],polygon_id,polygon,polygon_area,TI_zone_assigned,self.TI_type,self.TI_no])
    
    def update_agent_TI(self,TI_type,TI_no):
        """ update agent TI"""
        self.TI_type = TI_type
        self.TI_no = TI_no
        if self.isTracked:
            self.track.append([self.pos[0], self.pos[1], self.pos[2],self.polygon_id,self.polygon,self.polygon_area,self.TI_zone_assigned,TI_type,TI_no])


    def kill(self, iteration):
        """Kill the agent recordind the iteration when it died (ie. End_Turn)"""
        self.status = 'Dead'
        self.dead_iteration = iteration
 