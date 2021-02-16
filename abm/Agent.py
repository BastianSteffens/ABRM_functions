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
        self.polygon_id = None
        #voronoi polygon shape of agent
        self.polygon = None
        self.polygon_area = None
        #what area TI zone area is most abundant in polygon
        self.TI_zone_assigned = None
        #what TI type (crest, flank etc.) is assigned to agent
        self.TI_type = None
        # which TI from that TI type is assigned to agent
        self.TI_no = None
        # misfit of agent
        self.misfit = None
        #current postition of the agent
        self.pos = pos
        if self.isTracked:
            self.track = None

        self.blocked_turns = 0

    def move_to(self, new_pos):
        """Move the agent to a specified position keeping the track of it if specified"""
        self.pos = new_pos
        self.blocked_turns = 0
     
    def update_agent_properties(self,polygon_id,polygon,polygon_area,TI_zone_assigned):
        """ update agent properties"""
        self.polygon_id = polygon_id
        self.polygon = polygon
        self.polygon_area = polygon_area
        self.TI_zone_assigned = TI_zone_assigned
        
    def update_agent_TI(self,TI_type,TI_no):
        """ update agent TI"""
        self.TI_type = TI_type
        self.TI_no = TI_no

    def update_agent_misfit(self,misfit):
        """ update misfit of all agents for current iteration"""
        self.misfit = misfit
      
    def track_agent(self):
            """ track agent movement etc."""

            if self.isTracked:
                if self.track is None:
                    self.track = [[self.pos[0], self.pos[1], self.pos[2],self.TI_zone_assigned,self.TI_type,self.TI_no,self.misfit]]
                else:
                    self.track.append([self.pos[0], self.pos[1], self.pos[2],self.TI_zone_assigned,self.TI_type,self.TI_no,self.misfit])

    def kill(self, iteration):
        """Kill the agent recordind the iteration when it died (ie. End_Turn)"""
        self.status = 'Dead'
        self.dead_iteration = iteration
 