#Some part of the code are adapted from mesa  --> https://github.com/projectmesa/mesa/blob/master/mesa/space.py
#Adaptation from 2D grid to 3D

from typing import Any, Dict, Iterable, Iterator, List, Optional, Set, Tuple, Union
import itertools
import random as rn
import numpy as np

from Agent import Agent


Coordinate = Tuple[int, int, int]

class Grid:
    """ Base class for a 3D grid."""

    def __init__(self, X: int, Y: int, Z: int) -> None:
        """ Create a new grid.
        Args:
            X, Y, Z: The 3 dimensions of the grid
            torus: Boolean whether the grid wraps or not.
        """
        self.X = X
        self.Y = Y
        self.Z = Z

        self.grid = np.zeros((self.X, self.Y, self.Z))*self.default_val()

        # # Add all cells to the empties list making a distinction between layers for generation
        self.empties_layers = [set(itertools.product(*(range(self.X), range(self.Y), [i]))) for i in range (self.Z)]

        self.layer = self.Z-1

    @staticmethod
    def default_val() -> int:
        """ Default value for empty cell elements. """
        return 0

    @staticmethod
    def agent_val() -> int:
        """ Default value for agent cell elements. """
        return 1

    # @staticmethod
    # def barrier_val() -> int:
    #     """ Default value for barrier cell elements. """
    #     return 2

    # def populate_grid_with_barriers(self, barrier_pos) -> None:
    #     print('Populating grid with barriers...')
    #     self.grid[barrier_pos[:,0], barrier_pos[:,1], barrier_pos[:,2]] = self.barrier_val()
    #     for [x, y, z] in barrier_pos:
    #         self.empties_layers[z].discard((x,y,z))
    
    def iter_empty_neighborhood(
        self,
        pos: Coordinate,
        # include_down: bool = False,
        # include_center: bool = False,
        radius: int = 1,
    ) -> Iterator[Coordinate]:
        """ Return an iterator over empty cell coordinates that are in the
        neighborhood of a certain point.
        Args:
            pos: Coordinate tuple for the neighborhood to get.
            include_down: If True, include down neighborhood
                   If False, exclude down neighborhood
            include_center: If True, return the (x, y, z) cell as well.
                            Otherwise, return surrounding cells only.
            radius: radius, in cells, of neighborhood to get.
        Returns:
            An iterator of coordinate tuples representing the empty neighborhood.
        """
        x, y, z = pos
        # if not include_down:
        #     zlim = 0
        # else:
        #     zlim = radius
        zlim = 0
        for dz in range(-radius, zlim + 1):
            for dy in range(-radius, radius + 1):
                for dx in range(-radius, radius + 1):
                    if dx == 0 and dy == 0 and dz == 0:# and not include_center:
                        continue
                    # Skip if new coords out of bounds.
                    if (not (0 <= dx + x < self.X) or not (0 <= dy + y < self.Y) or not (0 <= dz + z < self.Z)):
                        continue
                    coords = ( dx + x, dy + y, dz + z)
                    if self.is_cell_empty(coords):
                        yield coords
    
    def get_empty_neighborhood(
        self,
        pos: Coordinate,
        n_active_agents,
        # include_down: bool = False,
        include_center: bool = True,
        remove_agent: bool = True,
        radius: int = 1,
    ) -> List[Coordinate]:
        """ Return a list of coordinates of empty cells that are in the neighborhood of a
        certain point.
        Args:
            pos: Coordinate tuple for the neighborhood to get.
            include_down: If True, include down neighborhood
                   If False, exclude down neighborhood
            include_center: If True, return the (x, y) cell as well.
                            Otherwise, return surrounding cells only.
            radius: radius, in cells, of neighborhood to get.
        Returns:
            A list of coordinate tuples representing the neighborhood.
        """
        if include_center == True:
            free_neighbours = list(self.iter_empty_neighborhood(pos, radius))
            free_neighbours.append(pos)
        else:
            free_neighbours = list(self.iter_empty_neighborhood(pos, radius))
        if remove_agent == True:
            if n_active_agents >5: # voronoi tesselation does not work with less than 5 agents.
                free_neighbours.append((None,None,None))

        return free_neighbours

    def is_cell_empty(self, pos: Coordinate) -> bool:
        """ Returns a bool of the contents of a cell. """
        x, y, z = pos
        return self.grid[x, y, z] == self.default_val()
        
    def move_agent(self, agent: Agent, pos: Coordinate) -> None:
        """
        Move an agent from its current position to a new position.
        Args:
            agent: Agent object to move. Assumed to have its current location
                   stored in a 'pos' tuple.
            pos: Tuple of new position to move the agent to.
        """
        self._remove_agent(agent.pos)

        self._place_agent(pos)
        agent.move_to(pos)

    def _remove_agent(self, pos: Coordinate) -> None:
        """ Remove the agent from the given location. """
        x, y, z = pos
        self.grid[x, y, z] = self.default_val()
        self.empties_layers[z].add((x,y,z))


    def _place_agent(self, pos: Coordinate) -> None:
        """ Place the agent at the correct location. """
        if pos != (None,None,None):
            if self.is_cell_empty(pos):
                x, y, z = pos
                self.grid[x, y, z] = self.agent_val()
                self.empties_layers[z].discard((x,y,z))
            else:
                raise Exception("Cell not empty")
        else:
            print("agent removed from grid")



    def initiate_agent_on_grid(self, agent: Agent) -> None:
        """ Position an agent on the grid. To be used for new generated agents """
        pos = agent.pos
        if self.is_cell_empty(pos):
            x, y, z = pos
            self.grid[x, y, z] = self.agent_val()
        else:
            raise Exception("Cell not empty")

    def find_empty_location(self) -> Union[Coordinate, str]:
        """ Pick a random empty cell. """
        not_empty = False
        for z in range(self.Z):
            if len(self.empties_layers[self.layer]) > 0:
                not_empty = True
                break
        if not_empty:
            z = rn.randint(int(0.4*self.Z), self.Z-1)
            while len(self.empties_layers[z]) == 0:
                z = rn.randint(0, self.Z-1)
            pos = self.empties_layers[z].pop()
            return pos
        else:
            print("Warning: No empty cells")
            return 'No Empty Cells'

    def exists_empty_cells(self) -> bool:
        """ Return True if any cells empty else False. """
        return np.any(self.grid == self.default_val())

    def exists_empty_cells_layer(self, layer: int = 0) -> bool:
        """ Return True if any cells empty else False. """
        return np.any(self.grid[:, :, layer] == self.default_val())