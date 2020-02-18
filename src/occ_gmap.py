#!/usr/bin/python
import numpy as np


class OccGridMap(object):
    """
    A class to implement the Occupancy Grid Map
    """

    def __init__(self):
        """
        Constructor for the class
        """
        # set each cells size to 5 cm
        self.cell_size = 0.05

        # the total side length the grid should span in meters
        self.grid_dims = 20

        # grid size is now calculated
        self.grid_size = int(self.grid_dims / self.cell_size)

        # set map(world frame) origin
        self.origin = (200, 200)

        # initialize the grid map to 0's
        self.grid = np.zeros((self.grid_size, self.grid_size))

        # maintain a history of grid maps
        # append map to history before each update to map
        self.history = []

        # path to save the occupancy grid map to
        self.save_path = "./outputs/dead_reckoning/occ_map.npy"

    def world_to_grid(self, x, y, theta):
        """
        Convert world fame pose to grid frame pose
        :return:
        """
        xg = x / self.cell_size + self.origin[0]
        yg = y / self.cell_size + self.origin[1]
        tg = theta
        return xg, yg, tg

    def save_history(self):
        """
        Save the map for every state so far
        :return:
        """
        # append the final grid, which may/may not have been appended
        self.history.append(self.grid)
        np.save(self.save_path, np.array(self.history))
