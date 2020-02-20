#!/usr/bin/python
import numpy as np
import p2_utils as pu


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
        self.grid_dims = 40

        # grid size is now calculated
        self.grid_size = int(self.grid_dims / self.cell_size)

        # set map(world frame) origin
        self.origin = (int(self.grid_size / 2), int(self.grid_size / 2))

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
        xg = int(round(x / self.cell_size)) + self.origin[0]
        yg = int(round(y / self.cell_size)) + self.origin[1]
        tg = theta
        return xg, yg, tg

    def bres(self, sx, sy, ex, ey):
        """
        Function to implement bresenham update
        :param sx: start x
        :param sy: start y
        :param ex: end x
        :param ey: end y
        :return:
        """
        ray_coords = pu.bresenham2D(sx, sy, ex, ey)
        occ_coords = ray_coords[:, -1]
        free_coords = ray_coords[:, :-1]

    def update_map(self, l_scans, r_pose):
        """
        Updates the Occupancy Grid Map with the Laser Scan values
        :param l_scans: Laser Scan Points
        :param r_pose: Robot pose at current time
        :return:
        """
        # save the old map
        self.history.append(self.grid)
        sx, sy, _ = self.world_to_grid(r_pose[0], r_pose[1], r_pose[2])
        n_scans = l_scans.shape[1]
        for s in range(n_scans):
            x, y = l_scans[:, s]
            ex, ey, _ = self.world_to_grid(x, y, 0)
            ray_coords = pu.bresenham2D(sx, sy, ex, ey)
            print(ray_coords.shape)
            occ_coords = ray_coords[:, -1]
            free_coords = ray_coords[:, :-1]
            print(occ_coords.shape, free_coords.shape)

    def save_history(self):
        """
        Save the map for every state so far
        :return:
        """
        # append the final grid, which may/may not have been appended
        self.history.append(self.grid)
        np.save(self.save_path, np.array(self.history))

    def render_map(self):
        """
        Function to render the map
        :return:
        """
        pass
