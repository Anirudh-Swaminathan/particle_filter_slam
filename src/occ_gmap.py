#!/usr/bin/python
import numpy as np
import p2_utils as pu
import matplotlib.pyplot as plt
from scipy.special import expit


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

        # occupancy delta log odds
        # For now, it is 80%/20% = 4 => log(4)
        self.delta_log = np.log(4)

        # min and max required for map correlation (in meters)
        self.xmin = -20
        self.ymin = -20
        self.xmax = 20
        self.ymax = 20

        # vectorized functions to handle numpy inputs instead of scalar inputs
        self._v_bres = np.vectorize(self._bres)
        self._v_update_free_logs = np.vectorize(self._update_free_logs)
        self._v_world_to_grid = np.vectorize(self.world_to_grid)

    def world_to_grid(self, x, y, theta):
        """
        Convert world fame pose to grid frame pose
        :return:
        """
        # Since grid starts with 0,0 top left, positive y in world is negative x from grid origin
        # xg = self.origin[0] - int(round(y / self.cell_size))
        xg = self.origin[0] + int(round(x / self.cell_size))
        # Since grid starts with 0,0 top left, positive x in world is positive y from grid origin
        # yg = self.origin[1] + int(round(x / self.cell_size))
        yg = self.origin[1] + int(round(y / self.cell_size))
        tg = theta
        return xg, yg, tg

    def _update_occupied_logs(self, x, y):
        """
        INCREASE the log odds for the occupied cells
        :param x: the x coordinate of the cell
        :param y: the y coordinate of the cell
        :return:
        """
        # to avoid very high occupied logs as we progress through the time steps
        # also easy to convert to grayscale -> 255/2 = 127.5
        if self.grid[x, y] < 100.0:
            self.grid[x, y] += self.delta_log

    def _update_free_logs(self, x, y):
        """
        DECREASE the log odds for the free cells
        :param x: the x coordinate of the cell
        :param y: the y coordinate of the cell
        :return:
        """
        # to avoid very low occupied logs as we progress through the time steps
        # also easy to convert to grayscale -> 255/2 = 127.5
        if self.grid[x, y] > -100.0:
            self.grid[x, y] -= self.delta_log

    def _bres(self, sx, sy, ex, ey):
        """
        Function to implement bresenham update
        :param sx: start x
        :param sy: start y
        :param ex: end x
        :param ey: end y
        :return:
        """
        # print(type(sx), type(sy), type(ex), type(ey))
        ray_coords = pu.bresenham2D(sx, sy, ex, ey).astype(np.int64)
        occ_coords = ray_coords[:, -1]
        free_coords = ray_coords[:, :-1]
        # print(occ_coords.dtype, free_coords.dtype)
        # print(occ_coords)

        # update occupied log odds cells
        self._update_occupied_logs(occ_coords[0], occ_coords[1])

        # print(free_coords[0].shape, free_coords[1].shape)
        # update free log odds cells
        self._v_update_free_logs(free_coords[0], free_coords[1])

    def update_map(self, l_scans, r_pose):
        """
        Updates the Occupancy Grid Map with the Laser Scan values
        :param l_scans: Laser Scan Points
        :param r_pose: Robot pose at current time
        :return:
        """
        sx, sy, _ = self.world_to_grid(r_pose[0], r_pose[1], r_pose[2])

        # extract all the x and y coordinates of scan end-points
        lxs = l_scans[0]
        lys = l_scans[1]
        lts = np.zeros(l_scans.shape)

        # convert all the coordinates into grid coordinates
        # exs, eys, _ = self._v_world_to_grid(lxs, lys, lts)
        ans = self._v_world_to_grid(lxs, lys, lts)
        exs = ans[0]
        eys = ans[1]
        ezs = ans[2]
        # print("Printing values now!!")
        # # print(l_scans.shape, lxs.shape, lys.shape)
        # # print(len(ans))
        # # print(exs.shape)
        # # print(eys.shape)
        # # print(ezs.shape)
        # # print(exs[0][:11])
        # # print(exs[0][500:510])
        # # print(exs[0][:-10])
        # print(eys[0][:11])
        # print(eys[0][500:510])
        # print(eys[0][:-10])
        # print(eys[1][:11])
        # print(eys[1][500:510])
        # print(eys[1][:-10])
        # print(ezs[0][:11])
        # print(ezs[0][500:510])
        # print(ezs[0][:-10])
        # print("printed values!!")
        # n = l_scans.shape[1]
        # print(l_scans.shape)
        # chk_arr_x = []
        # chk_arr_y = []

        # for i in range(n):
        #     x, y = l_scans[:, i]
        #     ex, ey, _ = self.world_to_grid(x, y, 0)
        #     chk_arr_x.append(ex)
        #     chk_arr_y.append(ey)
        # chk_arr_x = np.array(chk_arr_x)
        # chk_arr_y = np.array(chk_arr_y)
        # assert(abs(np.sum(exs - chk_arr_x)) <= 1e-5)
        # assert(abs(np.sum(eys - chk_arr_y)) <= 1e-5)

        # call vectorized bresenham function to update the corresponding cell log odds
        self._v_bres(sx, sy, exs[0], eys[0])

    def render_map(self, pth):
        """
        Function to render the map
        :return:
        """
        fig = plt.figure()
        # print(self.grid, self.grid.min(), self.grid.max(), self.grid.dtype, self.grid.shape)
        # plotter = -1 * self.grid + 127.5
        # convert the log odds to probabilities using logistic function
        plotter = 1 - expit(self.grid)
        plt.imshow(plotter, cmap="gray")
        plt.savefig(pth)
        plt.show()

    def save_grid(self, pth):
        """
        save the occupancy grid as numpy array(to be loaded when necessary)
        :param pth: path to save the current grid to
        :return:
        """
        np.save(pth, self.grid)
