#!/usr/bin/python

# Created by anicodebreaker at 22/02/20
import numpy as np
from scipy.special import expit
from scipy.special import softmax
import p2_utils as pu
from matplotlib import pyplot as plt


class Particles(object):
    """
    Particle Class to implement particles
    """
    def __init__(self, n):
        """
        Constructor for particles
        """
        self.num_particles = n
        self.poses = None
        self.weights = None
        self._init_particles()

        # standard deviations for x, y and yaw
        self.predict_noise = np.array([0.025, 0.025, 0.01])

        # store best particles trajectory(list of numpy arrays)
        self.best_traj = []

        # set the threshold for resampling
        # this is 20% of the initial number of particles
        self.Nthresh = 0.35 * self.num_particles

    def __len__(self):
        return self.num_particles

    def _init_particles(self):
        """
        Initializes particle weights and their positions to [0, 0, 0]
        :return:
        """
        self.poses = np.array([[0.0, 0.0, 0.0] for p in range(self.num_particles)])
        self.weights = np.array([1/self.num_particles for p in range(self.num_particles)])

    def dead_reckon_move(self, delta_pose):
        """
        Perform deterministic predict step, i.e., simply add delta_pose
        :param delta_pose: odometry delta pose in world frame
        :return:
        """
        # print("\nIn dead_reckon_move()")
        # obtain the previous best particle and record its trajectory
        best_part = self.get_best_particle()
        self.best_traj.append(np.copy(best_part))

        self.poses += delta_pose.reshape((1, 3))
        # print(delta_pose, self.poses, self.poses.shape, best_part, len(self.best_traj))

    def predict(self, delta_pose):
        """
        Noisy predict with fixed Gaussian Noise in x, y and yaw
        :param delta_pose: the odometry delta poses
        :return:
        """
        # obtain the previous best particle and record its trajectory
        best_part = self.get_best_particle()
        self.best_traj.append(np.copy(best_part))

        # generate noise (num_particles, 1)
        dx = np.random.normal(0.0, self.predict_noise[0], (self.num_particles, 1))
        dy = np.random.normal(0.0, self.predict_noise[1], (self.num_particles, 1))
        dt = np.random.normal(0.0, self.predict_noise[2], (self.num_particles, 1))

        # stack them together
        dp = np.hstack((dx, dy))
        dp = np.hstack((dp, dt))

        # add noise to delta_pose
        add_pose = dp + delta_pose.reshape((1, 3))
        self.poses += add_pose

    def update(self, scan_body_frame, li, mp, t):
        """
        UPDATE step for Particle Filter using LASER Scan Matching
        :param scan_body_frame: lidar scan in body frame
        :param li: reference to LiDAR class object
        :param mp: reference to OccupancyGridMap object
        :return:
        """
        # setup map coordinates in world frame
        x_im = np.arange(mp.xmin, mp.xmax + mp.cell_size, mp.cell_size)
        y_im = np.arange(mp.xmin, mp.xmax + mp.cell_size, mp.cell_size)

        # setup the neighborhood around current particle to calculate correlation for
        x_range = np.arange(-0.2, 0.2 + 0.05, 0.05)
        y_range = np.arange(-0.2, 0.2 + 0.05, 0.05)

        # binarize map
        # bin_map = (expit(mp.grid) > 0.5).astype(np.int)
        # bin_map = (mp.grid > mp.delta_log).astype(np.int)
        bin_map = (mp.grid > 0).astype(np.int)
        # print(bin_map.shape, bin_map.min(), bin_map.max(), np.sum(bin_map))
        max_correlations = []

        # find the map correlation for each particle
        for i in range(self.num_particles):
            p = self.poses[i]
            # convert scans to world frame for given particle
            scan_world_frame = li.body_to_world(scan_body_frame, p)

            # Remove points hitting/close to floor
            fin_scan_inds = np.where(scan_world_frame[2, :] > 0.1)
            scan_world_coords = scan_world_frame[:3, fin_scan_inds[0]]

            # call mapCorrelation
            c = pu.mapCorrelation(bin_map, x_im, y_im, scan_world_coords, x_range, y_range)
            # if t % 10 == 0:
            #     c_cpy = np.copy(c)
            #     c_cpy = c_cpy / np.sum(c_cpy)
            #     print("Correlations for particle: ", p, "is as follows")
            #     print(c.shape)
            #     print(c)
            #     arm = np.argmax(c)
            #     arm_i = np.unravel_index(arm, c.shape)
            #     print(arm, arm_i, c[arm_i[0]][arm_i[1]])
            #     print(p.shape)
            #     print(p)
            #     new_p = np.copy(p)
            #
            #     # this is for the grid back to world frame
            #     new_p[1] -= ((arm_i[0] - 4) * 0.05)
            #     new_p[0] += ((arm_i[1] - 4) * 0.05)
            #     print("Updating x, y of pose would set it to:", new_p)
            #     plt.imshow(c_cpy, cmap="gray")
            #     plt.savefig("./outputs/ani_correctcorr/corr_parts_b.png")
            #     plt.show()
            # c = pu.mapCorrelation(mp.grid, x_im, y_im, scan_world_coords, x_range, y_range)
            mc = c.max()
            c_cpy = np.copy(c)
            arm = np.argmax(c_cpy)
            arm_i = np.unravel_index(arm, c_cpy.shape)
            # print("Maximum correlation is", mc, "for particle at index ", arm_i)

            # Update the pose of this particle to it's max correlated value index
            new_p = np.copy(p)
            print(new_p.shape, new_p[0].shape, new_p[1].shape)

            # this is for the grid back to world frame
            new_p[1] -= ((arm_i[0] - 4) * 0.05)
            new_p[0] += ((arm_i[1] - 4) * 0.05)
            self.poses[i] = np.copy(new_p)
            max_correlations.append(np.copy(mc))
        max_cs = np.array(max_correlations)
        # print("Current max correlations are :")
        # print(max_cs[:5], max_cs[-5:])
        # print(max_cs.shape, max_cs.min(), max_cs.max())

        # observation model from laser correlation
        obs_model = softmax(max_cs - np.max(max_cs))
        # print(obs_model.shape)
        assert(np.sum(abs(obs_model - softmax(max_cs))) <= 1e-6)
        # print("Observation model is:")
        # print(obs_model[:5], obs_model[-5:])
        # print(obs_model.shape, obs_model.min(), obs_model.max())

        assert(obs_model.shape == self.weights.shape)
        # print("Old weights:")
        # print(self.weights[:5], self.weights[-5:])
        # print(self.weights.shape, self.weights.min(), self.weights.max(), len(np.unique(self.weights)))
        # particle filter update equation
        # numer = self.weights * obs_model
        numer = np.multiply(self.weights, obs_model)
        self.weights = numer / np.sum(numer)
        # print("New Weights!:")
        # print(self.weights[:5], self.weights[-5:])
        # print(self.weights.shape, self.weights.min(), self.weights.max(), len(np.unique(self.weights)))

    def resample(self):
        Neff = 1.0 / np.sum(self.weights ** 2)
        print("Neff is :", Neff)
        if Neff > self.Nthresh:
            return
        print("RESAMPLING particles!")
        # print(len(np.unique(self.weights)))

        # print("Old weights:")
        # print(self.weights[:5], self.weights[-5:])
        # print(self.weights.shape, self.weights.min(), self.weights.max(), len(np.unique(self.weights)))

        # print("Old poses:")
        # print(self.poses[:5], self.poses[-5:])
        # print(self.poses.shape, self.poses.min(), self.poses.max(), len(np.unique(self.poses)))

        # Use Sample Importance Resampling
        part_inds = np.random.choice(np.arange(self.num_particles), self.num_particles, p=self.weights)
        new_parts = self.poses[part_inds]
        self.poses = np.copy(new_parts)
        self.weights = np.array([1/self.num_particles for p in range(self.num_particles)])
        # print(len(np.unique(self.weights)))
        # print("New Weights!:")
        # print(self.weights[:5], self.weights[-5:])
        # print(self.weights.shape, self.weights.min(), self.weights.max(), len(np.unique(self.weights)))

        # print("New poses:")
        # print(self.poses[:5], self.poses[-5:])
        # print(self.poses.shape, self.poses.min(), self.poses.max(), len(np.unique(self.poses)))
        print("RESAMPLING DONE!!")

    def get_best_particle(self):
        """
        Return the best particle at this time step t
        :return: best_particle - the best particle at given time step
        """
        # find the index of particle with largest weight(probability)
        ind = np.argmax(self.weights)
        best_particle = self.poses[ind, :]
        return best_particle

    def save_particles(self, p_pth, w_pth):
        """
        Save the particles to the specified paths when invoked
        :param p_pth: path to save the particle positions
        :param w_pth: path to save the particle weights
        :return:
        """
        np.save(p_pth, self.poses)
        np.save(w_pth, self.weights)

    def get_best_path(self):
        """
        Return the trajectory of the best particle
        :return:
        """
        # print("In get_best_path() of particle class")
        # print(len(self.best_traj))
        # append the most recent pose and return
        ret = list(self.best_traj)
        best_part = self.get_best_particle()
        ret.append(best_part)
        # print(len(self.best_traj), len(ret))
        return ret

    def save_best_path(self, pth):
        """
        Save the best particle trajectory
        :param pth: path to save the best path to
        :return:
        """
        best_traj = self.get_best_path()
        bp = np.array(best_traj)
        np.save(pth, bp)
