#!/usr/bin/python

# Created by anicodebreaker at 22/02/20
import numpy as np


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
        self.predict_noise = np.array([0.005, 0.005, 0.0001])

        # store best particles trajectory(list of numpy arrays)
        self.best_traj = []

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
        print("\nIn dead_reckon_move()")
        # obtain the previous best particle and record its trajectory
        best_part = self.get_best_particle()
        self.best_traj.append(best_part[:])

        self.poses += delta_pose.reshape((1, 3))
        print(delta_pose, self.poses, self.poses.shape, best_part, len(self.best_traj))

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
        print("In get_best_path() of particle class")
        print(len(self.best_traj))
        # append the most recent pose and return
        ret = list(self.best_traj)
        best_part = self.get_best_particle()
        ret.append(best_part)
        print(len(self.best_traj))
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
