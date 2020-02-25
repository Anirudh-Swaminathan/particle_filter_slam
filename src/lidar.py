#!/usr/bin/python
import load_data as ld
import numpy as np


class LiDAR(object):
    """
    A class to implement LiDAR class for encapsulation
    """
    def __init__(self):
        """
        Constuctor for the LiDAR class
        """
        self.lidar_path = "data/lidar/train_lidar0"
        self.joint_path = "data/joint/train_joint0"

        # load the data and keep it for now
        self.lidar_list = ld.get_lidar(self.lidar_path)
        self.joint_dict = ld.get_joint(self.joint_path)

        # timestamps for joint angles
        self.j_ts = self.joint_dict["ts"][0]

        # neck angles and head angles extract
        self.j_h = self.joint_dict["head_angles"]
        self.j_necks = self.j_h[0]
        self.j_heads = self.j_h[1]

        # the scan angles for LiDAR
        self.scan_theta = np.arange(-135, 135.25, 0.25)*np.pi/float(180)

        # LiDAR origin(position) wrt head frame
        self.hPl = np.array([[0.0], [0.0], [0.15]])
        self.bPh = np.array([[0.0], [0.0], [0.33]])

    def __len__(self):
        """
        length of LiDAR data
        :return:
        """
        return len(self.lidar_list)

    def get_timestamp(self, t):
        """
        Returns the lidar timestamp at time t
        :param t: t time
        :return:
        """
        return self.lidar_list[t]["t"][0][0]

    def get_scans(self, t):
        """
        Return scan at time t
        :param t: t time
        :return:
        """
        return self.lidar_list[t]["scan"][0]

    def polar_to_c(self, scans):
        """
        returns cartesian coordinates of LiDAR scans
        :param scans: LiDAR scans at time t
        :param thets: angles for the scans
        :return:
        """
        # Remove scan points too close or too far
        indValid = np.logical_and((scans < 29.9), (scans > 0.1))
        val_scans = np.copy(scans[indValid])
        val_thets = np.copy(self.scan_theta)
        val_thets = val_thets[indValid]

        # LIDAR - Polar to Cartesian
        #xl = np.array([val_scans * np.cos(val_thets)])
        #yl = np.array([val_scans * np.sin(val_thets)])
        xl = val_scans * np.cos(val_thets)
        yl = val_scans * np.sin(val_thets)
        # It's on the XY plane, so z is 0 for all the points
        zl = np.zeros(xl.shape)
        return xl, yl, zl

    def get_joints(self, ts):
        """
        Funtion to get the joint angles with closest timestamp to input timestamp
        :param ts: timestamp of LiDAR
        :return:
        """
        # identify the closest timestamp to LASER scan
        b_arr = (self.j_ts > ts).astype(np.int8)
        id = np.argmax(b_arr) - 1
        # print("Index is", id)
        # print(id, l_ts, j_ts[id], j_ts[0], j_ts[id + 1])
        yaw = self.j_necks[id]
        pitch = self.j_heads[id]
        return yaw, pitch

    def lidar_to_head(self, lidar_coords):
        """
        Function to convert lidar frame coordinates to head frame coordinates
        :param lidar_coords: lidar frame coordinates -> 2D ndarray with x, y and z
        :return:
        """
        return lidar_coords + self.hPl

    def head_to_body(self, head_coords, yaw, pitch):
        """
        Convert from the head to the body frame
        :param head_coords: the coordinates of scan in head frame
        :param yaw: neck angle -> applied first
        :param pitch: head angle -> applied next
        :return:
        """
        # First Rz(yaw) is calculated
        yaw_mat = np.array([[np.cos(yaw), -np.sin(yaw), 0.0], [np.sin(yaw), np.cos(yaw), 0.0], [0.0, 0.0, 1.0]])
        pitch_mat = np.array(
            [[np.cos(pitch), 0.0, np.sin(pitch)], [0.0, 1.0, 0.0], [-np.sin(pitch), 0.0, np.cos(pitch)]])
        # bRh = np.matmul(pitch_mat, yaw_mat)
        bRh = np.matmul(yaw_mat, pitch_mat)

        # transform from head frame to body frame
        rot_scan_poses = np.matmul(bRh, head_coords)
        scan_body_frame = rot_scan_poses + self.bPh
        return scan_body_frame

    def body_to_world(self, body_coords, body_pose):
        """
        Convert the LiDAR scan from body coordinates to world frame coordinates
        :param body_coords: LiDAR scan body frame coordinates
        :param body_pose: coordinates of the body in world frame
        :return:
        """
        # position of origin of body in world frame
        wPb = np.array([[body_pose[0]], [body_pose[1]], [0.93]])
        psib = body_pose[2]
        wRb = np.array([[np.cos(psib), -np.sin(psib), 0.0], [np.sin(psib), np.cos(psib), 0.0], [0.0, 0.0, 1.0]])

        # convert the scans from body into the world frame
        rot_body_poses = np.matmul(wRb, body_coords)
        scan_world_frame = rot_body_poses + wPb
        return scan_world_frame

    def get_delta_pose(self, t):
        """
        Get's the delta_pose of odometry at index t
        :param t: index t of lidar scan
        :return: delta_pose -> the odometry world frame delta_pose
        """
        dels = self.lidar_list[t]["delta_pose"]
        return dels[0]
