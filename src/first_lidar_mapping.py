#!/usr/bin/python
import numpy as np
import load_data as ld


def main():
    # load the LIDAR data
    lidar_file = "data/lidar/train_lidar0"
    lidar_list = ld.get_lidar(lidar_file)

    # load the joint angles data
    joint_file = "data/joint/train_joint0"
    joint_dict = ld.get_joint(joint_file)

    # world poses -> the orientation of the body in the world frame at each time-step t
    world_poses = np.load("./outputs/dead_reckoning/world_poses_final.npy")
    print(world_poses.shape)

    print(lidar_list[0].keys())
    print(joint_dict.keys())

    # timestamps for joint angle data
    j_ts = joint_dict["ts"][0]

    j_h = joint_dict["head_angles"]
    # neck angle (yaw) in radians
    j_necks = j_h[0]
    # head angle(pitch) in radians
    j_heads = j_h[1]
    print(len(j_ts))
    print(len(j_necks), len(j_heads))

    # the thetas for the scans in the lidar frame -> radians
    # right to left scan (counterclockwise)
    scan_theta = np.arange(-135, 135.25, 0.25)*np.pi/float(180)
    print(len(lidar_list))

    # lidar origin(position) with respect to body frame
    lidar_orig = np.array([[0.0], [0.0], [0.48]])

    for t in range(1):
        l_ts = lidar_list[t]["t"][0][0]
        scans = lidar_list[t]["scan"][0]
        print(scans.shape, len(scan_theta))

        # Remove scan points too close or too far
        indValid = np.logical_and((scans < 29.9), (scans > 0.1))
        val_scans = scans[indValid]
        val_thets = scan_theta[indValid]

        # LIDAR - Polar to Cartesian
        xl = np.array([val_scans * np.cos(val_thets)])
        yl = np.array([val_scans * np.sin(val_thets)])
        # It's on the XY plane, so z is 0 for all the points
        zl = np.zeros(xl.shape)

        # identify the closest timestamp to LASER scan
        b_arr = (j_ts > l_ts).astype(np.int8)
        id = np.argmax(b_arr) - 1
        # print(id, l_ts, j_ts[id], j_ts[0], j_ts[id + 1])
        yaw = j_necks[id]
        pitch = j_heads[id]

        # First Rz(yaw) is calculated
        yaw_mat = np.array([[np.cos(yaw), -np.sin(yaw), 0.0], [np.sin(yaw), np.cos(yaw), 0.0], [0.0, 0.0, 1.0]])
        pitch_mat = np.array([[np.cos(pitch), 0.0, np.sin(pitch)], [0.0, 1.0, 0.0], [-np.sin(pitch), 0.0, np.cos(pitch)]])
        rot_mat = np.matmul(pitch_mat, yaw_mat)

        # compute xb, yb, zb - Lidar coords in body frame
        print(xl.shape, yl.shape, zl.shape)
        scan_poses = np.vstack((xl, yl))
        scan_poses = np.vstack((scan_poses, zl))
        print(scan_poses.shape)

        # transform the lidar scan to body frame
        rot_scan_poses = np.matmul(rot_mat, scan_poses)
        print(rot_scan_poses.shape)
        scan_body_frame = rot_scan_poses + lidar_orig
        print(scan_body_frame.shape)

        # obtain the pose of the body at time t
        body_pose = world_poses[t, :]
        print(body_pose.shape)

        # position of origin of body in world frame
        wPb = np.array([[body_pose[0]], [body_pose[1]], [0.93]])
        psib = body_pose[2]
        wRb = np.array([[np.cos(psib), -np.sin(psib), 0.0], [np.sin(psib), np.cos(psib), 0.0], [0.0, 0.0, 1.0]])

        # convert the scans from body into the world frame
        rot_body_poses = np.matmul(wRb, scan_body_frame)
        scan_world_frame = rot_body_poses + wPb
        print(scan_world_frame.shape)

        # Remove points hitting/close to floor
        fin_scan_inds = np.where(abs(scan_world_frame[2, :]) > 0.1)
        scan_world_coords = scan_world_frame[:2, fin_scan_inds[0]]
        print(scan_world_coords.shape)


if __name__ == '__main__':
    main()