#!/usr/bin/python
import numpy as np
import load_data as ld
from occ_gmap import OccGridMap as OGM
from lidar import LiDAR


def main():
    # load the LIDAR data
    li = LiDAR()

    # world poses -> the orientation of the body in the world frame at each time-step t
    world_poses = np.load("./outputs/dead_reckoning/world_poses_final.npy")
    print(world_poses.shape)

    # initialize the occupancy grid map
    grid_map = OGM()

    for t in range(1):
        l_ts = li.get_timestamp(t)
        scans = li.get_scans(t)
        print(scans.shape)

        xl, yl, zl = li.polar_to_c(scans)
        print("Computed the Cartesian form of the LiDAR coordinates")

        # identify the closest timestamp to LASER scan
        yaw, pitch = li.get_joints(l_ts)
        print("Identified the closest Joint Angles timestamp to the LiDAR timestamp")

        # stack up lidar frame coordinates
        print(xl.shape, yl.shape, zl.shape)
        scan_poses = np.vstack((xl, yl))
        scan_poses = np.vstack((scan_poses, zl))
        print(scan_poses.shape)

        # transform scans from lidar frame to head frame
        scan_head_frame = li.lidar_to_head(scan_poses)
        print(scan_head_frame.shape)
        print("LiDAR frame transformed to head frame")

        # transform from head frame to body frame
        scan_body_frame = li.head_to_body(scan_head_frame, yaw, pitch)
        print(scan_body_frame.shape)
        print("Head frame transformed to body frame")

        # obtain the pose of the body at time t
        body_pose = world_poses[t, :]
        print(body_pose.shape)

        scan_world_frame = li.body_to_world(scan_body_frame, body_pose)
        print(scan_world_frame.shape)
        print("Body frame transformed to World Frame")

        # Remove points hitting/close to floor
        fin_scan_inds = np.where(abs(scan_world_frame[2, :]) > 0.1)
        scan_world_coords = scan_world_frame[:2, fin_scan_inds[0]]
        print(scan_world_coords.shape)
        print("Removed scan points hitting on/close to the floor, and moved to the 2D frame in meters!")

        # update the map using the given scans
        grid_map.update_map(scan_world_coords, body_pose)

    # draw the map
    grid_map.render_map()

    # save the map to file to draw later on
    grid_map.save_history()


if __name__ == '__main__':
    main()