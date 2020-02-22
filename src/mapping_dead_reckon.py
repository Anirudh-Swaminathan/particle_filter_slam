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

    for t in range(len(li)):
        l_ts = li.get_timestamp(t)
        scans = li.get_scans(t)

        xl, yl, zl = li.polar_to_c(scans)

        # identify the closest timestamp to LASER scan
        yaw, pitch = li.get_joints(l_ts)

        # stack up lidar frame coordinates
        scan_poses = np.vstack((xl, yl))
        scan_poses = np.vstack((scan_poses, zl))

        # transform scans from lidar frame to head frame
        scan_head_frame = li.lidar_to_head(scan_poses)

        # transform from head frame to body frame
        scan_body_frame = li.head_to_body(scan_head_frame, yaw, pitch)

        # obtain the pose of the body at time t
        body_pose = world_poses[t, :]

        scan_world_frame = li.body_to_world(scan_body_frame, body_pose)

        # Remove points hitting/close to floor
        fin_scan_inds = np.where(abs(scan_world_frame[2, :]) > 0.1)
        scan_world_coords = scan_world_frame[:2, fin_scan_inds[0]]

        # update the map using the given scans
        grid_map.update_map(scan_world_coords, body_pose)

    # map save path
    save_map = "./outputs/mapping_dead_reckon/occ_maps_dataset0.npy"
    save_img = "./outputs/mapping_dead_reckon/occ_maps_dataset0.png"

    # draw the map
    grid_map.render_map(save_img)

    # save the map to file to draw later on
    grid_map.save_history(save_map)


if __name__ == '__main__':
    main()
