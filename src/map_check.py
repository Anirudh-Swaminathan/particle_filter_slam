#!/usr/bin/python

# Created by anicodebreaker at February, 24 2020
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

    t_list = [0, 2500, 5000, 7500, 10000]
    for t in range(8078, 8079):
        print("\nTime :", t)
        l_ts = li.get_timestamp(t)
        scans = li.get_scans(t)
        print(scans.shape)
        print(scans[:][:11])
        print(scans[:][500:510])
        print(scans[:][-10:])

        xl, yl, zl = li.polar_to_c(scans)
        print("Computed the Cartesian form of the LiDAR coordinates")
        print(xl[:][:11])
        print(xl[:][500:510])
        print(xl[:][-10:])

        print(yl[:][:11])
        print(yl[:][500:510])
        print(yl[:][-10:])

        print(zl[:][:11])
        print(zl[:][500:510])
        print(zl[:][-10:])

        # identify the closest timestamp to LASER scan
        yaw, pitch = li.get_joints(l_ts)
        print("Identified the closest Joint Angles timestamp to the LiDAR timestamp")

        # stack up lidar frame coordinates
        print(xl.shape, yl.shape, zl.shape)
        scan_poses = np.vstack((xl, yl))
        scan_poses = np.vstack((scan_poses, zl))
        print(scan_poses.shape)
        print(scan_poses[:][:11])
        print(scan_poses[:][500:510])
        print(scan_poses[:][-10:])

        # transform scans from lidar frame to head frame
        scan_head_frame = li.lidar_to_head(scan_poses)
        print("LiDAR frame transformed to head frame")
        print(scan_head_frame.shape)
        print(scan_head_frame[0][:11])
        print(scan_head_frame[0][500:510])
        print(scan_head_frame[0][-10:])
        print(scan_head_frame[0].min(), scan_head_frame[0].max())
        print(scan_head_frame[1][:11])
        print(scan_head_frame[1][500:510])
        print(scan_head_frame[1][-10:])
        print(scan_head_frame[1].min(), scan_head_frame[1].max())
        print(scan_head_frame[2][:11])
        print(scan_head_frame[2][500:510])
        print(scan_head_frame[2][-10:])
        print(scan_head_frame[2].min(), scan_head_frame[2].max())

        # transform from head frame to body frame
        scan_body_frame = li.head_to_body(scan_head_frame, yaw, pitch)
        print("Head frame transformed to body frame")
        print(scan_body_frame.shape)
        print(scan_body_frame[0][:11])
        print(scan_body_frame[0][500:510])
        print(scan_body_frame[0][-10:])
        print(scan_body_frame[0].min(), scan_body_frame[0].max())
        print(scan_body_frame[1][:11])
        print(scan_body_frame[1][500:510])
        print(scan_body_frame[1][-10:])
        print(scan_body_frame[1].min(), scan_body_frame[1].max())
        print(scan_body_frame[2][:11])
        print(scan_body_frame[2][500:510])
        print(scan_body_frame[2][-10:])
        print(scan_body_frame[2].min(), scan_body_frame[2].max())

        # obtain the pose of the body at time t
        body_pose = world_poses[t, :]
        print(body_pose.shape)

        scan_world_frame = li.body_to_world(scan_body_frame, body_pose)
        print("Body frame transformed to World Frame")
        print(scan_world_frame.shape)
        print(scan_world_frame[0][:11])
        print(scan_world_frame[0][500:510])
        print(scan_world_frame[0][-10:])
        print(scan_world_frame[0].min(), scan_world_frame[0].max())
        print(scan_world_frame[1][:11])
        print(scan_world_frame[1][500:510])
        print(scan_world_frame[1][-10:])
        print(scan_world_frame[1].min(), scan_world_frame[1].max())
        print(scan_world_frame[2][:11])
        print(scan_world_frame[2][500:510])
        print(scan_world_frame[2][-10:])
        print(scan_world_frame[2].min(), scan_world_frame[2].max())

        # Remove points hitting/close to floor
        fin_scan_inds = np.where(scan_world_frame[2, :] > 0.1)
        # fin_scan_inds = np.logical_and(True, (scan_world_frame[2] > 0.1))
        scan_world_coords = scan_world_frame[:2, fin_scan_inds[0]]
        print(scan_world_coords.shape)
        print("Removed scan points hitting on/close to the floor, and moved to the 2D frame in meters!")

        print(scan_world_coords[0][:11])
        print(scan_world_coords[0][500:510])
        print(scan_world_coords[0][-10:])
        print(scan_world_coords[0].min(), scan_world_coords[0].max())
        print(scan_world_coords[1][:11])
        print(scan_world_coords[1][500:510])
        print(scan_world_coords[1][-10:])
        print(scan_world_coords[1].min(), scan_world_coords[1].max())

        print("Calling update_map()")
        grid_map.update_map(scan_world_coords, body_pose)
    grid_map.render_map("test_map.png")


if __name__ == '__main__':
    main()