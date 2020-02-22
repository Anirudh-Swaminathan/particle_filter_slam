#!/usr/bin/python
import numpy as np
import load_data as ld
from occ_gmap import OccGridMap as OGM
from lidar import LiDAR
from scipy.special import expit
from matplotlib import pyplot as plt


fig = plt.figure()


def plot_map(poses, mp, t, pth):
    data = poses
    x_p = list(list(zip(*data))[0])
    y_p = list(list(zip(*data))[1])
    x_p = np.array(x_p)
    y_p = np.array(y_p)
    x_g, y_g, _ = mp._v_world_to_grid(x_p, y_p, np.zeros(x_p.shape[0]))
    x_g = x_g.tolist()
    y_g = y_g.tolist()
    # im.set_array(occ_maps[1])
    # path.set_data(x_p, y_p)

    # convert occupancy log odds to probabilities
    p = 1 - expit(mp.grid)
    I = np.dstack([p, p, p])
    # set path cells as RED
    I[x_g, y_g, :] = [1.0, 0.0, 0.0]
    plt.imshow(I, extent=[0, mp.grid_size, 0, mp.grid_size])
    plt.title("Occupancy Grid at time: " + str(t))
    # ax.plot(x_g, y_g, 'r')
    plt.savefig(pth + str(t) + ".png")
    plt.show(block=False)


def main():
    # load the LIDAR data
    li = LiDAR()

    # world poses -> the orientation of the body in the world frame at each time-step t
    world_poses = np.load("./outputs/dead_reckoning/world_poses_final.npy")
    print(world_poses.shape)

    # initialize the occupancy grid map
    grid_map = OGM()

    # map save path
    save_img = "./outputs/mapping_dead_reckon/occ_maps_dataset0_"

    for t in range(len(li)):
        print("Time:", t + 1)
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

        if t % 500 == 0:
            plot_map(world_poses[:(t+1)], grid_map, t, save_img)

    # draw the map
    grid_map.render_map(save_img + str(len(li)) + ".png")


if __name__ == '__main__':
    main()
