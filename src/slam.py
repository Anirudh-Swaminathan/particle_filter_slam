#!/usr/bin/python

# Created by anicodebreaker at February, 23 2020
import numpy as np
from scipy.special import expit
from matplotlib import pyplot as plt

from occ_gmap import OccGridMap as OGM
from lidar import LiDAR
from particles import Particles

fig = plt.figure()


def plot_map(ps, mp, t, pth):
    print("In plot_map()")
    plt.clf()
    # ax1 = fig.add_subplot(111)
    plt.subplot(111)
    # convert occupancy log odds to probabilities
    p = 1 - expit(mp.grid)
    I = np.dstack([p, p, p])
    plt.imshow(I, extent=[0, mp.grid_size, 0, mp.grid_size])
    plt.title("Occupancy Grid at time: " + str(t))

    # convert particle coordinates to grid coordinates
    parts = ps.poses
    print(parts.shape)
    x_p = parts[:, 0]
    y_p = parts[:, 1]

    # convert to pyplot coordinates with 0,0 in the bottom left
    x_g = mp.origin[0] + np.round(x_p / mp.cell_size).astype(np.int)
    y_g = mp.origin[1] + np.round(y_p / mp.cell_size).astype(np.int)
    x_g = x_g.tolist()
    y_g = y_g.tolist()
    print(len(x_g), len(y_g))

    # # plot the best particle's trajectory
    btraj = ps.get_best_path()
    # print(btraj)
    print(len(btraj))
    bt = np.array(btraj)
    x_t = bt[:, 0]
    y_t = bt[:, 1]
    print(x_t)
    print(y_t)
    z_t = bt[:, 2]

    # convert to pyplot coordinates with 0,0 in the bottom left
    x_gt = mp.origin[0] + np.round(x_t / mp.cell_size).astype(np.int)
    y_gt = mp.origin[1] + np.round(y_t / mp.cell_size).astype(np.int)
    x_gt = x_gt.tolist()
    y_gt = y_gt.tolist()
    print(len(x_gt), len(y_gt))

    # plot the path
    plt.plot(x_gt, y_gt, 'b,', zorder=1)

    # plot the current particles
    plt.scatter(x_g, y_g, s=1, color='r')
    plt.savefig(pth + str(t) + ".png")
    print("Saved map as png to file after time:", t)
    plt.show(block=False)


def main():
    print("Starting SLAM on dataset0")
    # load the LIDAR data
    li = LiDAR()

    # initialize the occupancy grid map
    grid_map = OGM()

    # map save path
    save_pth = "./outputs/slam_25parts_noyaw/occ_maps_25parts_dataset0_"

    # initialize 1 particles with dead reckoning first
    particles = Particles(n=25)

    for t in range(len(li)):
        print("Time:", t + 1)

        ### MAPPING!!
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

        # get BEST particle at this time step
        body_pose = particles.get_best_particle()

        scan_world_frame = li.body_to_world(scan_body_frame, body_pose)

        # Remove points hitting/close to floor
        fin_scan_inds = np.where(abs(scan_world_frame[2, :]) > 0.1)
        scan_world_coords = scan_world_frame[:2, fin_scan_inds[0]]

        # update the map using the given scans
        grid_map.update_map(scan_world_coords, body_pose)


        ### PREDICTION!
        delta_p = li.get_delta_pose(t)
        delta_p = delta_p.reshape((1, 3))
        particles.predict(delta_p)


        ### UPDATE STEP!
        # Would get out of bounds at the end(final) step
        if t + 1 == len(li):
            continue

        l_ts = li.get_timestamp(t+1)
        scans = li.get_scans(t+1)

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

        # update step on the particles
        particles.update(scan_body_frame, li, grid_map)

        # resample (if required)
        particles.resample()


        ### Saving stuff
        if t % 500 == 0:
            plot_map(particles, grid_map, t, save_pth)

        if t % 2500 == 0:
            grid_map.save_grid(save_pth + str(t) + ".npy")
            print("Saved the occupancy grid at time", t, "to file")
            particles.save_particles(
                save_pth + "particles_" + str(t) + ".npy",
                save_pth + "weights_" + str(t) + ".npy"
            )
            print("Saved the particle positions and weights at time", t, "to file")
            particles.save_best_path(save_pth + "path_" + str(t) + ".npy")

    # draw the map
    grid_map.render_map(save_pth + str(len(li)) + ".png")
    grid_map.save_grid(save_pth + str(len(li)) + ".npy")
    print("Saved the Occupancy Grid finally to png and numpy")

    # draw final time step with particles
    plot_map(particles, grid_map, len(li), save_pth)
    particles.save_particles(
        save_pth + "particles_" + str(len(li)) + ".npy",
        save_pth + "weights_" + str(len(li)) + ".npy"
    )
    print("Saved the particle positions and weights at the end to file")
    particles.save_best_path(save_pth + "path_" + str(len(li)) + ".npy")
    print("Saved the final best-particle trajectory at the end!!")
    print("\nSLAM DUNK!!!!\n")


if __name__ == "__main__":
    main()
