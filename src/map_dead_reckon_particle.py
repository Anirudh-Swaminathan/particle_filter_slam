#!/usr/bin/python

# Created by anicodebreaker at 22/02/20
import numpy as np
from scipy.special import expit
from matplotlib import pyplot as plt


from occ_gmap import OccGridMap as OGM
from lidar import LiDAR
from particles import Particles

fig = plt.figure()


def plot_map(parts, mp, t, pth):
    # convert occupancy log odds to probabilities
    p = 1 - expit(mp.grid)
    I = np.dstack([p, p, p])
    plt.imshow(I, extent=[0, mp.grid_size, 0, mp.grid_size])
    plt.title("Occupancy Grid at time: " + str(t))

    # convert particle coordinates to grid coordinates
    x_p = parts[:, 0]
    y_p = parts[:, 1]
    x_g, y_g, _ = mp._v_world_to_grid(x_p, y_p, 0)
    x_g = x_g.tolist()
    y_g = y_g.tolist()

    # scatter current particles on grid
    plt.scatter(x_g, y_g, s=3, color='r')
    plt.savefig(pth + str(t) + ".png")
    print("Saved map as png to file after time:", t)
    plt.show(block=False)


def main():
    # load the LIDAR data
    li = LiDAR()

    # initialize the occupancy grid map
    grid_map = OGM()

    # map save path
    save_pth = "./outputs/map_dead_reckon_particle/occ_maps_dataset0_"

    # initialize 1 particles with dead reckoning first
    particles = Particles(n=1)

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

        # get BEST particle at this time step
        body_pose = particles.get_best_particle()

        scan_world_frame = li.body_to_world(scan_body_frame, body_pose)

        # Remove points hitting/close to floor
        fin_scan_inds = np.where(abs(scan_world_frame[2, :]) > 0.1)
        scan_world_coords = scan_world_frame[:2, fin_scan_inds[0]]

        # update the map using the given scans
        grid_map.update_map(scan_world_coords, body_pose)

        if t % 500 == 0:
            plot_map(particles.poses, grid_map, t, save_pth)

        if t % 2500 == 0:
            grid_map.save_grid(save_pth + str(t) + ".npy")
            print("Saved the occupancy grid at time", t, "to file")
            particles.save_particles(
                save_pth + "particles_" + str(t) + ".npy",
                save_pth + "weights_" + str(t) + ".npy"
            )
            print("Saved the particle positions and weights at time", t, "to file")

    # draw the map
    grid_map.render_map(save_pth + str(len(li)) + ".png")
    grid_map.save_grid(save_pth + str(len(li)) + ".npy")
    print("Saved the Occupancy Grid finally to png and numpy")


if __name__ == "__main__":
    main()
