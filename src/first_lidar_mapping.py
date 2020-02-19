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

    for t in range(1):
        l_ts = lidar_list[t]["t"]
        scans = lidar_list[t]["scan"]


if __name__ == '__main__':
    main()