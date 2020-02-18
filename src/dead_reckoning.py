#!/usr/bin/python
import load_data as ld
import numpy as np


def body_to_world(del_x, del_y, world_theta):
    """
    Function to convert the odometry information given in body frame to world frame coordinates
    :param del_x: the forward value for body
    :param del_y: the left value for body
    :param world_pose: the world pose
    :return: the world frame del_xw, del_yw
    """
    transform_mat = np.array([[np.cos(world_theta), -1*np.sin(world_theta)], [np.sin(world_theta), np.cos(world_theta)]])
    del_world = np.matmul(transform_mat, np.array([del_x, del_y]).reshape((2, 1)))
    del_xw = del_world[0][0]
    del_yw = del_world[1][0]
    return del_xw, del_yw


def main():
    lidar_file = "data/lidar/train_lidar0"
    lidar_list = ld.get_lidar(lidar_file)

    # initial world frame pose
    poses = [[] for _ in range(len(lidar_list) + 1)]
    cur_pose = [0, 0, 0]
    #poses.append(cur_pose)
    poses[0].extend(cur_pose)

    for t in range(len(lidar_list)):
        dels = lidar_list[t]['delta_pose']
        del_xb, del_yb, del_t = dels[0]
        del_xw, del_yw = body_to_world(del_xb, del_yb, cur_pose[2])
        cur_pose[0] += del_xw
        cur_pose[1] += del_yw
        cur_pose[2] += del_t
        poses[t + 1].extend(cur_pose)
        if t % 1000 == 0:
            print("\nCurrent time:", t)
            print("Read data")
            print(dels, type(dels), len(dels))
            print("Body frame")
            print(del_xb, del_yb, del_t)
            print("World Frame")
            print(del_xw, del_yw)
            print("Current robot position")
            print(cur_pose, len(cur_pose))
            print("First 3 poses")
            print(poses[:3])
            print("Previous 3 poses")
            print(poses[-3:], len(poses))

    # save the calculated poses to file
    world_poses = np.array(poses)
    save_pth = "./outputs/dead_reckoning/world_poses"
    np.save(save_pth + ".npy", world_poses)
    with open(save_pth + ".txt", "w") as f:
        f.writelines("%s\n" % pose for pose in poses)


if __name__ == '__main__':
    main()
