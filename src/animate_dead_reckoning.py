#!/usr/bin/python

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as anim
from matplotlib import style

data_base_path = "./outputs/dead_reckoning/dataset2/world_poses_final"
# import the data
poses = np.load(data_base_path + ".npy")
poses = poses.tolist()
print(len(poses))

# style.use("seaborn-pastel")
style.use('fivethirtyeight')

fig = plt.figure()
# ax = plt.axes(xlim=(-1,4), ylim=(-1, 3))
ax = plt.axes(xlim=(-20, 20), ylim=(-20, 20))

path, = ax.plot([], [], lw=1)


def init():
    path.set_data([], [])
    return path,


def animate(i):
    data = poses[:2 * (i + 1)]
    # print(type(poses), len(poses), poses[0])
    # print(type(data), len(data), data[-1])
    x_p = list(list(zip(*data))[0])
    y_p = list(list(zip(*data))[1])
    # print(len(x_p), len(y_p), x_p[-1], y_p[-1])
    path.set_data(x_p, y_p)
    return path,


ani = anim.FuncAnimation(fig, animate, init_func=init, frames=int(len(poses) / 2) + 1, interval=2, blit=True,
                         repeat=False)
plt.show()

# save plot to file
# ani = anim.FuncAnimation(fig, animate, frames=int(len(poses) / 2) + 1, interval=100, blit=False, repeat=False)
# ani.save(data_base_path + "vid.mp4", extra_args=['-vcodec', 'libx264'])
# ani.save("./outputs/dead_reckoning/path.mp4", writer=writer)
