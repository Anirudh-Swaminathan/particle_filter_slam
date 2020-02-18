#!/usr/bin/python

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as anim
from matplotlib import style

# import the data
poses = np.load("./outputs/dead_reckoning/world_poses.npy")
poses = poses.tolist()

#style.use("seaborn-pastel")
style.use('fivethirtyeight')

fig = plt.figure()
ax = plt.axes(xlim=(-1,4), ylim=(-1, 3))

path, = ax.plot([], [], lw=3)


def init():
    path.set_data([], [])
    return path,


def animate(i):
    data = poses[:2 * (i+1)]
    #print(type(poses), len(poses), poses[0])
    print(type(data), len(data), data[-1])
    x_p = list(list(zip(*data))[0])
    y_p = list(list(zip(*data))[1])
    print(len(x_p), len(y_p), x_p[-1], y_p[-1])
    path.set_data(x_p, y_p)
    return path,


ani = anim.FuncAnimation(fig, animate, init_func=init, frames=6025, interval=5, blit=True, repeat=False)
#plt.show()
ani.save("./outputs/dead_reckoning/path.gif", writer="imagemagick")
