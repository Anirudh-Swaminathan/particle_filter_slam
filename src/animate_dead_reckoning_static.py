#!/usr/bin/python

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as anim
from matplotlib import style
from occ_gmap import OccGridMap as OGM

# import the data
poses = np.load("./outputs/dead_reckoning/world_poses_final.npy")
poses = poses.tolist()
occ_maps = np.load("./outputs/first_map/occ_map.npy")

# style.use("seaborn-pastel")
style.use('fivethirtyeight')

fig, ax = plt.subplots()
# im = plt.imshow(a)
# ax = plt.axes(xlim=(-1,4), ylim=(-1, 3))

path, = ax.plot([], [], lw=1)
im = ax.imshow(occ_maps[1])
print(occ_maps[1].shape, occ_maps[1].min(), occ_maps.max(), occ_maps.dtype)

maps = OGM()


def init():
    path.set_data([], [])
    return path,


def animate(i):
    data = poses[:100 * (i + 1)]
    x_p = list(list(zip(*data))[0])
    y_p = list(list(zip(*data))[1])
    x_p = np.array(x_p)
    y_p = np.array(y_p)
    x_g, y_g, _ = maps._v_world_to_grid(x_p, y_p, np.zeros(x_p.shape[0]))
    x_g = x_g.tolist()
    y_g = y_g.tolist()
    # im.set_array(occ_maps[1])
    # path.set_data(x_p, y_p)
    ax.clear()
    im = occ_maps[1]
    p = -1 * im + 127.5
    I = np.dstack([p, p, p]).astype(np.int)
    # set path cells as RED
    I[x_g, y_g, :] = [255, 0, 0]
    ax.imshow(I, extent=[0, 800, 0, 800])
    # ax.plot(x_g, y_g, 'r')
    return [ax, ]


# show plot to the user
# ani_show = anim.FuncAnimation(fig, animate, init_func=init, frames=122, interval=2, blit=False, repeat=False)
# plt.show()

# save plot to file
ani = anim.FuncAnimation(fig, animate, init_func=init, frames=122, interval=100, blit=False, repeat=False)
ani.save('./outputs/dead_reckoning/dead_reckon_map_001.mp4', extra_args=['-vcodec', 'libx264'])
# ani.save("./outputs/dead_reckoning/path.mp4", writer=writer)
