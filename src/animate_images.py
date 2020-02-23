#!/usr/bin/python

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as anim
from matplotlib import style
from occ_gmap import OccGridMap as OGM
from scipy.special import expit

# import the data
img_base_path = "./outputs/map_predict/occ_maps_dataset0_"

# style.use("seaborn-pastel")
style.use('fivethirtyeight')

fig, ax = plt.subplots()
# im = plt.imshow(a)
# ax = plt.axes(xlim=(-1,4), ylim=(-1, 3))
plt.tight_layout()

maps = OGM()


def center_crop(ima):
    """
    Center crop the input image
    :param ima: image input
    :return:
    """
    xs = ima.shape[0]
    ys = ima.shape[1]
    ret = ima[int(xs/4):int(3*xs/4), int(ys/4):int(3*ys/4), :]
    return ret


def animate(i):
    im_path = img_base_path + str(500 * i) + ".png"
    if i == 25:
        im_path = img_base_path + str(12048) + ".png"
    img = plt.imread(im_path)
    ax.clear()

    # center crop
    ima = center_crop(img)

    ax.imshow(ima)
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])
    return [ax, ]


# show plot to the user
# ani_show = anim.FuncAnimation(fig, animate, frames=26, interval=500, blit=False, repeat=False)
# plt.show()

# save plot to file
ani = anim.FuncAnimation(fig, animate, frames=26, interval=500, blit=False, repeat=False)
ani.save('./outputs/map_predict/dataset0_25frames_4parts.mp4', extra_args=['-vcodec', 'libx264'])
# ani.save("./outputs/dead_reckoning/path.mp4", writer=writer)
