import numpy as np
import imageio

to8b = lambda x: (255 * np.clip(x, 0, 1)).astype(np.uint8)
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
import matplotlib as mpl
from matplotlib import cm
import cv2

def get_vertical_colorbar(h, vmin, vmax, cmap_name='jet', label=None):
    fig = Figure(figsize=(1.2, 8), dpi=100)
    fig.subplots_adjust(right=1.5)
    canvas = FigureCanvasAgg(fig)
    # Do some plotting.
    ax = fig.add_subplot(111)
    cmap = cm.get_cmap(cmap_name)
    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    tick_cnt = 6
    tick_loc = np.linspace(vmin, vmax, tick_cnt)
    cb1 = mpl.colorbar.ColorbarBase(ax, cmap=cmap,
                                    norm=norm,
                                    ticks=tick_loc,
                                    orientation='vertical')
    tick_label = ['{:3.2f}'.format(x) for x in tick_loc]
    cb1.set_ticklabels(tick_label)
    cb1.ax.tick_params(labelsize=18, rotation=0)
    if label is not None:
        cb1.set_label(label)
    fig.tight_layout()
    canvas.draw()
    s, (width, height) = canvas.print_to_buffer()
    im = np.frombuffer(s, np.uint8).reshape((height, width, 4))
    im = im[:, :, :3].astype(np.float32) / 255.
    if h != im.shape[0]:
        w = int(im.shape[1] / im.shape[0] * h)
        im = cv2.resize(im, (w, h), interpolation=cv2.INTER_AREA)
    return im

def colorize_np(x, cmap_name='jet', mask=None, append_cbar=False):
    HUGE_NUMBER = 1e10
    TINY_NUMBER = 1e-6  
    if mask is not None:
        # vmin, vmax = np.percentile(x[mask], (1, 99))
        vmin = np.min(x[mask])
        vmax = np.max(x[mask])
        vmin = vmin - np.abs(vmin) * 0.01
        x[np.logical_not(mask)] = vmin
        x = np.clip(x, vmin, vmax)
        # print(vmin, vmax)
    else:
        vmin = x.min()
        vmax = x.max() + TINY_NUMBER
    x = (x - vmin) / (vmax - vmin)
    # x = np.clip(x, 0., 1.)
    cmap = cm.get_cmap(cmap_name)
    x_new = cmap(x)[:, :, :3]
    if mask is not None:
        mask = np.float32(mask[:, :, np.newaxis])
        x_new = x_new * mask + np.zeros_like(x_new) * (1. - mask)
    cbar = get_vertical_colorbar(h=x.shape[0], vmin=vmin, vmax=vmax, cmap_name=cmap_name)
    if append_cbar:
        x_new = np.concatenate((x_new, np.zeros_like(x_new[:, :5, :]), cbar), axis=1)
        return x_new
    else:
        return x_new, cbar
    