import numpy as np
import matplotlib.colors as colors

from matplotlib.lines import Line2D


def plot_overlay(img, mask, axes, img_cmap, mask_alpha, mask_color):
    color = colors.colorConverter.to_rgba(mask_color)
    cmap = colors.LinearSegmentedColormap.from_list(
        'overlay', [color, color], 256
    )
    cmap._init()

    alphas = np.linspace(0, mask_alpha, cmap.N+3)
    cmap._lut[:, -1] = alphas
    axes.imshow(img, cmap=img_cmap)
    axes.imshow(mask, cmap=cmap)


def custom_legend(colors, colorlabels, markers, markerlabels, ax):
    custom_lines = [
        Line2D([0], [0], color=color) for color in colors
    ] + [
        Line2D(
            [0], [0], color="gray", marker=marker
        ) for marker in markers
    ]
    custom_descr = colorlabels + markerlabels
    ax.legend(custom_lines, custom_descr)


def errorbar_alpha(ax, x, y, yerr, color=None, label=None, alpha=0.3):
    ax.plot(x, y, color=color, label=label)
    ax.fill_between(x, y-yerr, y+yerr, color=color, alpha=alpha)
