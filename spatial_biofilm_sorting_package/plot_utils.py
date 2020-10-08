#import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
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


def plot_outline(mask, axes, color=(1, 0, 0), lw=3):
    img_mask = mask.astype(np.uint8)
    contours, _ = cv.findContours(
        img_mask, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE
    )
    contours_ = cv.drawContours(img_mask, contours, -1, (255, 0, 0), lw)
    contours = np.zeros((img_mask.shape[0], img_mask.shape[1], 4))
    for i in range(3):
        contours[contours_ == 255, i] = color[i]
    contours[contours_ == 255, 3] = 1
    axes.imshow(contours)


def custom_legend(
    colors, colorlabels, markers, markerlabels, ax,
    linestyles=[], linestylelabels=[],
):
    custom_lines = [
        Line2D([0], [0], color=color) for color in colors
    ] + [
        Line2D(
            [0], [0], color="gray", marker=marker
        ) for marker in markers
    ] + [
        Line2D(
            [0], [0], color="black", linestyle=linestyle
        ) for linestyle in linestyles
    ]
    custom_descr = colorlabels + markerlabels + linestylelabels
    ax.legend(custom_lines, custom_descr)


def errorbar_alpha(ax, x, y, yerr, color=None, label=None, alpha=0.3):
    line = ax.plot(x, y, color=color, label=label)
    ax.fill_between(x, y-yerr, y+yerr, color=line[0].get_c(), alpha=alpha)
    return line


def identity_plot(axes, x, y, colors, markers):
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            axes.errorbar(
                x[i, j, 0], y[i, j, 0],
                xerr=x[i, j, 1], yerr=y[i, j, 1],
                linestyle="none",
                marker=markers[j],
                markersize=10,
                color=colors[i]
            )
    mins = np.concatenate([
        ids[..., 0] - ids[..., 1] for ids in [x, y]
    ])
    maxs = np.concatenate([
        ids[..., 0] + ids[..., 1] for ids in [x, y]
    ])
    mi_min = np.nanmin(mins)
    mi_max = np.nanmax(maxs)
    axes.plot(
            [mi_min, mi_max],
            [mi_min, mi_max],
            linestyle="--",
            color="gray",
        )


def subplots(y, x, size):
    return plt.subplots(y, x, figsize=[x*size, y*size])
