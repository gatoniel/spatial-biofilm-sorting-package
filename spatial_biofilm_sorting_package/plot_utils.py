import numpy as np
import matplotlib.colors as colors


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
