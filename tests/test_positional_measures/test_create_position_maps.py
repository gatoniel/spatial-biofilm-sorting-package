import pytest
import numpy as np
from skimage.draw import polygon
from spatial_biofilm_sorting_package.positional_measures import (
    create_position_maps
)


# create a square of 4x4 in center of 8x8 image
rr, cc = polygon([2, 2, 6, 6], [2, 6, 6, 2])
test_img = np.zeros((8, 8), dtype=bool)
test_img[rr, cc] = True
test_shape = test_img.shape


def test_shapes_and_len():
    flat, img, info = create_position_maps(test_img)
    assert img.ndim == 3
    assert flat.shape[1] == len(info)
    assert flat.shape[1] == img.shape[2]
    assert test_shape[0] == img.shape[0] and test_shape[1] == img.shape[1]
    assert flat.shape[0] == test_img.sum()


def test_square():
    flat, img, info = create_position_maps(test_img)
    for i in range(len(info)):
        assert flat[:, i].max() == img[..., i].max()
        assert img.min() == 0.
    assert img[..., 0].max() == 2.
    assert img[..., 1].max() == np.sqrt((3.5-2.)**2. + (3.5-2.)**2.)
