import pytest
import numpy as np

from spatial_biofilm_sorting_package.binning import bin_array_along_axis


min1, max1 = 0, 10
min2, max2 = -10, 0
test_x = np.stack([
    np.linspace(min1, max1), np.linspace(min2, max2)
], axis=-1)
num_bins = 20


def test_raise():
    with pytest.raises(np.AxisError):
        bin_array_along_axis(test_x, 20, axis=2)


def test_min_max():
    bins, labels = bin_array_along_axis(test_x, 20, axis=1)
    assert bins[0, :].min() == min1
    assert bins[0, :].max() == max1
    assert bins[1, :].min() == min2
    assert bins[1, :].max() == max2


def test_shape():
    bins, labels = bin_array_along_axis(test_x, num_bins, axis=1)
    assert bins.shape[0] == 2
    assert bins.shape[1] == num_bins + 1
    assert labels.shape[0] == 2
    assert labels.shape[1] == num_bins


def test_matching():
    bins, labels = bin_array_along_axis(test_x, num_bins, axis=1)
    for i in range(2):
        assert labels[i, 0] == (bins[i, 0]+bins[i, 1]) / 2
