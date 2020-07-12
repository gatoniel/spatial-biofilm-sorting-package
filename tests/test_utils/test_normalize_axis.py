import pytest
import numpy as np
from spatial_biofilm_sorting_package.utils import normalize_axis


def test_raise():
    with pytest.raises(np.AxisError):
        normalize_axis(np.zeros((2, 2)), 2)


def test_min_max():
    a = np.stack([
        np.arange(-5, 5),
        np.arange(-6, 4),
    ], axis=0)
    a_norm = normalize_axis(a, 0)
    assert a_norm.min() == 0.
    assert a_norm.max() == 1.


def test_linspace():
    a = np.stack([
        np.arange(-5, 5),
        np.arange(-6, 4),
    ], axis=0)
    a_norm = normalize_axis(a, 0)
    b = np.stack([
        np.linspace(0, 1, 10),
        np.linspace(0, 1, 10),
    ], axis=0)
    print(b.shape)
    print(a.shape)
    print(a_norm.shape)
    print(a_norm.dtype)
    print(b.dtype)
    print(a_norm)
    print(b)
    print(a_norm-b)
    assert np.allclose(b, a_norm)
